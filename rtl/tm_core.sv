// TARS-mini core (contract v0.3): one NPU core that runs a descriptor queue from DDR.
//
// The visible behaviour is the contract's: per descriptor, every check happens before any write;
// GEMM / Conv are INT8 x INT8 -> exact INT32 (systolic array) -> fp32 epilogue in the contract's
// order; activations and row ops (softmax, LayerNorm, RMSNorm) follow the contract's fixed fp32
// sequences and 16-lane order; an error stops the queue with ERR_CODE and OP_IDX.
//
// Microarchitecture: an R x C output-stationary array; 16-byte beats. A/W K-chunks stream in
// through a loader that keeps up to OT reads in flight (the port returns them in order) into NB
// chunk buffers; the feed advances only when every row it touches has its chunk, and a whole-
// array enable freezes the array otherwise. Conv is an implicit GEMM: the loader generates the
// im2col addresses and fills padding with zero beats. Each PE has two accumulator banks: a tile
// accumulates in one while the epilogue engine (row-parallel, C lanes, scale/bias latched per
// tile) turns the previous tile's bank into output rows, beside the next tile's feed. A segment
// writer turns rows into byte-strobed write beats. Reuse: one operand's row block is held on chip
// (HB beats) across the inner tile loop, the loop order chosen per descriptor. Row ops run in the
// row engine: pass 0 streams the row from DDR into the hold buffer, later passes read VG beats per
// cycle from it; when two rows fit, the next row's pass 0 overlaps the current row's later passes.
// Nothing here may change a result bit: the golden model has no R, C, NB, OT, HB, VG, latency or
// stall parameter.
module tm_core
  import tm_fp_pkg::*;
#(
  parameter int R         = 8,
  parameter int C         = 8,
  parameter longint MEM_BYTES = 64'h40_0000,
  parameter int CORE_ID   = 0,
  parameter int NB        = 8,            // K-chunk buffers (power of two, NB * 16 >= max(R, C), NB >= LG)
  parameter int OT        = 128,          // loader reads in flight (power of two): DRAM latency x bandwidth
  parameter int LG        = 4,            // loader: consecutive K-chunks of a row per visit (one 64 B burst)
  parameter int HB        = 1024,         // hold buffer, 16-byte beats (16 KB)
  parameter int VG        = 4,            // row ops: groups (beats) per cycle in hold-buffer passes (1, 2, 4)
  parameter int OQ        = 128,          // output queue slots between the pipelined lanes and the writer (>= the longest lane, GELU 123-127 cycles, to keep one launch per cycle)
  parameter int VQMIN     = 32,           // row reader: at least this many reads in flight (power of two)
  parameter int GB        = 8,            // GEMV: rows at most (a GEMM with M <= GB runs on the extension unit)
  parameter int RW        = 1,            // read port: 1 (16-byte beats), 2 or 4 (also wide 16 RW-byte requests)
  parameter int ARR       = 1             // 0: no systolic array (the KV260 build): every GEMM with K <= 4096
                                          // runs on the GEMV unit, GB rows (fewer if K / 16 < GB) a pass;
                                          // a Conv or a GEMM with K > 4096 is rejected (E_OPCODE)
) (
  input  logic          clk,
  input  logic          rst,
  input  logic          start,
  input  logic [63:0]   q_addr,
  input  logic [31:0]   q_len,
  output logic          halted,
  output logic          done,
  output logic [3:0]    err_code,
  output logic [31:0]   op_idx,
  output logic [63:0]   cycles,
  // read port: a request is accepted when mr_ready; data returns later on mr_rvalid, in request
  // order, any number of requests may be in flight
  output logic          mr_req,
  output logic [63:0]   mr_addr,
  output logic          mr_wide,          // RW > 1: a 16 RW-byte request (address aligned to it): mr_rhi has its upper 16 (RW - 1) bytes
  input  logic          mr_ready,
  input  logic          mr_rvalid,
  input  logic [127:0]  mr_rdata,
  input  logic [128*((RW > 1) ? RW - 1 : 1)-1:0] mr_rhi,
  // write port: request held until accepted (mw_ready); the write is done when accepted
  output logic          mw_req,
  output logic [63:0]   mw_addr,
  output logic [127:0]  mw_data,
  output logic [15:0]   mw_strb,
  input  logic          mw_ready
);
  // ------------------------------------------------------------------ contract constants
  localparam logic [15:0] MAGIC = 16'hAA55, VERSION = 16'h0002;
  localparam logic [15:0] OP_GEMM = 16'h0002, OP_RELU = 16'h0003, OP_ROW = 16'h0004,
                          OP_CONV = 16'h0001, OP_CONV_ACT = 16'h0005,
                          OP_ELT = 16'h0006, OP_ROPE = 16'h0007, OP_FDOT = 16'h0008;   // v0.4
  localparam logic [3:0] E_OK = 0, E_MAGIC = 1, E_VERSION = 2, E_CRC = 3, E_VALID = 4,
                         E_OPCODE = 5, E_FIELD = 6, E_ALIGN = 7, E_RANGE = 8;
  localparam int SK    = (R > C) ? R : C;  // skew depth
  localparam int NP    = 2 * SK;           // loader slot positions per chunk: A0 W0 A1 W1 ...
  localparam int LB    = (NB > 1) ? $clog2(NB) : 1;
  localparam int LO    = $clog2(OT);
  localparam int LP    = $clog2(NP + 1);
  localparam int VB    = C / 4 + 2;        // beats a C-column fp32 vector can span
  localparam int RVB   = R / 4 + 2;        // beats an R-row fp32 vector can span (v0.4 row scales)
  localparam int RB    = (4 * C > 16 * VG) ? 4 * C : 16 * VG;   // bytes of one output segment at most
  localparam int VQ    = (OT > VQMIN) ? OT : VQMIN;   // row-op reader: reads in flight (latency x bandwidth)
  localparam int KR    = HB / SK;          // K-chunks a held row can keep
  localparam int LGE   = (LG < NB) ? LG : NB;   // chunks per row per loader visit (at most NB)
`ifdef TM_FPGA
  // one fpu_act for the core (tm_sfu.sv): a long-act step of W lanes goes through it one lane a
  // cycle, a step at most every W cycles, W - 1 cycles more latency
  localparam int SFU   = 1;
`else
  localparam int SFU   = 0;
`endif
  logic [31:0] xp_sx, sfu_y;               // the extension unit's SFU input; the SFU's result
  logic [31:0] xs_la [16], xs_lb [16], xs_fl [16], xs_rx, f_sum;   // the extension unit on the row engine's units
  initial begin
    if ((1 << $clog2(NB)) != NB || NB < 2 || NB * 16 < SK) $fatal(1, "bad NB");
    if (LG < 1) $fatal(1, "bad LG");
    if ((1 << LO) != OT) $fatal(1, "bad OT");
    if (HB < SK) $fatal(1, "bad HB");
    if (SK < 2) $fatal(1, "R or C must be at least 2");   // the loader's tile start takes >= 2 cycles
    if (VG != 1 && VG != 2 && VG != 4) $fatal(1, "bad VG");
  end

  typedef enum logic [5:0] {
    S_IDLE, S_NEXT, S_FETCH, S_FETCH_GOT, S_CRC, S_CHKW, S_CHK4, S_CHECK,
    S_G_TILE, S_G_CLR, S_G_FEED,
    S_G_EPI, S_G_TILE_END,
    S_V_GB, S_V_ROW, S_V_PASS, S_V_FOLD, S_V_NEXT, S_W_RUN, S_V_INV, S_X_RUN,
    S_RD_REQ, S_RD_WAIT, S_HALT
  } state_t;

  state_t state, ret;

  // ------------------------------------------------------------------ queue and descriptor
  logic [63:0]  qa;
  logic [31:0]  qn, qi;
  logic [63:0]  q_daddr;                  // qa + 64 qi (kept beside qi)
  logic [511:0] desc;
  logic [1:0]   beat;
  logic [127:0] rd_data;
  logic [63:0]  rd_addr;

  wire [15:0] d_magic  = desc[511:496];
  wire [15:0] d_ver    = desc[495:480];
  wire [7:0]  d_core   = desc[479:472];
  wire [23:0] d_hrsv   = desc[471:448];
  wire [15:0] d_op     = desc[447:432];
  wire [15:0] d_ctrl   = desc[431:416];
  wire [15:0] d_group  = desc[415:400];
  wire [15:0] d_epi    = desc[399:384];
  wire [63:0] d_base   = desc[383:320];
  wire [31:0] d_aoff   = desc[319:288];
  wire [31:0] d_woff   = desc[287:256];
  wire [31:0] d_yoff   = desc[255:224];
  wire [31:0] d_soff   = desc[223:192];
  wire [31:0] d_boff   = desc[191:160];
  wire [15:0] d_m      = desc[159:144];
  wire [15:0] d_n      = desc[143:128];
  wire [15:0] d_k      = desc[127:112];
  wire [15:0] d_lda16  = desc[111:96];
  wire [31:0] d_oscale = desc[95:64];
  wire [31:0] d_crc    = desc[63:32];
  wire        d_valid  = desc[31];
  wire [14:0] d_ldy16  = desc[30:16];
  wire [15:0] d_dep    = desc[15:0];

  // ------------------------------------------------------------------ CRC-32/ISO-HDLC
  logic [31:0] crc;
  logic [6:0]  cb;
  function automatic logic [31:0] crc_byte(input logic [31:0] c_in, input logic [7:0] b);
    logic [31:0] c;
    c = c_in ^ {24'b0, b};
    for (int i = 0; i < 8; i++) c = c[0] ? ((c >> 1) ^ 32'hEDB88320) : (c >> 1);
    return c;
  endfunction
  wire [7:0] crc_in = (cb >= 4 && cb <= 7) ? 8'h00 : desc[8 * cb[5:0] +: 8];

  // ------------------------------------------------------------------ checks (contract order)
  logic [3:0] chk, r_chk;
  logic [7:0] cv, cv_first;
  logic [2:0] f_qm;
  logic       f_sym, f_fus, f_gemm, f_bias, f_row, f_norm, f_conv;
  logic       f_gm;                        // GEMM or Conv (the matrix path)
  logic [31:0] f_gpr;                      // ARR 0: GEMV rows a pass
  logic       f_elt, f_rope, f_fdot, f_v4, f_quant, f_x, f_soft, f_hdr_ok, f_grp_ok;   // v0.4
  logic       f_gv;                        // GEMM M = 1 on the extension unit (GEMV)
  logic [31:0] f_brow, f_ldb;              // v0.4: B row bytes and stride (ELT dense B, FDOT)
  logic [31:0] g_h, g_w, g_kh, g_kw, g_s, g_p, g_cin, g_oh, g_ow;
  logic [65:0] hi_g;
  logic [3:0] f_adt, f_wdt;
  logic [2:0] f_act;
  logic [1:0] f_out;
  logic [31:0] f_arow, f_yrow, f_lda, f_ldy, f_es, f_ns;
  logic [65:0] hi_a, hi_w, hi_s, hi_b, hi_y;
  // The check's and the dispatch's arithmetic (1 GHz: no wide multiply or divide in one cycle)
  // runs in a sequencer beside the CRC (64 cycles): stride divisions, three 17-step restoring
  // dividers (K / (KH KW), the tile counts M / R and N / C), one shift-add multiplier taking 4
  // bits of its second operand per cycle for the products, then the range ends and dispatch
  // values, all registered; S_CHK4 evaluates the rules on them, S_CHECK dispatches.
  logic [31:0] r_kk, r_hw, r_oh, r_ow, r_cin, r_krem;
  logic [47:0] r_pa, r_py, r_pnk;
  logic [31:0] r_pohw;
  logic [47:0] r_pwl;                     // W * lda (conv: one image row)
  logic [65:0] r_hia, r_hiy, r_hiw, r_his, r_hib, r_hig;
  logic [65:0] r_hix, r_hic, r_hiq, r_hir;  // v0.4: B (ELT / ROPE table / FDOT), C, QUANT scales, row scales
  logic [63:0] r_pb, r_ptab;               // v0.4: (rows of B - 1) ldb, ROPE table bytes
  logic [65:0] r_bx;                       // v0.4: B's extent past W_OFF
  logic [2:0]  ck_st;                     // 0 idle, 1 small, 2 strides, 3 divide, 4 lda, 5 products, 6 sums, 7 done
  logic [4:0]  ck_i;
  logic [3:0]  ck_j;
  logic        ck_ld;
  logic [31:0] r_ohn, r_own;
  logic        r_ohv, r_owv;
  logic [16:0] dv_r [6], dv_q [6], dv_d [6], dv_s [6];
  logic [31:0] r_lda, r_ldy, r_arow;
  logic [63:0] mu_a, mu_p;
  logic [31:0] mu_b;
  logic [63:0] r_rlda, r_ck, r_rldy, r_slda, r_swlda, r_kwl, r_pp, r_d1, r_d2, r_porig;
  logic        r_fits, r_ov, r_gbfit, r_wfit, r_minner;
  logic [31:0] r_gb0;


  always_comb begin
    f_qm   = d_ctrl[2:0];
    f_sym  = d_ctrl[3];
    f_adt  = d_ctrl[7:4];
    f_wdt  = d_ctrl[11:8];
    f_fus  = d_ctrl[12];
    f_act  = {d_epi[5], d_epi[1:0]};      // v0.3: EPI[5] extends the activation code
    f_out  = d_epi[3:2];
    f_bias = d_epi[4];
    f_gemm = (d_op == OP_GEMM);
    f_conv = (d_op == OP_CONV || d_op == OP_CONV_ACT);
    f_gm   = f_gemm || f_conv;
    // v0.3 conv geometry: GROUP_SIZE = W; HDR_RSV = H | KH | KW | STRIDE-1 | PAD
    g_h  = 32'(d_hrsv[23:12]); g_kh = 32'(d_hrsv[11:8]); g_kw = 32'(d_hrsv[7:4]);
    g_s  = 32'(d_hrsv[3:2]) + 32'd1; g_p = 32'(d_hrsv[1:0]); g_w = 32'(d_group);
    g_cin = r_cin;                          // registered (check sequencer)
    g_oh  = r_oh;
    g_ow  = r_ow;
    f_row  = (d_op == OP_ROW);
    f_norm = f_row && d_ctrl[1:0] != 2'd0 && !d_ctrl[2];   // LayerNorm, RMSNorm, L2NORM (gamma)
    f_soft = f_row && d_ctrl[2:0] == 3'd0;
    f_quant = f_row && d_ctrl[2:0] == 3'd4;
    f_elt  = (d_op == OP_ELT); f_rope = (d_op == OP_ROPE); f_fdot = (d_op == OP_FDOT);
    f_v4   = f_elt || f_rope || f_fdot;
    // GEMV: a GEMM with M <= GB rows and K <= 4096 (x fits the unit's 256-beat buffers) runs on
    // the extension unit, the weights streaming once at the read port's rate for all M rows
    // (decode; a prefill chunk); M > 1 needs K / 16 >= M (a channel's M rows launch one a cycle
    // within its beats) (microarchitecture only)
    f_gpr  = (32'(d_k >> 4) < GB) ? ((d_k >> 4) == 16'd0 ? 32'd1 : 32'(d_k >> 4)) : 32'(GB);
    f_gv   = (ARR == 0) ? f_gemm && d_m != 16'd0 && d_k <= 16'd4096
                        : f_gemm && d_m != 16'd0 && 32'(d_m) <= GB && d_k <= 16'd4096 && (d_m == 16'd1 || 32'(d_k >> 4) >= 32'(d_m));
    f_x    = f_v4 || f_quant || f_gv;      // the extension unit's
    f_hdr_ok = f_conv || (f_gemm && f_qm == 3'd3) || (f_elt && d_hrsv < 24'd8);
    f_grp_ok = f_conv || f_v4;
    f_brow = f_fdot ? (d_ctrl[0] ? 32'(d_n) : 32'(d_k)) << 2 : (f_elt && d_hrsv[1:0] == 2'd0) ? 32'(d_n) << 2 : 32'd0;
    f_ldb  = (d_group != 0) ? 32'(d_group) << 4 : f_brow;
    f_es   = (f_out == 2'd0) ? 32'd4 : (f_out == 2'd1) ? 32'd2 : (f_out == 2'd2) ? 32'd1 : 32'd4;
    f_arow = f_gemm ? 32'(d_k) : f_conv ? g_cin : f_fdot ? 32'(d_k) * 32'd4 : 32'(d_n) * 32'd4;
    f_yrow = 32'(d_n) << ((f_out == 2'd1) ? 1 : (f_out == 2'd2) ? 0 : 2);
    f_lda  = (d_lda16 != 0) ? 32'(d_lda16) * 32'd16 : f_arow;
    f_ldy  = (d_ldy16 != 0) ? 32'(d_ldy16) * 32'd16 : f_yrow;
    f_ns   = (f_qm == 3'd1) ? 32'd1 : 32'(d_n);
    hi_a = r_hia; hi_y = r_hiy; hi_w = r_hiw; hi_s = r_his; hi_b = r_hib; hi_g = r_hig;   // registered (check sequencer)

    // the rules in the contract's order, evaluated in parallel; the first that holds decides
    cv[0]  = d_magic != MAGIC;
    cv[1]  = d_ver != VERSION;
    cv[2]  = ~crc != d_crc;
    cv[3]  = !d_valid;
    cv[4]  = (d_op != OP_GEMM && d_op != OP_RELU && d_op != OP_ROW && !f_conv && !f_v4) ||
             (ARR == 0 && (f_conv || (f_gemm && d_k > 16'd4096)));
    cv[5]  = d_core != 8'(CORE_ID) || d_dep != 0 || (d_hrsv != 0 && !f_hdr_ok) || (d_group != 0 && !f_grp_ok) ||
             d_ctrl[15:13] != 0 || d_epi[15:6] != 0 || f_out == 2'd3 ||
             (f_out != 2'd2 && d_oscale != 0) ||
             (f_gm && (f_qm == 3'd0 || f_qm > 3'd3 || (f_conv && f_qm == 3'd3) || !f_sym || f_adt != 4'd1 ||
                       f_wdt != 4'd1 || ((f_act != 3'd0) != f_fus))) ||
             (d_op == OP_CONV_ACT && f_act == 3'd0) ||
             (f_conv && (g_h == 0 || g_w == 0 || g_kh == 0 || g_kw == 0 || g_oh == 0 || g_ow == 0 ||
                         32'(d_m) != r_pohw)) ||
             (d_op == OP_RELU && (d_ctrl != 0 || f_bias || d_k != 0 || d_woff != 0 || d_soff != 0 ||
                                  d_boff != 0 || f_act == 3'd0)) ||
             (f_row && (d_ctrl[15:3] != 0 || d_ctrl[2:0] > 3'd4 || f_act != 3'd0 || (d_k != 0 && !f_soft))) ||
             (f_soft && (f_bias || d_woff != 0 || d_soff != 0 || d_boff != 0)) ||
             (f_quant && (f_bias || d_woff != 0 || d_boff != 0 || f_out != 2'd2 || d_oscale != 0)) ||
             (f_row && !f_bias && d_boff != 0) ||
             (f_elt && (d_ctrl[15:2] != 0 || d_k != 0 || d_soff != 0 || f_bias != (d_ctrl[1:0] == 2'd2) ||
                        (d_ctrl[1:0] == 2'd2 && f_out != 2'd0) || (!f_bias && d_boff != 0))) ||
             (f_rope && (d_ctrl != 0 || f_act != 3'd0 || f_bias || d_soff != 0 || d_boff != 0 || d_group == 0)) ||
             (f_fdot && (d_ctrl[15:1] != 0 || f_act != 3'd0 || f_bias || d_soff != 0 || d_boff != 0));
    cv[6]  = d_m == 0 || d_n == 0 || d_n[3:0] != 0 || ((f_gm || f_fdot) && (d_k == 0 || d_k[3:0] != 0)) ||
             (f_rope && (d_k == 0 || d_k[4:0] != 0 || d_k > d_n)) ||
             (f_conv && (r_krem != 0 || g_cin[3:0] != 0)) ||
             d_aoff[3:0] != 0 || (!f_row && (f_elt && d_hrsv[1] ? d_woff[1:0] != 0 : d_woff[3:0] != 0)) ||
             d_yoff[3:0] != 0 || d_soff[3:0] != 0 ||
             d_boff[3:0] != 0 || f_lda < f_arow || f_ldy < f_yrow || ((f_elt || f_fdot) && f_ldb < f_brow);
    cv[7]  = hi_a > 66'(MEM_BYTES) || hi_y > 66'(MEM_BYTES) ||
             (f_gm && (hi_w > 66'(MEM_BYTES) || hi_s > 66'(MEM_BYTES) ||
                       (f_bias && hi_b > 66'(MEM_BYTES)))) ||
             (f_norm && (hi_g > 66'(MEM_BYTES) || (f_bias && hi_b > 66'(MEM_BYTES)))) ||
             (f_v4 && r_hix > 66'(MEM_BYTES)) || (f_elt && f_bias && r_hic > 66'(MEM_BYTES)) ||
             (f_quant && r_hiq > 66'(MEM_BYTES)) || (f_gemm && f_qm == 3'd3 && r_hir > 66'(MEM_BYTES));
    cv_first = cv & (~cv + 8'd1);
    chk = cv_first[0] ? E_MAGIC : 4'd0;
    chk = chk | (cv_first[1] ? E_VERSION : 4'd0) | (cv_first[2] ? E_CRC : 4'd0) |
          (cv_first[3] ? E_VALID : 4'd0) | (cv_first[4] ? E_OPCODE : 4'd0) |
          (cv_first[5] ? E_FIELD : 4'd0) | (cv_first[6] ? E_ALIGN : 4'd0) | (cv_first[7] ? E_RANGE : 4'd0);
  end


  // the check sequencer: starts as the CRC does (the descriptor is complete), done in < 64 cycles
  always_ff @(posedge clk) begin
    if (rst) ck_st <= 3'd0;
    else if (state == S_FETCH_GOT && beat == 2'd3) begin
      ck_st <= 3'd1; r_cin <= 0; r_krem <= 0;
    end else case (ck_st)
      3'd1: begin                             // small products, the stride numerators
        r_kk  <= g_kh * g_kw;                 // 4 x 4 bits
        r_ohv <= g_h + 2 * g_p >= g_kh; r_ohn <= g_h + 2 * g_p - g_kh;
        r_owv <= g_w + 2 * g_p >= g_kw; r_own <= g_w + 2 * g_p - g_kw;
        ck_st <= 3'd2;
      end
      3'd2: begin
        // dividers: K / (KH KW) (conv), ceil(M / R), ceil(N / C), the output size (H + 2P - KH) / S
        dv_d[0] <= 17'(d_k);                       dv_s[0] <= 17'(r_kk);
        dv_d[1] <= 17'(d_m) + 17'(R - 1);          dv_s[1] <= 17'(R);
        dv_d[2] <= 17'(d_n) + 17'(C - 1);          dv_s[2] <= 17'(C);
        dv_d[3] <= 17'(r_ohn);                     dv_s[3] <= 17'(g_s);
        dv_d[4] <= 17'(r_own);                     dv_s[4] <= 17'(g_s);
        dv_d[5] <= 17'(d_m) - 17'd1;               dv_s[5] <= 17'(d_group);   // v0.4 ROPE: (M - 1) / G
        dv_r <= '{default: 17'd0}; dv_q <= '{default: 17'd0};
        ck_i <= 5'd17;
        ck_st <= 3'd3;
      end
      3'd3: begin                             // restoring division, a bit per cycle
        for (int d = 0; d < 6; d++) begin
          logic [17:0] t;
          t = {dv_r[d], dv_d[d][ck_i - 5'd1]};
          if (t >= 18'(dv_s[d])) begin dv_r[d] <= 17'(t - 18'(dv_s[d])); dv_q[d] <= {dv_q[d][15:0], 1'b1}; end
          else begin dv_r[d] <= 17'(t); dv_q[d] <= {dv_q[d][15:0], 1'b0}; end
        end
        if (ck_i == 5'd1) ck_st <= 3'd4;
        else ck_i <= ck_i - 5'd1;
      end
      3'd4: begin                             // K / (KH KW) known: Cin, then the strides
        if (f_conv && g_kh != 0 && g_kw != 0) begin r_cin <= 32'(dv_q[0]); r_krem <= 32'(dv_r[0]); end
        r_minner <= dv_q[1] > dv_q[2];
        r_oh <= r_ohv ? 32'(dv_q[3]) + 32'd1 : 32'd0;
        r_ow <= r_owv ? 32'(dv_q[4]) + 32'd1 : 32'd0;
        ck_st <= 3'd5; ck_j <= 0; ck_ld <= 1'b1;
      end
      3'd5: begin                             // the products: job ck_j, 4 bits of mu_b per cycle
        if (ck_ld) begin
          if (ck_j == 0) begin r_lda <= f_lda; r_ldy <= f_ldy; r_arow <= f_arow; end
          case (ck_j)
            4'd0:  begin mu_a <= 64'(g_w);  mu_b <= g_h; end
            4'd1:  begin mu_a <= 64'(r_lda); mu_b <= f_conv ? r_hw - 32'd1 : 32'(d_m) - 32'd1; end
            4'd2:  begin mu_a <= 64'(r_ldy); mu_b <= 32'(d_m) - 32'd1; end
            4'd3:  begin mu_a <= 64'(d_k);  mu_b <= 32'(d_n); end
            4'd4:  begin mu_a <= 64'(r_oh); mu_b <= r_ow; end
            4'd5:  begin mu_a <= 64'(r_lda); mu_b <= f_conv ? g_w : 32'd0; end
            4'd6:  begin mu_a <= 64'(r_lda); mu_b <= 32'(R); end
            4'd7:  begin mu_a <= 64'(d_k);  mu_b <= 32'(C); end
            4'd8:  begin mu_a <= 64'(r_ldy); mu_b <= 32'(R); end
            4'd9:  begin mu_a <= 64'(r_lda); mu_b <= 32'(g_s[2:0]); end
            4'd10: begin mu_a <= 64'(r_pwl); mu_b <= f_conv ? 32'(g_s[2:0]) : 32'd0; end
            4'd11: begin mu_a <= 64'(r_lda); mu_b <= f_conv ? g_kw - 32'd1 : 32'd0; end
            4'd12: begin mu_a <= 64'(r_pwl) + 64'(r_lda); mu_b <= f_conv ? 32'(g_p[1:0]) : 32'd0; end
            4'd13: begin mu_a <= 64'(f_ldb);                // v0.4: the last B row
                         mu_b <= f_elt ? 32'(d_m) - 32'd1 : f_fdot ? (d_ctrl[0] ? 32'(d_k) : 32'(d_n)) - 32'd1 : 32'd0; end
            default: begin mu_a <= 64'(d_k) << 2; mu_b <= 32'(dv_q[5]) + 32'd1; end   // ROPE table rows
          endcase
          mu_p <= 0; ck_ld <= 1'b0;
        end else begin
          logic [63:0] nx;
          nx = mu_p + (mu_b[0] ? mu_a : 64'd0) + (mu_b[1] ? mu_a << 1 : 64'd0) +
                      (mu_b[2] ? mu_a << 2 : 64'd0) + (mu_b[3] ? mu_a << 3 : 64'd0);
          if (mu_b[31:4] != 0) begin
            mu_p <= nx; mu_a <= mu_a << 4; mu_b <= mu_b >> 4;
          end else begin
            case (ck_j)
              4'd0:  r_hw <= 32'(nx);
              4'd1:  r_pa <= 48'(nx);
              4'd2:  r_py <= 48'(nx);
              4'd3:  r_pnk <= 48'(nx);
              4'd4:  r_pohw <= 32'(nx);
              4'd5:  r_pwl <= 48'(nx);
              4'd6:  r_rlda <= nx;
              4'd7:  r_ck <= nx;
              4'd8:  r_rldy <= nx;
              4'd9:  r_slda <= nx;
              4'd10: r_swlda <= nx;
              4'd11: r_kwl <= nx;
              4'd12: r_pp <= nx;
              4'd13: r_pb <= nx;
              default: r_ptab <= nx;
            endcase
            ck_ld <= 1'b1;
            if (ck_j == 4'd14) ck_st <= 3'd6; else ck_j <= ck_j + 4'd1;
          end
        end
      end
      3'd6: begin                             // the range ends, the dispatch values
        r_hia <= 66'(d_base) + 66'(d_aoff) + 66'(r_pa) + 66'(r_arow);   // (H W - 1 or M - 1) lda + row
        r_hiy <= 66'(d_base) + 66'(d_yoff) + 66'(r_py) + 66'(f_yrow);
        r_hiw <= 66'(d_base) + 66'(d_woff) + 66'(r_pnk);
        r_his <= 66'(d_base) + 66'(d_soff) + (66'(f_ns) << 2);
        r_hib <= 66'(d_base) + 66'(d_boff) + (66'(d_n) << 2);
        r_hig <= 66'(d_base) + 66'(d_soff) + (66'(d_n) << 2);
        // v0.4 extents: B (ELT by mode, the ROPE table, FDOT), C (MULADD), QUANT's scales, row scales
        r_bx  <= f_rope ? 66'(r_ptab) : f_fdot ? 66'(r_pb) + 66'(f_brow) :      // r_hix: the next state
                 (d_hrsv[1:0] == 2'd0) ? 66'(r_pb) + (66'(d_n) << 2) : (d_hrsv[1:0] == 2'd1) ? 66'(d_n) << 2 :
                 (d_hrsv[1:0] == 2'd2) ? 66'(d_m) << 2 : 66'd4;
        r_hic <= 66'(d_base) + 66'(d_boff) + 66'(r_py) + (66'(d_n) << 2);
        r_hiq <= 66'(d_base) + 66'(d_soff) + (66'(d_m) << 2);
        r_hir <= 66'(d_base) + {38'b0, d_hrsv, 4'b0} + (66'(d_m) << 2);
        if (f_elt && d_hrsv[2]) r_hia <= 66'(d_base) + 66'(d_aoff) + (66'(d_n) << 2);   // one A row
        r_d1  <= 64'(r_lda) - 64'((r_cin >> 4) - 32'd1) * 64'd16;
        r_d2  <= 64'(r_pwl) - r_kwl - 64'((r_cin >> 4) - 32'd1) * 64'd16;
        r_porig <= d_base + 64'(d_aoff) - r_pp;
        r_fits  <= (32'(d_n) >> 2) * (f_norm ? (f_bias ? 32'd3 : 32'd2) : 32'd1) <= 32'(SK * KR);
        // FPGA: no overlap (stage A's lanes and fold are left out); rows run one after another
        r_ov    <= SFU == 0 && f_row && !(d_ctrl[2:0] == 3'd0 && d_k != 0) && (32'(d_n) >> 2) * (f_norm ? (f_bias ? 32'd4 : 32'd3) : 32'd2) <= 32'(SK * KR);
        r_gb0   <= (SFU == 0 && f_row && (32'(d_n) >> 2) * (f_norm ? (f_bias ? 32'd4 : 32'd3) : 32'd2) <= 32'(SK * KR))
                   ? 32'(d_n) >> 1 : 32'(d_n) >> 2;
        r_gbfit <= f_norm && (32'(d_n) >> 2) * (f_bias ? 32'd3 : 32'd2) <= 32'(SK * KR);
        r_wfit  <= (32'(d_n) >> 2) * 32'd2 <= 32'(SK * KR);
        ck_st <= 3'd7;
      end
      default: r_hix <= 66'(d_base) + 66'(d_woff) + r_bx;   // 7 (done): S_CHK4 reads it a cycle on
    endcase
  end

  // ------------------------------------------------------------------ latched descriptor
  logic        x_gemm, x_bias, x_perch;
  logic [31:0] gp_rem, gp_pr;             // ARR 0: a GEMM's rows after this GEMV pass; rows a pass
  logic [2:0]  x_act;
  logic [1:0]  x_out;
  logic [63:0] x_a, x_w, x_y, x_s, x_b;
  logic [31:0] x_m, x_n, x_k, x_lda, x_ldy, x_es, x_osc;
  logic [1:0]  x_esl;                     // log2 of the output element size
  logic [63:0] x_rldy;                    // R * ldy (one tile row block down)
  logic [63:0] ymr;                       // m0 * ldy, kept with m0 (no multiply per tile)
  logic [63:0] e_yaddr;                   // the next epilogue row's address
  logic [63:0] e_ynx;                     // the current tile's first row address, registered
                                          // (m0 / n0 settle at the tile's end, many cycles before)
  logic        x_minner, x_reuse;
  logic        x_conv;                    // v0.3 conv: implicit GEMM, A rows from the image
  logic [31:0] x_cw, x_ch, x_ckw, x_cs, x_cp, x_ow, x_cc;   // W, H, KW, stride, pad, OW, Cin/16
  logic        x_row, x_fits;            // v0.3 row op; its row (+ gamma, beta) fit in hbuf
  logic        x_l2;                     // v0.4 L2NORM: RMSNorm with 1/N = 1
  logic        x_causal;                 // v0.4 causal softmax: row m keeps [0, x_kst + m)
  logic [31:0] x_kst, v_vlen;
  logic        x_ov;                     // two rows (+ gamma, beta) fit: rows overlap (S_W_RUN)
  logic [31:0] x_gb0, x_gb1;             // hbuf beat where gamma starts, where beta starts
  logic [1:0]  x_fn;
  logic        x_xpu;                     // v0.4: ELT / ROPE / FDOT / QUANT on the extension unit
  logic        xp_start, xp_busy, xp_rq_go, xp_lq_go, xp_rq_wide;
  logic        x_run;                     // state == S_X_RUN, as its own register (the read port's select)
  logic [2:0]  xp_op;
  logic [1:0]  xp_eop, xp_bmode;
  logic        xp_arow;
  logic [31:0] xp_g, xp_ldb, xp_lat, xp_lq_len;
  logic [63:0] xp_c, xp_rq_addr, xp_lq_addr, xp_r;
  logic [1:0]  xp_qm;
  logic [8*RB-1:0] xp_row;
  logic [31:0] x_eps, x_invn, x_nbr;     // eps bits, recip(i2f(N)), beats per row         // loop order m-inner (hold W) / n-inner (hold A); K fits

  // ------------------------------------------------------------------ GEMM tiling and array
  logic [31:0] m0, n0, cnt;                // cnt: feed step within the tile (row i takes k = cnt - i)
  always_ff @(posedge clk) e_ynx <= x_y + ymr + (64'(n0) << x_esl);
  logic [127:0] abuf [NB][R];
  logic [127:0] wbuf [NB][C];
  logic [R-1:0] a_full [NB], a_pend [NB];  // slot holds its chunk / slot has a read in flight
  logic [C-1:0] w_full [NB], w_pend [NB];
  logic signed [7:0] a_edge [R];
  logic              a_v_edge [R];
  logic signed [7:0] w_edge [C];
  logic              w_v_edge [C];
  logic signed [31:0] acc [2][R][C];      // two banks: the feeding tile's and the epilogue's
  logic              a_t_edge [R], a_f_edge [R];
  logic        f_tag;                     // accumulator bank of the tile being fed
  logic        arr_en, feed_go;
`ifdef TM_FPGA
  // FPGA: hbuf as SK banks of KR beats (bank b: row / column b's chunks), distributed RAM with one
  // write port (hw_*, the writers below never coincide) and 1 + 3 VG read ports: the feed's
  // (row / column b) and the row engine's x, gamma, beta beats (hr_*: the same beat in every bank,
  // the bank picked after the read)
  localparam int HNR = 1 + 3 * VG;
  logic         hw_e;
  logic [31:0]  hw_b, hw_c;
  logic [127:0] hw_d;
  logic [31:0]  hr_l [3 * VG];             // the row engine's linear beat indices
  logic [127:0] hr_q [3 * VG];             // ... and the beats
  logic [127:0] hf_q [SK];                 // the feed's beat of bank b
  logic [127:0] hb_q [SK][HNR];
`else
  logic [127:0] hbuf [SK][KR];             // the held operand's rows, one beat per K-chunk
`endif
  logic        a_hold, w_hold;            // this tile reads A (W) from hbuf, not DDR
  logic        h_first;                   // this tile is the first of its outer step: it fills hbuf
  assign a_hold  = x_reuse && !x_minner && n0 != 0;
  assign w_hold  = x_reuse &&  x_minner && m0 != 0;
  assign h_first = x_reuse && (x_minner ? m0 == 0 : n0 == 0);
  // Cross-tile prefetch: the loader has its own tile (lm0, ln0), at most one tile ahead of the
  // feed (l_tidx / f_tidx count tiles started). Chunk c of a tile lives in buffer
  // (base + c) mod NB, the base running on by K/16 per tile (l_base for the loader, f_base for
  // the feed), so the next tile's chunks fill the slots the current tile frees.
  logic [31:0]   lm0, ln0, l_tidx, f_tidx;
  logic [LB-1:0] l_base, f_base, f_bnx;
  logic          l_valid, l_st, l_tag;     // loader has a tile; it starts next cycle; its bank
  logic          la_hold, lw_hold, lh_first;
  logic [31:0]   l_nm0, l_nn0;            // the tile after the loader's (registered at its start)
  logic          l_has_next, l_tdone, l_adv;
  assign la_hold  = x_reuse && !x_minner && ln0 != 0;
  assign lw_hold  = x_reuse &&  x_minner && lm0 != 0;
  assign lh_first = x_reuse && (x_minner ? lm0 == 0 : ln0 == 0);
  logic [31:0] feed_last;                  // x_k + SK - 2: the last step at which some row takes k = K-1
  // per row / column, registered as the feed steps (no wide compare on the array enable):
  // k = cnt - i is inside [0, K) (fa_in), the slot of its chunk (fa_sl), the row is in the tensor
  logic          fa_in [R], fw_in [C], rlive [R], clive [C];
  logic [LB-1:0] fa_sl [R], fw_sl [C];

  // the edges are registered before the array (the feeder's byte selects and the K-range
  // compares would otherwise sit in front of the PE's multiply-accumulate): one more array step,
  // the same for every row and column, so the skew alignment holds; held while the array freezes
  logic signed [7:0] a_edge_r [R], w_edge_r [C];
  logic              a_v_r [R], a_t_r [R], a_f_r [R], w_v_r [C];
  always_ff @(posedge clk)
    if (arr_en) begin
      a_edge_r <= a_edge; a_v_r <= a_v_edge; a_t_r <= a_t_edge; a_f_r <= a_f_edge;
      w_edge_r <= w_edge; w_v_r <= w_v_edge;
    end
  tm_array #(.R(R), .C(C)) u_array (
    .clk, .en(arr_en), .a_edge(a_edge_r), .a_v_edge(a_v_r), .a_t_edge(a_t_r), .a_f_edge(a_f_r),
    .w_edge(w_edge_r), .w_v_edge(w_v_r), .acc);

  always_comb begin
    for (int i = 0; i < R; i++) begin
      int k;
      k = int'(cnt) - i;
      a_v_edge[i] = (state == S_G_FEED) && fa_in[i] && rlive[i];
      a_t_edge[i] = f_tag;
      a_f_edge[i] = (k == 0);
      a_edge[i]   = !a_v_edge[i] ? 8'sd0
`ifdef TM_FPGA
                  : a_hold ? hf_q[(i < SK) ? i : 0][8 * (k & 15) +: 8]
`else
                  : a_hold ? hbuf[(i < SK) ? i : 0][((k >> 4) < KR) ? (k >> 4) : 0][8 * (k & 15) +: 8]
`endif
                  : abuf[fa_sl[i]][i][8 * (k & 15) +: 8];
    end
    for (int j = 0; j < C; j++) begin
      int k;
      k = int'(cnt) - j;
      w_v_edge[j] = (state == S_G_FEED) && fw_in[j] && clive[j];
      w_edge[j]   = !w_v_edge[j] ? 8'sd0
`ifdef TM_FPGA
                  : w_hold ? hf_q[(j < SK) ? j : 0][8 * (k & 15) +: 8]
`else
                  : w_hold ? hbuf[(j < SK) ? j : 0][((k >> 4) < KR) ? (k >> 4) : 0][8 * (k & 15) +: 8]
`endif
                  : wbuf[fw_sl[j]][j][8 * (k & 15) +: 8];
    end
  end

  // the feed advances only when every live row and column has its current chunk in place
  always_comb begin
    feed_go = (state == S_G_FEED);
    for (int i = 0; i < R; i++)
      if (!a_hold && rlive[i] && fa_in[i] && !a_full[fa_sl[i]][i]) feed_go = 1'b0;
    for (int j = 0; j < C; j++)
      if (!w_hold && clive[j] && fw_in[j] && !w_full[fw_sl[j]][j]) feed_go = 1'b0;
  end
  assign arr_en  = (state != S_G_FEED) || feed_go;

  // ------------------------------------------------------------------ chunk loader
  // Issues chunk group lc .. lc+LG-1 in the order A0 (its LG chunks), W0, A1, W1, ... (skipping
  // rows and columns outside the tensor), one beat per cycle: a row's LG consecutive chunks are
  // one DRAM burst. Chunk c goes to buffer (base + c) mod NB. A slot is reused only after its row has
  // consumed the previous chunk there (full and pend both clear). Returns arrive in order, so a
  // FIFO of destinations says where each one goes.
  logic [31:0]   lc;                      // chunk group being issued (a multiple of LG)
  logic [31:0]   lk;                      // chunk within the group
  logic [31:0]   lcc;                     // the chunk being issued: lc + lk
  logic          l_step_row;              // the next issue moves to the next slot position
  logic [LP-1:0] lp;                      // slot position being issued (always a live one)
  logic [LO:0]   l_out;                   // reads in flight
  logic [LO-1:0] fq_h, fq_t;
  logic [LB-1:0] fq_b [OT];
  logic [2:0]    fq_k [OT];               // 0 A row, 1 W row, 2 scale beat, 3 bias beat, 4 row-scale beat
  logic [LP-1:0] fq_i [OT];
  logic [15:0]   fq_c [OT];               // K-chunk of an A/W read
  logic          fq_h1 [OT];              // ... and whether it also goes into hbuf
  logic [NP-1:0] l_live;                  // registered at the tile start (l_fi 0)
  int            l_first, l_next, l_idx;
  logic          l_isw, l_isv, l_busy, l_go, l_iss, l_ret, l_zero, l_zfill;
  logic [LB-1:0] l_b;
  // Two stages (1 GHz): the generate stage walks (lp, lk, lc) and the vector beats (vg) and puts
  // the next read into a candidate register (address, slot, padding flag precomputed); the issue
  // stage sends the candidate once its slot is free. Reads stay in order.
  logic          l_gok, l_gtake, l_ctake;
  logic          cd_v, cd_isv, cd_isw, cd_zero, cd_h1;
  logic [LB-1:0] cd_b;
  logic [LP-1:0] cd_i;                    // row / column (A, W) or vector beat index
  logic [2:0]    cd_k;                    // fq kind
  logic [15:0]   cd_c;
  logic [63:0]   cd_addr;
  logic [7:0]    vg;                      // vector beats generated
  // Addresses without multipliers (1 GHz): at the start of the loader's tile (l_st, SK cycles)
  // a walker fills per-row tables: GEMM row bases a / w (running sums of lda / K); for conv, per
  // output pixel the image position of its window corner (aiy, aix = oy S - P, ox S - P) and the
  // address of that corner pixel (stepping S lda along a row, S W lda to the next row). The tile
  // bases themselves run on as the tile moves (+ R lda, + C K, or the walker's state after R rows).
  // Conv chunk offsets come from a chunk walker (ky, kx, cc; + 16, + d1, + d2) that fills a table
  // for each chunk group (l_gfill, LG cycles).
  logic [31:0]   l_fi;                    // row being filled
  logic [63:0]   abase [SK], wbase [SK];
  logic signed [17:0] aiy [SK], aix [SK];
  logic [63:0]   fa, fw;                  // GEMM fill walkers
  logic [63:0]   l_arow0, l_wrow0;        // the tile's first A / W row address
  logic [15:0]   f_oy, f_ox, cb_oy, cb_ox, nb_oy, nb_ox;
  logic [63:0]   f_p, f_rowp, cb_p, cb_rowp, nb_p, nb_rowp;
  logic          l_gfill;
  logic [31:0]   l_gi;                    // group-table entry being filled
  logic [15:0]   cw_ky, cw_kx, cw_cc;
  logic [63:0]   cw_off;
  logic [15:0]   gt_ky [LGE], gt_kx [LGE];
  logic [63:0]   gt_off [LGE];
  logic [63:0]   x_rlda, x_ck, x_slda, x_swlda, x_d1, x_d2, x_porig;
  logic [31:0]   x_kc;                    // K / 16
  logic [63:0]   l_addr;

  // scale / bias vectors of the tile's columns: nv beats issued first (vp counts them), scale
  // beats from tag vs_t0, bias beats from vb_t0; held while the column block (vec_n0) stays
  logic [127:0]  svb [2][VB], bvb [2][VB];   // per accumulator bank: the loader fills the next
  logic          vec_ok [2];                  // tile's while the epilogue latches this one's
  logic [31:0]   vec_n0 [2];
  logic [59:0]   vs_t0 [2], vb_t0 [2];
  logic [31:0]   lt_cols;                     // live columns of the loader's tile
  logic [7:0]    nv, nvs, nvsb, vp;           // vector beats: all, scale, scale + bias
  // v0.4 QM_ROWCOL: the tile's row scales (R rows from x_rs + 4 m0), reloaded every tile
  logic          x_rowcol;
  logic [63:0]   x_rs;
  logic [127:0]  rvb [2][RVB];
  logic [59:0]   rv_t0 [2];
  logic [63:0]   rv_lo, lv_rlo;
  logic [31:0]   lt_rows, lv_rows;
  logic [7:0]    r_nb;
  logic [31:0]   t_cols;                  // live columns of the tile
  logic [63:0]   s_lo, b_lo, lv_slo, lv_blo;   // the vectors' first byte (lv_: registered, l_fi 0)
  logic [31:0]   lv_cols;
  logic [7:0]    s_nb, b_nb;
  logic          vec_need;

  always_comb begin
    t_cols   = (x_n - n0 < 32'(C)) ? x_n - n0 : 32'(C);
    lt_cols  = (x_n - ln0 < 32'(C)) ? x_n - ln0 : 32'(C);
    s_lo     = x_s + (x_perch ? {ln0, 2'b0} : 64'd0);
    b_lo     = x_b + {ln0, 2'b0};
    // beats the vectors span, from the low bits only: ((lo + len - 1) >> 4) - (lo >> 4) + 1
    s_nb     = 8'((32'(lv_slo[3:0]) + (x_perch ? lv_cols << 2 : 32'd4) - 32'd1) >> 4) + 8'd1;
    b_nb     = x_bias ? 8'((32'(lv_blo[3:0]) + (lv_cols << 2) - 32'd1) >> 4) + 8'd1 : 8'd0;
    vec_need = !(vec_ok[l_tag] && vec_n0[l_tag] == ln0);
    lt_rows  = (x_m - lm0 < 32'(R)) ? x_m - lm0 : 32'(R);
    rv_lo    = x_rs + {lm0, 2'b0};
    r_nb     = x_rowcol ? 8'((32'(lv_rlo[3:0]) + (lv_rows << 2) - 32'd1) >> 4) + 8'd1 : 8'd0;
    l_tdone = l_valid && !l_st && !l_gfill && lc >= (x_k >> 4) && vp >= nv && !cd_v;
    // move on: the feed has started the loader's tile, and the next tile does not refill the
    // hold buffer while the feed's tile still reads it
    l_adv   = l_tdone && l_has_next && (l_tidx < f_tidx ||
              (l_tidx == f_tidx && !(x_reuse && (x_minner ? l_nm0 == 0 : l_nn0 == 0) && (a_hold || w_hold))));
  end

  always_comb begin
    l_first = NP;
    l_next  = NP;
    for (int p = NP - 1; p >= 0; p--) begin
      if (l_live[p]) l_first = p;
      if (l_live[p] && p > int'(lp)) l_next = p;
    end
    l_idx  = int'(lp) >> 1;
    l_isw  = lp[0];
    lcc    = lc + lk;
    l_step_row = lk + 1 >= 32'(LGE) || lcc + 1 >= (x_k >> 4);
    l_b    = LB'(l_base + lcc[LB-1:0]);
    l_isv  = vg < nv;
    // conv: a window pixel outside the image is a zero beat, filled without a read
    begin
      logic signed [18:0] iy, ix;
      int ri, gk;
      ri = (l_idx < SK) ? l_idx : 0;
      gk = (int'(lk) < LGE) ? int'(lk) : 0;
      iy = 19'(aiy[ri]) + 19'($signed({1'b0, gt_ky[gk]}));
      ix = 19'(aix[ri]) + 19'($signed({1'b0, gt_kx[gk]}));
      l_zero = x_conv && !l_isw && (iy < 0 || ix < 0 || iy >= 19'(x_ch) || ix >= 19'(x_cw));
    end
    // generate: a next read exists; it goes into the candidate register when that is free
    l_gok  = l_valid && !l_st && !l_gfill && x_gemm && (l_isv || lcc < (x_k >> 4));
    // issue: the candidate's slot is free (vector beats have their own buffers)
    l_busy = cd_isv ? 1'b0
           : cd_isw ? (w_full[cd_b][(int'(cd_i) < C) ? int'(cd_i) : 0] || w_pend[cd_b][(int'(cd_i) < C) ? int'(cd_i) : 0])
                    : (a_full[cd_b][(int'(cd_i) < R) ? int'(cd_i) : 0] || a_pend[cd_b][(int'(cd_i) < R) ? int'(cd_i) : 0]);
    l_go   = cd_v && !cd_zero && !l_busy && l_out < (LO + 1)'(OT);
    l_ret  = mr_rvalid && l_out != 0 && !(state inside {S_V_GB, S_V_PASS, S_V_FOLD});
    l_zfill = cd_v && cd_zero && !l_busy && !(SFU == 1 && l_ret);   // FPGA: one write port per buffer
    l_ctake = (l_go && mr_ready) || l_zfill;
    l_gtake = l_gok && (!cd_v || l_ctake);
    if (l_isv)
      l_addr = (vg < nvs) ? {vs_t0[l_tag] + 60'(vg), 4'b0} : (vg < nvsb) ? {vb_t0[l_tag] + 60'(vg - nvs), 4'b0}
             : {rv_t0[l_tag] + 60'(vg - nvsb), 4'b0};
    else
      l_addr = l_isw ? wbase[(l_idx < SK) ? l_idx : 0] + {28'b0, lcc, 4'b0}
             : abase[(l_idx < SK) ? l_idx : 0] + (x_conv ? gt_off[(int'(lk) < LGE) ? int'(lk) : 0] : {28'b0, lcc, 4'b0});
    l_iss  = l_go && mr_ready;
  end

  // ------------------------------------------------------------------ output queue
  // OQ slots between the pipelined producers (epilogue lanes, later the row engine) and the
  // segment writer. A slot is taken at launch (address and length known then) and filled when the
  // data leaves the lanes, in launch order; the writer takes the head once it is filled. A
  // producer launches only while a slot is free, so the queue never overflows and the lanes
  // never stall.
  localparam int LQO = $clog2(OQ);
  logic [8*RB-1:0] oq_dh;                 // the head slot's row, address and length
  logic [63:0]     oq_ah;
  logic [31:0]     oq_lh;
`ifndef TM_FPGA                             // (FPGA: u_oqd, u_oqal, distributed RAM, at the end)
  logic [8*RB-1:0] oq_d [OQ];
  logic [63:0]     oq_a [OQ];
  logic [31:0]     oq_l [OQ];
  assign oq_dh = oq_d[oq_hd];
  assign oq_ah = oq_a[oq_hd];
  assign oq_lh = oq_l[oq_hd];
`endif
  logic [OQ-1:0]   oq_rdy;
  logic [LQO-1:0]  oq_tl, oq_dp, oq_hd;
  logic [LQO:0]    oq_cnt;
  logic            oq_pop;

  // ------------------------------------------------------------------ v0.4 extension unit
  // ELT, ROPE, FDOT and the row QUANT: its own reader (the read port is its own in S_X_RUN) and
  // datapath; its outputs take output-queue slots like the epilogue's rows (tap: xp_lat)
  tm_xpu #(.VQ(VQ), .RB(RB), .MB(GB), .RW(RW)) u_xpu (
    .clk, .rst, .start(xp_start), .op(xp_op), .qm(xp_qm), .bias(x_bias), .r_base(xp_r), .eop(xp_eop), .bmode(xp_bmode), .arow(xp_arow),
    .act(x_act), .out(x_out), .osc(x_osc), .m_n(x_m), .n_n(x_n), .k_n(x_k), .g_n(xp_g),
    .a_base(x_a), .b_base(x_w), .c_base(xp_c), .y_base(x_y), .s_base(x_s),
    .lda(x_lda), .ldb(xp_ldb), .ldy(x_ldy), .busy(xp_busy), .lat(xp_lat),
    .rq_go(xp_rq_go), .rq_addr(xp_rq_addr), .rq_ready(mr_ready && x_run),
    .rq_ret(mr_rvalid && x_run), .rq_data(mr_rdata), .rq_wide(xp_rq_wide), .rq_hi(mr_rhi),
    .oq_free(32'(oq_cnt) + 32'(xp_lq_go) < 32'(OQ)), .lq_go(xp_lq_go), .lq_addr(xp_lq_addr), .lq_len(xp_lq_len),
    .x_row(xp_row)
`ifdef TM_FPGA
    , .sfu_x(xp_sx), .sfu_y(sfu_y), .sh_la(xs_la), .sh_lb(xs_lb), .sh_ly(lane_y), .sh_fl(xs_fl),
      .sh_fsum(f_sum), .sh_rx(xs_rx), .sh_ry(f_r)
`endif
    );

  // ------------------------------------------------------------------ GEMM row epilogue
  // The epilogue engine: row e_ei of the handed-off tile (bank e_bank, origin e_m0 / e_n0, its
  // scale and bias latched at handoff) is launched into C pipelined lanes, each the contract's
  // steps 2-6 (each rounded once, no FMA), and leaves them EL cycles later into the output queue.
  // It runs while the next tile feeds. Row i is final once the array has advanced i + 2 times after
  // the tile's last feed step (e_adv counts the advances): the edge registers and the PE's product
  // register add a step each, so PE (i, j) adds a[i][K-1] at step K+i+j+1 <= K+SK+i, the last
  // feed step is K+SK-2.
  //   lane (cycles after the launch registers): i2f 2, fmul row scale 5 (v0.4; 1.0 unless
  //   QM_ROWCOL, exact), fmul scale 8, fadd bias 11 (or bypass), act 12 (identity / relu / relu6),
  //   133 (GELU, SiLU, sigmoid, exp) or 163 (softplus), then fp32 +0, bf16 +1, int8 +4.
  logic [31:0]     ei;                    // GEMM: unused (kept for the row-op dispatch reset)
  logic            e_busy, e_bank;
  logic [31:0]     e_ei, e_m0, e_n0, e_cols, e_adv, f_adv;
  logic [31:0]     e_nrow;                // rows of the handed-off tile inside the tensor
  logic [31:0]     e_s [C], e_b [C];      // the tile's scale / bias per column
  logic [31:0]     t_s [C], t_b [C];      // the current tile's, from the vector beats
  logic [31:0]     e_r [R], t_r [R];      // v0.4: the tile's row scales (1.0 unless QM_ROWCOL)
  logic [31:0]     q_r;
  logic            e_go;
  logic [31:0]     r_len;                 // the launched row's live bytes
  logic [63:0]     r_addr;
  logic signed [31:0] q_acc [C];          // launch registers
  logic [31:0]     q_s [C], q_b [C];
  logic [31:0]     e_lane [C];            // each lane's result as it leaves (4 bytes)
  localparam int   EL_MAX = 23 + SP_LAT     // GEMV with softplus and INT8 out: 16 + SP_LAT + 4, and margin
                          + SFU * ((C > 4 * VG) ? C : 4 * VG);   // the shared SFU's lane skew
  logic [EL_MAX:0] e_vsr;                 // launched rows in flight, by age
  logic [31:0]     e_lat;                 // this descriptor's lane latency
  logic            e_emerge;
  logic            x_gs;                  // the activation takes the 122-cycle path (GELU, SiLU, sigmoid, exp)
  logic            x_sp;                  // the activation is softplus (SP_LAT cycles)
  logic            e_sok;                 // the shared SFU takes a row now (FPGA: every C cycles)
  logic [31:0]     e_scnt;
  logic [31:0]     e_ax [C], e_ay [C];    // the lanes' long-act inputs and results (shared SFU)

  always_comb begin
    // column j's scale / bias word: the tag's vectors start at the beat of x_s + 4 n0 (x_b + 4 n0),
    // so the word is (that address's word in its beat + j) from the first beat: small adds only
    for (int j = 0; j < C; j++) begin
      logic [31:0] ks, kb;
      ks = 32'(2'(x_s[3:2] + (x_perch ? n0[1:0] : 2'd0))) + (x_perch ? 32'(j) : 32'd0);
      kb = 32'(2'(x_b[3:2] + n0[1:0])) + 32'(j);
      t_s[j] = svb[f_tag][(ks >> 2) < VB ? ks >> 2 : 0][32 * ks[1:0] +: 32];
      t_b[j] = bvb[f_tag][(kb >> 2) < VB ? kb >> 2 : 0][32 * kb[1:0] +: 32];
    end
    for (int i = 0; i < R; i++) begin
      logic [31:0] kr;
      kr = 32'(2'(x_rs[3:2] + m0[1:0])) + 32'(i);
      t_r[i] = x_rowcol ? rvb[f_tag][(kr >> 2) < RVB ? kr >> 2 : 0][32 * kr[1:0] +: 32] : 32'h3F800000;
    end
    x_gs     = x_act == 3'd3 || x_act == 3'd4 || x_act == 3'd5 || x_act == 3'd7;
    x_sp     = x_act == 3'd6;
    e_lat    = (x_gs ? 32'd133 : x_sp ? 32'd11 + 32'(SP_LAT) : 32'd12) + ((x_gs || x_sp) ? 32'(SFU * (C - 1)) : 32'd0)
             + ((x_out == 2'd0) ? 32'd0 : (x_out == 2'd1) ? 32'd1 : 32'd4);
    r_len    = e_cols << x_esl;
    r_addr   = e_yaddr;
    e_sok    = SFU == 0 || C == 1 || !(x_gs || x_sp) || e_scnt == 0;
    e_go     = e_busy && e_ei < e_nrow && e_adv > e_ei + 32'd1 && oq_cnt < (LQO + 1)'(OQ) && e_sok;
  end

  // GEMM / conv lanes, the extension unit's, or the row engine's
  assign e_emerge = e_vsr[x_gemm ? e_lat : x_xpu ? xp_lat : v_olat];

  for (genvar j = 0; j < C; j++) begin : g_epi
    logic [31:0] f0, s1, s0, y0, b4, y1a, y0d, y1, ya1, yalong, yasp, ya, ybf, yq, yq3, oq, iu;
    logic [31:0] r0, r1, yr, s4;
    logic [7:0]  q8r;
    logic        is, iz;
    always_ff @(posedge clk) begin                                         // i2f in two stages
      is <= q_acc[j][31]; iz <= q_acc[j] == 0;                             // 1
      iu <= q_acc[j][31] ? 32'(-q_acc[j]) : 32'(q_acc[j]);
      s0 <= q_s[j]; r0 <= q_r;
      f0 <= iz ? 32'h0 : tm_fp_pkg::i2f_tail(is, iu);                      // 2
      s1 <= s0; r1 <= r0;
    end
    fpu_mul u_r (.clk, .a(f0), .b(r1), .y(yr));                            // 5: the row scale
    fpu_dly #(32, 3) u_s4 (.clk, .d(s1), .q(s4));
    fpu_mul u_m (.clk, .a(yr), .b(s4), .y(y0));                            // 8
    fpu_dly #(32, 8) u_b4 (.clk, .d(q_b[j]), .q(b4));
    fpu_add u_a (.clk, .a(y0), .b(b4), .y(y1a));                           // 11
    fpu_dly #(32, 3) u_y0d (.clk, .d(y0), .q(y0d));
    assign y1 = x_bias ? y1a : y0d;
`ifdef TM_FPGA
    fpu_act1 u_act (.clk, .x(y1), .act(x_act), .y1(ya1));                  // 12
    assign e_ax[j] = y1;
    assign yalong  = e_ay[j];                                              // 133 / 11 + SP_LAT, + C - 1
    assign yasp    = e_ay[j];
`else
    fpu_act u_act (.clk, .x(y1), .act(x_act), .y1(ya1), .ylong(yalong), .ysp(yasp));   // 12, 133, 163
`endif
    assign ya = x_gs ? yalong : x_sp ? yasp : ya1;
    always_ff @(posedge clk) ybf <= {16'b0, tm_fp_pkg::to_bf16(ya)};       // +1
    fpu_mul u_q (.clk, .a(ya), .b(x_osc), .y(yq3));                        // +3
    always_ff @(posedge clk) q8r <= tm_fp_pkg::q8_tail(yq3);               // +4
    assign oq = (x_out == 2'd0) ? ya : (x_out == 2'd1) ? ybf : {24'b0, q8r};
    assign e_lane[j] = oq;
  end
  // pack the lanes to the output size
  logic [8*RB-1:0] e_row;
  always_comb begin
    e_row = '0;
    for (int j = 0; j < C; j++)
      for (int b = 0; b < 4; b++)
        if (32'(b) < x_es) e_row[8 * (j * int'(x_es) + b) +: 8] = e_lane[j][8 * b +: 8];
  end

  // ------------------------------------------------------------------ segment writer
  // Holds one row segment and writes it as wr_nb byte-strobed beats, one per accepted cycle; it
  // takes the next row in the cycle its last beat is accepted.
  logic            wr_busy;
  logic [8*RB-1:0] wr_data;
  logic [31:0]     wr_len;
  logic [63:0]     wr_addr;
  logic [7:0]      wr_nb, wr_b;
  logic [127:0]    wr_bdata;
  logic [15:0]     wr_bstrb;
  logic            wr_acc, wr_free;

  always_comb begin
    for (int x = 0; x < 16; x++) begin
      int idx;
      idx = int'(wr_b) * 16 + x - int'(wr_addr[3:0]);
      wr_bstrb[x] = idx >= 0 && idx < int'(wr_len);
      wr_bdata[8 * x +: 8] = wr_bstrb[x] ? wr_data[8 * ((idx >= 0 && idx < RB) ? idx : 0) +: 8] : 8'h00;
    end
    wr_acc  = wr_busy && mw_ready;
    wr_free = !wr_busy || (wr_acc && wr_b == wr_nb - 8'd1);
  end
  assign oq_pop = oq_rdy[oq_hd] && wr_free;

  // ------------------------------------------------------------------ v0.3 row-op engine
  // Per row, 2-3 passes over the row's 4-element groups (one 16-byte beat each): pass 0 reads the
  // row from DDR (and, when it fits, copies it into hbuf); later passes read hbuf, VG groups per
  // step, or DDR again when it does not fit. Element j goes to lane j mod 16; each pass ends
  // with the contract's fold (max, or pairwise sums 0+1, 2+3, ...). The last pass writes.
  //   softmax:   p0 max -> mx;   p1 sum e = exp(x - mx) -> r = recip;     p2 y = e * r
  //   layernorm: p0 sum -> mean; p1 sum (x - mean)^2 -> rs = rsqrt;       p2 y = (d * rs) * g (+ b)
  //   rmsnorm:   p0 sum x^2 -> rs;                                        p1 y = (x * rs) * g (+ b)
  // Pipelined (1 GHz): a step is issued into 4 VG vpu_lane slots (rtl/tm_vpu.sv) with its own
  // state (kind, scalars, lane mapping); contributions reach the 16 lane accumulators a fixed
  // number of cycles later, each through a 3-cycle fadd (with the previous result forwarded), so a
  // step may touch a lane only 3 cycles after the last step that did (v_cool). A pass ends when
  // its steps are issued and drained (v_drain); the fold and the pass scalar then come out of
  // vpu_fold after its latency (v_fwait). Output steps take an output-queue slot at issue and fill
  // it when they leave the lanes; the next row may start meanwhile.
  // Gamma / beta: in hbuf (loaded once per descriptor, S_V_GB) when they fit, else read from DDR
  // with each group in the output pass. hbuf regions (beats): row [0, nbr), gamma [nbr, 2 nbr),
  // beta [2 nbr, 3 nbr), linear index i at hbuf[i / KR][i % KR].
  localparam int LQ = $clog2(VQ);
  localparam int NS = 4 * VG;                // element slots
  localparam int EXW = SFU * (NS - 1);       // FPGA: softmax's exp through the shared SFU, lanes in turn
  localparam int CL_MAX = 54 + EXW;          // longest contribution latency
  logic [31:0]  v_rm, v_gi_iss, v_gi_con;   // row; next group to issue / to consume
  logic [31:0]  v_left;                     // x_nbr - v_gi_con, kept beside it (no compare on the issue path)
  logic [31:0]  v_ileft;                    // x_nbr - v_gi_iss, likewise
  logic [1:0]   v_pass, v_kind_iss;         // pass; kind of the next read in the group
  logic [1:0]   v_gb;                       // S_V_GB: 1 gamma, 2 beta
  logic [31:0]  v_mx_or_mean, v_sc2;        // pass scalars (mx or mean; r or rs)
  logic [LQ:0]  v_occ, v_rcv;               // reads issued and not consumed / returned and not consumed
  logic [LQ-1:0] vq_t, vq_r, vq_h;
  logic [1:0]   vq_k [VQ];
  logic [127:0] vq_q [3];                   // the reader FIFO's head and the two beats after it
`ifndef TM_FPGA                             // (FPGA: u_vq, distributed RAM, at the end)
  logic [127:0] vq_d [VQ];
  for (genvar k = 0; k < 3; k++) begin : g_vqq
    assign vq_q[k] = vq_d[vq_h + LQ'(k)];
  end
`endif
  logic         v_last, v_hb, v_gsrc_hb;    // output pass; pass reads hbuf; gamma/beta in hbuf
  logic [1:0]   v_gs;                       // reads per group from DDR (1-3)
  logic         v_go, v_iss, v_ret, v_con, v_ready;
  logic [63:0]  v_addr;
  logic [127:0] vx;                         // group 0 of the step (copied to hbuf in pass 0)
  logic [127:0] vxw [VG], vgw [VG], vbw [VG];
  logic [31:0]  v_n;                        // groups in this step
  logic [31:0]  v_npass;
  logic [63:0]  v_yrow, v_ynext;            // output row address (running: + ldy per row)
  logic [63:0]  v_arow, v_anext, v_iaddr;   // input row address for the reader (running)
  // step issue registers (cycle 0 of the lanes)
  logic [31:0]  sx [NS], sg [NS], sb [NS];
  logic         skeep [NS];
  logic [31:0]  sc1, ssc;
  logic         skout, sct;                 // output step / contribution step at cycle 0
  logic [1:0]   sg4;                        // the step's first group mod 4 (lane mapping)
  logic [2:0]   sn;                         // groups in the step
  logic [31:0]  ssc_d3, ssc_d54;
  logic         skout_d3;
  logic [31:0]  vlc [NS], vly [NS];         // lane contributions / outputs
  // contributions: valid line (with the step's mapping), accumulators
  logic [CL_MAX:0] cvl;
  logic [1:0]   cg4 [CL_MAX+1];
  logic [2:0]   cn [CL_MAX+1];
  logic [31:0]  v_clat;                     // contribution latency of this pass
  logic         c_sum;                      // the pass sums (else max)
  logic         c_v;                        // a step's contributions arrive now (registered tap:
  logic [1:0]   c_g4;                       // one cycle after the lane's contribution cycle)
  logic [2:0]   c_n;
  logic         c_z;                        // the contribution is x itself (from the issue registers)
  logic [31:0]  vlc_r [NS], sx_r [NS];
  logic [31:0]  lane [16];                  // accumulators
  logic [31:0]  lane_c [16];                // contribution arriving per lane
  logic [15:0]  lane_in;                    // a contribution for the lane arrives now
  logic [31:0]  lane_y [16];                // adder results
  logic [2:0]   lane_p [16];                // adder in flight, by age
  logic [1:0]   v_cool [16];                // cycles until the lane may take another summed step
  logic [7:0]   v_drain;                    // cycles until the pass's contributions are all in
  logic [7:0]   v_fwait;                    // cycles until the fold / scalar is out
  logic         v_folding;
  logic [31:0]  f_mx, f_mean, f_rs, f_r;    // vpu_fold outputs (B)
  logic [15:0]  step_lanes;                 // lanes the next step touches
  logic         v_cool_ok;
  logic [8*RB-1:0] v_row;                   // output step leaving the lanes, packed
  logic [31:0]  v_olat;                     // output latency (with conversion)
  logic         v_sok;                      // the shared SFU takes a step now (FPGA, long act or softmax exp: every NS cycles)
  logic         v_sneed;
  logic [31:0]  v_scnt;
  logic [31:0]  v_sy [NS];                  // the lanes' long-act results (shared SFU)
  logic [31:0]  v_xf [NS], v_ye [NS];       // softmax: the lanes' exp arguments and results (shared SFU)
  // Overlap (x_ov): stage A runs pass 0 of row a_row (reader -> hbuf row region a_buf, its own
  // lanes and fold) while stage B (b_act) runs the later passes of the previous row from the other
  // region with the issue machinery above. Handoff when A's scalar is out and B has issued its
  // last step. The reader runs ahead across rows (v_irow).
  logic [31:0]  a_row, a_gi, v_irow, v_rbase;
  logic [31:0]  a_gleft, a_wad;             // x_nbr - a_gi; hbuf beat of group a_gi in A's region
  logic         a_more;                     // a_row < x_m
  logic         a_buf, b_buf, b_act, a_con, a_fold, a_ready;
  logic [7:0]   a_drain, a_fwait;
  logic [31:0]  a_c1, a_sc;                 // A's scalar for B: mx / mean, or rs
  logic [31:0]  a_lane [16], ax [4], axx [4], a_ly [16], a_lc [16], a_lcc [16];
  logic [2:0]   a_p [16];
  logic         ax_v, axx_v;
  logic [1:0]   ax_g4, axx_g4;
  logic [15:0]  a_in, a_inc;               // (a_in, a_lc: registered, one cycle after a_inc, a_lcc)
  logic [31:0]  af_mx, af_mean, af_rs, af_r;
  logic         b_wait;                     // B: pass end (drain, fold) in progress
  // 1/N for the norms: recip(i2f(N)) through a pipelined unit at dispatch (S_V_INV)
  logic [31:0]  inv_in, inv_out;
  logic [7:0]   inv_wait;
  state_t       v_after_inv;

`ifndef TM_FPGA
  function automatic logic [127:0] hb_rd(input logic [31:0] i);
    return hbuf[(i / KR) < SK ? i / KR : 0][i % KR];
  endfunction
`endif

  always_comb begin
    v_npass   = (x_fn == 2'd3) ? 32'd1 : (x_fn == 2'd2) ? 32'd2 : 32'd3;
    v_last    = 32'(v_pass) == v_npass - 1;
    v_hb      = x_fits && v_pass != 0;
    v_gsrc_hb = x_fits;
    v_rbase   = (x_ov && b_buf) ? x_nbr : 32'd0;
    v_gs      = (state == S_V_GB) ? 2'd1
              : (v_last && (x_fn == 2'd1 || x_fn == 2'd2) && !x_fits) ? (x_bias ? 2'd3 : 2'd2) : 2'd1;
    // reads: from DDR in S_V_GB, and in S_V_PASS unless the pass reads hbuf
    v_go   = (((state == S_V_GB) || (state == S_V_PASS && !v_hb && !b_wait)) && v_ileft != 0 ||
              (state == S_W_RUN && v_irow < x_m)) && v_occ < (LQ + 1)'(VQ);
    if (state == S_V_GB)
      v_addr = ((v_gb == 2'd1) ? x_s : x_b) + 64'(v_gi_iss) * 64'd16;
    else if (v_kind_iss == 2'd0)
      v_addr = ((state == S_W_RUN) ? v_iaddr : v_arow) + {28'b0, v_gi_iss, 4'b0};
    else
      v_addr = ((v_kind_iss == 2'd1) ? x_s : x_b) + {28'b0, v_gi_iss, 4'b0};
    v_iss  = v_go && mr_ready;
    v_ret  = mr_rvalid && (v_occ - v_rcv) != 0;
    // the groups of the next step: VG from hbuf (steps start at multiples of VG, so its 4 VG
    // elements go to distinct lanes and each lane sees its elements in order), one from the reader
    v_n = (!v_hb || v_left == 0) ? 32'd1 : (v_left < 32'(VG)) ? v_left : 32'(VG);
    for (int k = 0; k < VG; k++) begin
`ifdef TM_FPGA
      vxw[k] = v_hb ? hr_q[k] : (k == 0) ? vq_q[0] : 128'b0;
      vgw[k] = v_gsrc_hb ? hr_q[VG + k] : (k == 0) ? vq_q[1] : 128'b0;
      vbw[k] = v_gsrc_hb ? hr_q[2 * VG + k] : (k == 0) ? vq_q[2] : 128'b0;
`else
      vxw[k] = v_hb ? hb_rd(v_rbase + v_gi_con + 32'(k)) : (k == 0) ? vq_q[0] : 128'b0;
      vgw[k] = v_gsrc_hb ? hb_rd(x_gb0 + v_gi_con + 32'(k)) : (k == 0) ? vq_q[1] : 128'b0;
      vbw[k] = v_gsrc_hb ? hb_rd(x_gb1 + v_gi_con + 32'(k)) : (k == 0) ? vq_q[2] : 128'b0;
`endif
    end
    vx = vxw[0];
    // lanes the next step touches (element 4 gi + i goes to lane (4 gi + i) mod 16)
    step_lanes = '0;
    for (int i = 0; i < NS; i++)
      if (32'(i / 4) < v_n) step_lanes[((int'(v_gi_con[1:0]) + i / 4) % 4) * 4 + i % 4] = 1'b1;
    v_cool_ok = 1'b1;
    for (int l = 0; l < 16; l++) if (step_lanes[l] && v_cool[l] != 0) v_cool_ok = 1'b0;
    c_sum  = !(x_fn == 2'd0 && v_pass == 2'd0);
    // a step can be issued when its data is here, the lanes it sums into are free and (output
    // steps) an output-queue slot is
    v_ready = v_left != 0 && (v_hb ? 1'b1 : v_rcv >= (LQ + 1)'(v_gs)) &&
              (v_last ? oq_cnt < (LQO + 1)'(OQ) : (!c_sum || v_cool_ok));
    v_sneed = (x_fn == 2'd3 && (x_gs || x_sp)) || (x_fn == 2'd0 && v_pass != 0);   // the step uses the SFU
    v_sok  = SFU == 0 || !v_sneed || v_scnt == 0;
    v_con  = ((state == S_V_PASS && !b_wait) || (state == S_W_RUN && b_act && !b_wait)) && v_ready && v_sok;
    // latencies: contribution of this pass, output of the descriptor (vpu_lane)
    case ({x_fn, v_pass})
      {2'd0, 2'd1}: v_clat = 32'd54 + 32'(EXW);
      {2'd1, 2'd1}: v_clat = 32'd6;
      {2'd2, 2'd0}: v_clat = 32'd3;
      default:      v_clat = 32'd0;
    endcase
    v_olat = ((x_fn == 2'd0) ? 32'd57 + 32'(EXW) : (x_fn == 2'd1) ? 32'd12 : (x_fn == 2'd2) ? 32'd9
              : x_gs ? 32'd122 + 32'(SFU * (NS - 1)) : x_sp ? 32'(SP_LAT) + 32'(SFU * (NS - 1)) : 32'd1)
             + ((x_out == 2'd0) ? 32'd0 : (x_out == 2'd1) ? 32'd1 : 32'd4);
    // contributions arriving now
    lane_in = '0;
    for (int l = 0; l < 16; l++) lane_c[l] = 32'h0;
    for (int i = 0; i < NS; i++)
      if (c_v && 32'(i / 4) < 32'(c_n)) begin
        lane_in[((int'(c_g4) + i / 4) % 4) * 4 + i % 4] = 1'b1;
        lane_c[((int'(c_g4) + i / 4) % 4) * 4 + i % 4]  = c_z ? sx_r[i] : vlc_r[i];
      end
    // the output step leaving the lanes, packed to the output size
    v_row = '0;
    for (int i = 0; i < NS; i++)
      for (int b = 0; b < 4; b++)
        if (32'(b) < x_es) v_row[8 * (i * int'(x_es) + b) +: 8] = vly[i][8 * b +: 8];
    // stage A: one group per step from the reader, pass 0 of the function
    a_con  = state == S_W_RUN && a_more && a_gleft != 0 && v_rcv != 0 && !a_fold && !a_ready;
    a_inc  = '0;
    for (int l = 0; l < 16; l++) a_lcc[l] = 32'h0;
    // contributions: x one cycle after consumption (softmax, LayerNorm), x*x four (RMSNorm)
    for (int i = 0; i < 4; i++)
      if (x_fn == 2'd2 ? axx_v : ax_v) begin
        a_inc[int'(x_fn == 2'd2 ? axx_g4 : ax_g4) * 4 + i] = 1'b1;
        a_lcc[int'(x_fn == 2'd2 ? axx_g4 : ax_g4) * 4 + i] = (x_fn == 2'd2) ? axx[i] : ax[i];
      end
  end

  for (genvar i = 0; i < NS; i++) begin : g_vl
    vpu_lane #(.EXD(EXW)) u (.clk, .fn(x_fn), .bias(x_bias), .act(x_act), .out(x_out), .osc(x_osc), .keep(skeep[i]),
                .x(sx[i]), .g(sg[i]), .b(sb[i]), .c1(sc1), .sc_d0(ssc), .sc_d3(ssc_d3),
                .sc_d54(ssc_d54), .kout_d0(skout), .kout_d3(skout_d3), .contrib(vlc[i]), .y(vly[i])
`ifdef TM_FPGA
                , .yl(v_sy[i]), .xf(v_xf[i]), .ye(v_ye[i])
`endif
                );
  end
  fpu_dly #(32, 3)  u_sc3 (.clk, .d(ssc), .q(ssc_d3));
  fpu_dly #(32, 54 + EXW) u_sc54 (.clk, .d(ssc), .q(ssc_d54));
  fpu_dly #(1, 3)   u_ko3 (.clk, .d(skout), .q(skout_d3));
  // lane accumulators: a 3-cycle fadd each, the result forwarded to a contribution arriving as it
  // completes (the step spacing guarantees no other overlap)
  for (genvar l = 0; l < 16; l++) begin : g_acc
    logic [31:0] cur;
    assign cur = lane_p[l][2] ? lane_y[l] : lane[l];
`ifdef TM_FPGA
    // the extension unit's FDOT accumulators share these adders (the row engine is idle then)
    fpu_add u (.clk, .a(x_xpu ? xs_la[l] : cur), .b(x_xpu ? xs_lb[l] : lane_c[l]), .y(lane_y[l]));
`else
    fpu_add u (.clk, .a(cur), .b(lane_c[l]), .y(lane_y[l]));
`endif
  end
`ifdef TM_FPGA
  logic [31:0] f_lane [16];
  assign f_lane = x_xpu ? xs_fl : lane;
  vpu_fold u_fold (.clk, .lane(f_lane), .invn(x_invn), .eps(x_eps), .mx(f_mx), .mean(f_mean), .rs(f_rs), .r(f_r),
                   .rsel(x_xpu || state == S_V_INV), .rx(x_xpu ? xs_rx : inv_in), .fsum(f_sum));
`else
  vpu_fold u_fold (.clk, .lane, .invn(x_invn), .eps(x_eps), .mx(f_mx), .mean(f_mean), .rs(f_rs), .r(f_r));
`endif
  // stage A: its own accumulators and fold (not in the FPGA build, which never overlaps)
`ifdef TM_FPGA
  assign axx = '{default: 32'h0};
  assign axx_v = 1'b0;
  assign axx_g4 = 2'd0;
  assign a_ly = '{default: 32'h0};
  assign {af_mx, af_mean, af_rs, af_r} = '0;
`else
  for (genvar i = 0; i < 4; i++) begin : g_axx
    fpu_mul u (.clk, .a(ax[i]), .b(ax[i]), .y(axx[i]));
  end
  fpu_dly #(1, 3) u_axv (.clk, .d(ax_v), .q(axx_v));
  fpu_dly #(2, 3) u_axg (.clk, .d(ax_g4), .q(axx_g4));
  for (genvar l = 0; l < 16; l++) begin : g_aacc
    logic [31:0] cur;
    assign cur = a_p[l][2] ? a_ly[l] : a_lane[l];
    fpu_add u (.clk, .a(cur), .b(a_lc[l]), .y(a_ly[l]));
  end
  vpu_fold u_afold (.clk, .lane(a_lane), .invn(x_invn), .eps(x_eps), .mx(af_mx), .mean(af_mean),
                    .rs(af_rs), .r(af_r));
`endif
  always_ff @(posedge clk) inv_in <= tm_fp_pkg::i2f(32'(x_n));
`ifdef TM_FPGA
  assign inv_out = f_r;                    // the fold's recip, on inv_in in S_V_INV (42 cycles >= 1 + 38)
`else
  fpu_recip u_inv (.clk, .x(inv_in), .y(inv_out));
`endif

  // ------------------------------------------------------------------ ports
  // the loader and the single-read subroutine never overlap: the loader runs only in S_G_FEED,
  // and the feed ends only after every chunk has arrived (no loader read is then in flight)
  assign mr_req  = (state == S_RD_REQ) || l_go || v_go || (x_run && xp_rq_go);
  assign mr_addr = (state == S_RD_REQ) ? rd_addr : v_go ? v_addr : x_run ? xp_rq_addr : cd_addr;
  assign mr_wide = RW > 1 && !(state == S_RD_REQ) && !v_go && x_run && xp_rq_wide;
  // every write goes through the segment writer (GEMM / conv rows, row-op and activation groups)
  assign mw_req  = wr_busy;
  assign mw_addr = {wr_addr[63:4] + 60'(wr_b), 4'b0};
  assign mw_data = wr_bdata;
  assign mw_strb = wr_bstrb;

  // ------------------------------------------------------------------ performance counters
  // Observability only: no result depends on them. Every running cycle is charged to exactly one
  // phase (they sum to `cycles`); a memory subroutine cycle is charged to the phase that called it.
  localparam int NPH = 12;
  localparam int PH_FETCH = 0, PH_CHECK = 1, PH_G_LOAD = 2, PH_G_LOAD_MEM = 3, PH_G_FEED = 4,
                 PH_G_DRAIN = 5, PH_G_EPI = 6, PH_G_EPI_MEM = 7, PH_WRITE = 8, PH_R_ELEM = 9,
                 PH_R_MEM = 10, PH_OTHER = 11;
  logic [63:0] perf_phase [NPH];
  logic [63:0] perf_rd_ph [NPH];          // read beats issued, by the phase that asked
  logic [63:0] perf_rd_beats, perf_wr_beats, perf_rd_accept_wait, perf_rd_latency,
               perf_wr_accept_wait, perf_feed_valid;
  int          ph;

  always_comb begin
    case (state)
      S_RD_REQ, S_RD_WAIT:
        case (ret)
          S_FETCH_GOT:             ph = PH_FETCH;
          default:                 ph = PH_OTHER;
        endcase
      S_G_TILE_END:                                        ph = PH_WRITE;
      S_NEXT, S_FETCH, S_FETCH_GOT:                        ph = PH_FETCH;
      S_CRC, S_CHKW, S_CHK4, S_CHECK:                   ph = PH_CHECK;
      S_G_TILE:                                            ph = (x_minner ? n0 >= x_n : m0 >= x_m)
                                                                ? PH_WRITE : PH_G_LOAD;
      S_G_CLR:                                             ph = PH_G_LOAD;
      S_G_FEED:                                            ph = feed_go ? PH_G_FEED : PH_G_LOAD_MEM;
      S_G_EPI:                                             ph = PH_G_EPI;   // waiting for the epilogue engine
      S_V_PASS:                                            ph = v_con ? PH_R_ELEM : PH_R_MEM;
      S_V_GB:                                              ph = PH_R_MEM;
      S_V_ROW, S_V_FOLD, S_W_RUN:                          ph = PH_R_ELEM;
      S_X_RUN:                                             ph = (xp_op == 3'd5) ? PH_G_FEED : PH_R_ELEM;   // GEMV counts as the GEMM's feed
      S_V_NEXT:                                            ph = PH_WRITE;
      default:                                             ph = PH_OTHER;
    endcase
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      for (int i = 0; i < NPH; i++) begin perf_phase[i] <= 0; perf_rd_ph[i] <= 0; end
      perf_rd_beats <= 0; perf_wr_beats <= 0; perf_rd_accept_wait <= 0; perf_rd_latency <= 0;
      perf_wr_accept_wait <= 0; perf_feed_valid <= 0;
    end else if (state != S_IDLE && state != S_HALT) begin
      perf_phase[ph] <= perf_phase[ph] + 1;
      if (mr_req && mr_ready)             perf_rd_beats <= perf_rd_beats + 1;
      if (mr_req && mr_ready)             perf_rd_ph[!l_iss ? ph : cd_isv ? PH_G_EPI_MEM : PH_G_LOAD_MEM] <=
                                            perf_rd_ph[!l_iss ? ph : cd_isv ? PH_G_EPI_MEM : PH_G_LOAD_MEM] + 1;
      if (mr_req && !mr_ready)            perf_rd_accept_wait <= perf_rd_accept_wait + 1;
      if (state == S_RD_WAIT)             perf_rd_latency <= perf_rd_latency + 1;
      if (mw_req && mw_ready)             perf_wr_beats <= perf_wr_beats + 1;
      if (mw_req && !mw_ready)            perf_wr_accept_wait <= perf_wr_accept_wait + 1;
      if (feed_go)                        perf_feed_valid <= perf_feed_valid + 1;
    end
  end

  // ------------------------------------------------------------------ control
  // the end of a summing / max pass (stage B or the sequential engine): wait until the pass's
  // contributions are all in, then for the fold and scalar to come out of vpu_fold, then start the
  // next pass
  task automatic pass_end();
    if (!v_folding) begin
      if (v_drain == 0) begin
        v_folding <= 1'b1;
        case ({x_fn, v_pass})
          {2'd0, 2'd0}: v_fwait <= 8'd5;
          {2'd0, 2'd1}: v_fwait <= 8'd51;
          {2'd1, 2'd0}: v_fwait <= 8'd16;
          default:      v_fwait <= 8'd75;     // LayerNorm p1, RMSNorm p0: rs
        endcase
      end
    end else if (v_fwait == 0) begin
      case ({x_fn, v_pass})
        {2'd0, 2'd0}: v_mx_or_mean <= f_mx;
        {2'd0, 2'd1}: v_sc2 <= f_r;
        {2'd1, 2'd0}: v_mx_or_mean <= f_mean;
        default:      v_sc2 <= f_rs;
      endcase
      v_folding <= 1'b0; b_wait <= 1'b0;
      lane <= '{default: 32'h0};
      cvl[CL_MAX:1] <= '0;
      v_gi_con <= 0; v_left <= x_nbr;
      // the sequential engine re-reads the row from the start; in overlap the reader belongs to
      // stage A (it runs ahead across rows) and must not be reset
      if (state != S_W_RUN) begin v_gi_iss <= 0; v_ileft <= x_nbr; v_kind_iss <= 0; end
      v_pass <= v_pass + 1;
    end
  endtask

  task automatic read_beat(input logic [63:0] addr, input state_t back);
    rd_addr <= {addr[63:4], 4'b0};
    ret     <= back;
    state   <= S_RD_REQ;
  endtask

  always_ff @(posedge clk) begin
    if (rst) begin
      state <= S_IDLE; halted <= 1'b0; done <= 1'b0; err_code <= E_OK; op_idx <= 0;
      cycles <= 0; vec_ok <= '{default: 1'b0}; e_busy <= 1'b0; f_tag <= 1'b0; l_valid <= 1'b0; l_st <= 1'b0;
      e_vsr <= '0; oq_tl <= 0; oq_dp <= 0; oq_hd <= 0; oq_cnt <= 0; oq_rdy <= '0;
      l_out <= 0; fq_h <= 0; fq_t <= 0; wr_busy <= 0; nv <= 0; vp <= 0; vg <= 0; cd_v <= 1'b0;
      v_occ <= 0; v_rcv <= 0; vq_t <= 0; vq_r <= 0; vq_h <= 0; xp_start <= 1'b0; x_xpu <= 1'b0; x_run <= 1'b0;
    end else begin
      // ---------------------------------------------------- chunk loader (runs beside the FSM)
      if (l_st) begin                            // the loader's tile starts: tables, slots, vectors
        int fi;
        fi = (l_fi < SK) ? int'(l_fi) : 0;
        abase[fi] <= x_conv ? f_p : fa;
        aiy[fi] <= 18'($signed({2'b0, f_oy}) * $signed({15'b0, x_cs[2:0]})) - 18'(x_cp);
        aix[fi] <= 18'($signed({2'b0, f_ox}) * $signed({15'b0, x_cs[2:0]})) - 18'(x_cp);
        wbase[fi] <= fw;
        fa <= fa + 64'(x_lda);
        fw <= fw + 64'(x_k);
        if (32'(f_ox) + 32'd1 >= x_ow) begin
          f_ox <= 0; f_oy <= f_oy + 16'd1; f_rowp <= f_rowp + x_swlda; f_p <= f_rowp + x_swlda;
          if (l_fi == 32'(R - 1)) begin nb_ox <= 0; nb_oy <= f_oy + 16'd1; nb_rowp <= f_rowp + x_swlda; nb_p <= f_rowp + x_swlda; end
        end else begin
          f_ox <= f_ox + 16'd1; f_p <= f_p + x_slda;
          if (l_fi == 32'(R - 1)) begin nb_ox <= f_ox + 16'd1; nb_oy <= f_oy; nb_rowp <= f_rowp; nb_p <= f_p + x_slda; end
        end
        if (l_fi == 0) begin
          lc <= 0; lk <= 0; vp <= 0; vg <= 0;
          // the loader's next tile, in the feed's order (registered: lm0 / ln0 hold for the tile)
          if (x_minner) begin
            if (lm0 + 32'(R) >= x_m) begin l_nm0 <= 0; l_nn0 <= ln0 + 32'(C); end
            else begin l_nm0 <= lm0 + 32'(R); l_nn0 <= ln0; end
          end else begin
            if (ln0 + 32'(C) >= x_n) begin l_nn0 <= 0; l_nm0 <= lm0 + 32'(R); end
            else begin l_nn0 <= ln0 + 32'(C); l_nm0 <= lm0; end
          end
          l_has_next <= !(lm0 + 32'(R) >= x_m && ln0 + 32'(C) >= x_n);
          for (int p = 0; p < NP; p++)
            l_live[p] <= p[0] ? (!lw_hold && (p >> 1) < C && ln0 + 32'(p >> 1) < x_n)
                              : (!la_hold && (p >> 1) < R && lm0 + 32'(p >> 1) < x_m);
          cw_ky <= 0; cw_kx <= 0; cw_cc <= 0; cw_off <= 0;
          lv_slo <= s_lo; lv_blo <= b_lo; lv_cols <= lt_cols; lv_rlo <= rv_lo; lv_rows <= lt_rows;
        end
        if (l_fi == 1) begin                     // the vectors, from the registered first bytes
          if (vec_need) begin
            nv <= s_nb + b_nb + r_nb; nvs <= s_nb; nvsb <= s_nb + b_nb;
            vs_t0[l_tag] <= lv_slo[63:4]; vb_t0[l_tag] <= lv_blo[63:4];
            vec_ok[l_tag] <= 1'b1; vec_n0[l_tag] <= ln0;
          end else begin
            nv <= r_nb; nvs <= 0; nvsb <= 0;
          end
          rv_t0[l_tag] <= lv_rlo[63:4];
        end
        if (l_fi + 1 >= 32'(SK)) begin
          l_st <= 1'b0; lp <= LP'(l_first);
          if (x_conv) begin l_gfill <= 1'b1; l_gi <= 0; end
        end else l_fi <= l_fi + 1;
      end else if (l_gfill) begin                // conv: the chunk group's offsets, one per cycle
        int gi;
        gi = (l_gi < LGE) ? int'(l_gi) : 0;
        gt_ky[gi] <= cw_ky; gt_kx[gi] <= cw_kx; gt_off[gi] <= cw_off;
        if (32'(cw_cc) + 32'd1 < x_cc) begin cw_cc <= cw_cc + 16'd1; cw_off <= cw_off + 64'd16; end
        else if (32'(cw_kx) + 32'd1 < x_ckw) begin cw_cc <= 0; cw_kx <= cw_kx + 16'd1; cw_off <= cw_off + x_d1; end
        else begin cw_cc <= 0; cw_kx <= 0; cw_ky <= cw_ky + 16'd1; cw_off <= cw_off + x_d2; end
        if (l_gi + 1 >= 32'(LGE)) l_gfill <= 1'b0; else l_gi <= l_gi + 1;
      end else if (l_adv) begin                  // on to the next tile (one ahead of the feed at most)
        lm0 <= l_nm0; ln0 <= l_nn0; l_tidx <= l_tidx + 1; l_tag <= !l_tag;
        l_base <= LB'(l_base + LB'(x_k >> 4)); l_st <= 1'b1; l_fi <= 0;
        // the new tile's bases: one row block down, back to the top, or the same rows
        if (l_nm0 == 0) begin
          l_arow0 <= x_a; fa <= x_a;
          cb_oy <= 0; cb_ox <= 0; cb_p <= x_porig; cb_rowp <= x_porig;
          f_oy <= 0; f_ox <= 0; f_p <= x_porig; f_rowp <= x_porig;
        end else if (l_nm0 != lm0) begin
          l_arow0 <= l_arow0 + x_rlda; fa <= l_arow0 + x_rlda;
          cb_oy <= nb_oy; cb_ox <= nb_ox; cb_p <= nb_p; cb_rowp <= nb_rowp;
          f_oy <= nb_oy; f_ox <= nb_ox; f_p <= nb_p; f_rowp <= nb_rowp;
        end else begin
          fa <= l_arow0;
          f_oy <= cb_oy; f_ox <= cb_ox; f_p <= cb_p; f_rowp <= cb_rowp;
        end
        if (l_nn0 == 0) begin l_wrow0 <= x_w; fw <= x_w; end
        else if (l_nn0 != ln0) begin l_wrow0 <= l_wrow0 + x_ck; fw <= l_wrow0 + x_ck; end
        else fw <= l_wrow0;
      end else if (l_tdone && !l_has_next) l_valid <= 1'b0;
      // issue stage: the candidate goes out (a read, or a padding zero beat)
      if (l_iss && cd_isv) begin
        fq_b[fq_t] <= LB'(l_tag); fq_k[fq_t] <= cd_k; fq_i[fq_t] <= cd_i; fq_t <= fq_t + 1'b1;
        vp <= vp + 8'd1;
      end else if (l_zfill) begin                // conv padding: the slot is zero, no read
        abuf[cd_b][cd_i]   <= 128'b0;
        a_full[cd_b][cd_i] <= 1'b1;
`ifndef TM_FPGA
        if (lh_first && !x_minner) hbuf[cd_i][cd_c] <= 128'b0;
`endif
      end else if (l_iss) begin
        if (cd_isw) w_pend[cd_b][cd_i] <= 1'b1; else a_pend[cd_b][cd_i] <= 1'b1;
        fq_b[fq_t] <= cd_b; fq_k[fq_t] <= cd_k; fq_i[fq_t] <= cd_i; fq_t <= fq_t + 1'b1;
        fq_c[fq_t] <= cd_c; fq_h1[fq_t] <= cd_h1;
      end
      // generate stage: the next read into the candidate register, then step the walk
      if (l_gtake) begin
        cd_v <= 1'b1; cd_isv <= l_isv; cd_isw <= !l_isv && l_isw; cd_zero <= !l_isv && l_zero;
        cd_b <= l_b; cd_c <= 16'(lcc); cd_addr <= l_addr;
        cd_h1 <= lh_first && (l_isw == x_minner);
        cd_i <= l_isv ? LP'((vg < nvs) ? vg : (vg < nvsb) ? vg - nvs : vg - nvsb) : LP'(lp >> 1);
        cd_k <= l_isv ? ((vg < nvs) ? 3'd2 : (vg < nvsb) ? 3'd3 : 3'd4) : {2'b0, l_isw};
        if (l_isv) vg <= vg + 8'd1;
        else if (!l_step_row) lk <= lk + 1;
        else begin
          lk <= 0;
          if (l_next >= NP) begin
            lc <= lc + 32'(LGE); lp <= LP'(l_first);
            if (x_conv && lc + 32'(LGE) < (x_k >> 4)) begin l_gfill <= 1'b1; l_gi <= 0; end
          end else lp <= LP'(l_next);
        end
      end else if (l_ctake) cd_v <= 1'b0;
      if (l_ret) begin
        if (fq_k[fq_h] == 3'd2)
          svb[fq_b[fq_h][0]][fq_i[fq_h]] <= mr_rdata;
        else if (fq_k[fq_h] == 3'd3)
          bvb[fq_b[fq_h][0]][fq_i[fq_h]] <= mr_rdata;
        else if (fq_k[fq_h] == 3'd4)
          rvb[fq_b[fq_h][0]][(int'(fq_i[fq_h]) < RVB) ? int'(fq_i[fq_h]) : 0] <= mr_rdata;
        else if (fq_k[fq_h] == 3'd1) begin
`ifndef TM_FPGA
          if (fq_h1[fq_h]) hbuf[fq_i[fq_h]][fq_c[fq_h]] <= mr_rdata;
`endif
          wbuf[fq_b[fq_h]][fq_i[fq_h]]   <= mr_rdata;
          w_full[fq_b[fq_h]][fq_i[fq_h]] <= 1'b1;
          w_pend[fq_b[fq_h]][fq_i[fq_h]] <= 1'b0;
        end else begin
`ifndef TM_FPGA
          if (fq_h1[fq_h]) hbuf[fq_i[fq_h]][fq_c[fq_h]] <= mr_rdata;
`endif
          abuf[fq_b[fq_h]][fq_i[fq_h]]   <= mr_rdata;
          a_full[fq_b[fq_h]][fq_i[fq_h]] <= 1'b1;
          a_pend[fq_b[fq_h]][fq_i[fq_h]] <= 1'b0;
        end
        fq_h <= fq_h + 1'b1;
      end
      l_out <= l_out + (LO + 1)'(l_iss) - (LO + 1)'(l_ret);
      // ---------------------------------------------------- row-op reader (runs beside the FSM)
      begin
        logic [LQ:0] pop;
        pop = (state == S_V_GB && v_rcv != 0 && v_left != 0) ? (LQ + 1)'(1)
            : (state == S_V_PASS && v_con && !v_hb) ? (LQ + 1)'(v_gs)
            : a_con ? (LQ + 1)'(1) : '0;
        if (v_iss) begin
          vq_k[vq_t] <= (state == S_V_GB) ? v_gb : v_kind_iss;
          vq_t <= vq_t + 1'b1;
          if (state == S_W_RUN) begin           // row data only, running on across rows
            if (v_ileft <= 1) begin v_gi_iss <= 0; v_ileft <= x_nbr; v_irow <= v_irow + 1; v_iaddr <= v_iaddr + 64'(x_lda); end
            else begin v_gi_iss <= v_gi_iss + 1; v_ileft <= v_ileft - 1; end
          end else if (state == S_V_GB || v_kind_iss == v_gs - 2'd1) begin
            v_gi_iss <= v_gi_iss + 1; v_ileft <= v_ileft - 1; v_kind_iss <= 2'd0;
          end else v_kind_iss <= v_kind_iss + 2'd1;
        end
        if (v_ret) begin
`ifndef TM_FPGA
          vq_d[vq_r] <= mr_rdata;
`endif
          vq_r <= vq_r + 1'b1;
        end
        vq_h  <= vq_h + LQ'(pop);
        v_occ <= v_occ + (LQ + 1)'(v_iss) - pop;
        v_rcv <= v_rcv + (LQ + 1)'(v_ret) - pop;
      end
      // ---------------------------------------------------- row-engine steps, lanes, stage A
      // step issue: the operands into the lane registers; output steps take a queue slot
      cvl[0] <= v_con && !v_last;
      cg4[0] <= v_gi_con[1:0];
      cn[0]  <= 3'(v_n);
      for (int k = 1; k <= CL_MAX; k++) begin cg4[k] <= cg4[k-1]; cn[k] <= cn[k-1]; end
      cvl[CL_MAX:1] <= cvl[CL_MAX-1:0];
      c_v <= cvl[v_clat]; c_g4 <= cg4[v_clat]; c_n <= cn[v_clat]; c_z <= v_clat == 0;
      vlc_r <= vlc; sx_r <= sx;
      skout <= v_con && v_last;
      if (v_con) begin
        for (int i = 0; i < NS; i++) begin
          // v0.4 causal softmax: masked elements enter as -inf (out of the max) and their exp
          // as +0 (skeep, carried down the lane)
          skeep[i] <= v_gi_con * 32'd4 + 32'(i) < v_vlen;
          sx[i] <= (v_gi_con * 32'd4 + 32'(i) >= v_vlen) ? 32'hFF800000 : vxw[i / 4][32 * (i % 4) +: 32];
          sg[i] <= vgw[i / 4][32 * (i % 4) +: 32];
          sb[i] <= vbw[i / 4][32 * (i % 4) +: 32];
        end
        sc1 <= v_mx_or_mean; ssc <= v_sc2;
        if (v_last) begin
`ifndef TM_FPGA
          oq_a[oq_tl] <= v_yrow + (64'(v_gi_con) << (32'd2 + 32'(x_esl)));
          oq_l[oq_tl] <= v_n << (32'd2 + 32'(x_esl));
`endif
          oq_rdy[oq_tl] <= 1'b0; oq_tl <= oq_tl + 1'b1;
        end else begin
          v_drain <= 8'(v_clat + 32'd6);
        end
        v_gi_con <= v_gi_con + v_n; v_left <= v_left - v_n;
      end else if (v_drain != 0) v_drain <= v_drain - 8'd1;
      for (int l = 0; l < 16; l++)
        if (v_con && !v_last && c_sum && step_lanes[l]) v_cool[l] <= 2'd2;   // next step 3 cycles on (forwarded)
        else if (v_cool[l] != 0) v_cool[l] <= v_cool[l] - 2'd1;
      // accumulators: a 3-cycle add (result written as it completes) or a 1-cycle max
      for (int l = 0; l < 16; l++) begin
        lane_p[l] <= {lane_p[l][1:0], lane_in[l] && c_sum};
        if (lane_in[l] && !c_sum) lane[l] <= fmax(lane[l], lane_c[l]);
        else if (lane_p[l][2]) lane[l] <= lane_y[l];
      end
      // stage A: consumed groups' x, its x*x (RMSNorm) three cycles later; accumulate
      ax_v <= a_con;
      ax_g4 <= a_gi[1:0];
      if (a_con)
        for (int i = 0; i < 4; i++) ax[i] <= vq_q[0][32 * i +: 32];
      for (int l = 0; l < 16; l++) begin
        a_p[l] <= {a_p[l][1:0], a_in[l] && x_fn != 2'd0};
        if (a_in[l] && x_fn == 2'd0) a_lane[l] <= fmax(a_lane[l], a_lc[l]);
        else if (a_p[l][2]) a_lane[l] <= a_ly[l];
      end
      a_in <= a_inc; a_lc <= a_lcc;
      if (a_con) a_drain <= 8'd10; else if (a_drain != 0) a_drain <= a_drain - 8'd1;
      if (a_fold && a_fwait != 0) a_fwait <= a_fwait - 8'd1;
      if (v_folding && v_fwait != 0) v_fwait <= v_fwait - 8'd1;
      if (inv_wait != 0) inv_wait <= inv_wait - 8'd1;
      // epilogue engine: one tile row per cycle the writer can take, beside the next tile's feed
      if (e_busy) begin
        e_adv <= e_adv + 32'(arr_en);
        if (e_ei >= e_nrow) e_busy <= 1'b0;
      end
      // segment writer: one beat per accepted cycle (the epilogue hands it the next row)
      if (wr_acc) begin
        wr_b <= wr_b + 8'd1;
        if (wr_b == wr_nb - 8'd1) wr_busy <= 1'b0;
      end
      // epilogue launch: the row into the lane registers, a queue slot for it
      if (e_go) begin
        for (int j = 0; j < C; j++) begin
          q_acc[j] <= acc[e_bank][(e_ei < R) ? e_ei : 0][j];
          q_s[j] <= e_s[j]; q_b[j] <= e_b[j];
        end
        q_r <= e_r[(e_ei < R) ? e_ei : 0];
`ifndef TM_FPGA
        oq_a[oq_tl] <= r_addr; oq_l[oq_tl] <= r_len;
`endif
        oq_rdy[oq_tl] <= 1'b0; oq_tl <= oq_tl + 1'b1;
        e_ei <= e_ei + 1;
        e_yaddr <= e_yaddr + 64'(x_ldy);
      end
      // rows of an earlier descriptor have all left (its end waits for the queue to empty), but
      // their bits run on down the line: clear it at each dispatch, the latency tap may move
      // (shared by the row engine's output steps; the tap is the descriptor's latency)
      e_vsr <= (state == S_CHECK) ? '0 : {e_vsr[EL_MAX-1:0], e_go || (v_con && v_last) || xp_lq_go};
      if (xp_lq_go) begin
`ifndef TM_FPGA
        oq_a[oq_tl] <= xp_lq_addr; oq_l[oq_tl] <= xp_lq_len;
`endif
        oq_rdy[oq_tl] <= 1'b0; oq_tl <= oq_tl + 1'b1;
      end
      if (e_emerge) begin
`ifndef TM_FPGA
        oq_d[oq_dp] <= x_gemm ? e_row : x_xpu ? xp_row : v_row;
`endif
        oq_rdy[oq_dp] <= 1'b1; oq_dp <= oq_dp + 1'b1;
      end
      if (oq_pop) begin                          // the writer takes the head slot
        wr_busy <= 1'b1; wr_b <= 8'd0;
        wr_data <= oq_dh; wr_len <= oq_lh; wr_addr <= oq_ah;
        wr_nb   <= 8'((32'(oq_ah[3:0]) + oq_lh + 32'd15) >> 4);
        oq_rdy[oq_hd] <= 1'b0; oq_hd <= oq_hd + 1'b1;
      end
      oq_cnt <= oq_cnt + (LQO + 1)'(e_go || (v_con && v_last) || xp_lq_go) - (LQO + 1)'(oq_pop);

      if (state != S_IDLE && state != S_HALT) cycles <= cycles + 1;
      case (state)
        S_IDLE: if (start) begin
          qa <= q_addr; qn <= q_len; qi <= 0; q_daddr <= q_addr; state <= S_NEXT;
        end

        // ---------------------------------------------------- fetch and check
        S_NEXT: begin
          if (qi >= qn) begin
            done <= 1'b1; halted <= 1'b1; err_code <= E_OK; op_idx <= qn; state <= S_HALT;
          end else if (66'(q_daddr) + 66'd64 > 66'(MEM_BYTES)) begin
            halted <= 1'b1; err_code <= E_RANGE; op_idx <= qi; state <= S_HALT;
          end else begin
            beat <= 0; state <= S_FETCH;
          end
        end
        S_FETCH: read_beat(q_daddr + {58'b0, beat, 4'b0}, S_FETCH_GOT);
        S_FETCH_GOT: begin
          desc[128 * beat +: 128] <= rd_data;
          if (beat == 2'd3) begin
            crc <= 32'hFFFFFFFF; cb <= 0; state <= S_CRC;
          end else begin
            beat <= beat + 2'd1; state <= S_FETCH;
          end
        end
        S_CRC: begin
          crc <= crc_byte(crc, crc_in);
          cb  <= cb + 7'd1;
          if (cb == 7'd63) state <= S_CHKW;
        end
        S_CHKW: if (ck_st == 3'd7) state <= S_CHK4;   // the check sequencer is done
        S_CHK4: begin                           // the rules
          r_chk <= chk;
          state <= S_CHECK;
        end
        S_CHECK: begin
          if (r_chk != E_OK) begin
            halted <= 1'b1; err_code <= r_chk; op_idx <= qi; state <= S_HALT;
          end else begin
            x_gemm  <= ARR != 0 && f_gm && !f_gv; x_bias <= f_bias; x_perch <= (f_qm == 3'd2 || f_qm == 3'd3);
            x_rowcol <= f_gemm && f_qm == 3'd3; x_rs <= d_base + {36'b0, d_hrsv, 4'b0};
            x_conv <= f_conv; x_cw <= g_w; x_ch <= g_h; x_ckw <= g_kw; x_cs <= g_s; x_cp <= g_p;
            x_ow <= g_ow; x_cc <= g_cin >> 4;
            x_act   <= f_act;  x_out  <= f_out;
            x_a <= d_base + 64'(d_aoff); x_w <= d_base + 64'(d_woff); x_y <= d_base + 64'(d_yoff);
            x_s <= d_base + 64'(d_soff); x_b <= d_base + 64'(d_boff);
            x_m <= (ARR == 0 && f_gv && 32'(d_m) > f_gpr) ? f_gpr : 32'(d_m); x_n <= 32'(d_n);
            gp_rem <= (ARR == 0 && f_gv && 32'(d_m) > f_gpr) ? 32'(d_m) - f_gpr : 32'd0; gp_pr <= f_gpr; x_k <= 32'(d_k); feed_last <= 32'(d_k) + 32'(SK) - 32'd2;
            x_reuse  <= 32'(d_k >> 4) <= 32'(KR);
            x_minner <= r_minner;
            x_lda <= f_lda; x_ldy <= f_ldy; x_es <= f_es; x_osc <= d_oscale;
            x_esl <= (f_out == 2'd1) ? 2'd1 : (f_out == 2'd2) ? 2'd0 : 2'd2;
            x_rldy <= r_rldy; ymr <= 0;
            // x_fn 3 (internal): RELU, the activation-only single pass of the row engine
            x_row <= f_row && !f_quant; x_xpu <= f_x; xp_start <= f_x; x_run <= f_x;
            xp_op <= f_elt ? 3'd0 : f_rope ? 3'd1 : f_fdot ? (d_ctrl[0] ? 3'd3 : 3'd2) : f_gv ? 3'd5 : 3'd4;
            xp_qm <= f_qm[1:0]; xp_r <= d_base + {36'b0, d_hrsv, 4'b0};
            xp_eop <= d_ctrl[1:0]; xp_bmode <= d_hrsv[1:0]; xp_arow <= d_hrsv[2];
            xp_g <= 32'(d_group); xp_ldb <= f_ldb; xp_c <= d_base + 64'(d_boff);
            x_fn <= (d_op == OP_RELU) ? 2'd3 : (d_ctrl[1:0] == 2'd3) ? 2'd2 : d_ctrl[1:0];
            x_l2 <= f_row && d_ctrl[2:0] == 3'd3;
            x_causal <= f_row && d_ctrl[2:0] == 3'd0 && d_k != 0; x_kst <= 32'(d_k);
            v_vlen <= 32'(d_n);                   // every element (S_V_ROW narrows it for a causal row)
            x_eps <= d_woff;
            x_nbr <= 32'(d_n) >> 2;
            x_fits <= r_fits;
            x_ov   <= r_ov;
            x_gb0  <= r_gb0; x_gb1 <= r_gb0 + (32'(d_n) >> 2);
            a_row <= 0; a_more <= 1'b1; a_gi <= 0; a_gleft <= 32'(d_n) >> 2; a_wad <= 0; a_buf <= 1'b0; b_act <= 1'b0; b_buf <= 1'b0; v_irow <= 0;
            a_lane <= '{default: (d_ctrl[1:0] == 2'd0) ? 32'hFF800000 : 32'h0};
            a_fold <= 1'b0; a_ready <= 1'b0; a_drain <= 0; b_wait <= 1'b0; v_folding <= 1'b0;
            v_drain <= 0; v_cool <= '{default: 2'd0}; cvl <= '0; c_v <= 1'b0;
            v_ynext <= d_base + 64'(d_yoff); v_anext <= d_base + 64'(d_aoff); v_iaddr <= d_base + 64'(d_aoff);
            v_rm <= 0; v_gb <= 2'd1; v_gi_iss <= 0; v_gi_con <= 0; v_left <= 32'(d_n) >> 2; v_ileft <= 32'(d_n) >> 2;
            vec_ok <= '{default: 1'b0};
            m0 <= 0; n0 <= 0; ei <= 0;
            // the loader starts the first tile next cycle (after x_* are set), in the bank the
            // feed's first tile will take
            lm0 <= 0; ln0 <= 0; l_tidx <= 1; f_tidx <= 0; l_base <= 0; f_bnx <= 0;
            l_valid <= ARR != 0 && f_gm && !f_gv; l_st <= ARR != 0 && f_gm && !f_gv; l_tag <= !f_tag; nv <= 0; vp <= 0; vg <= 0; lc <= 0; lk <= 0; cd_v <= 1'b0;
            l_fi <= 0; l_gfill <= 1'b0;
            x_rlda <= r_rlda; x_ck <= r_ck; x_kc <= 32'(d_k) >> 4;
            x_slda <= r_slda;
            x_swlda <= r_swlda;
            x_d1 <= r_d1;
            x_d2 <= r_d2;
            x_porig <= r_porig;
            l_arow0 <= d_base + 64'(d_aoff); fa <= d_base + 64'(d_aoff);
            l_wrow0 <= d_base + 64'(d_woff); fw <= d_base + 64'(d_woff);
            cb_oy <= 0; cb_ox <= 0; f_oy <= 0; f_ox <= 0;
            cb_p <= r_porig; cb_rowp <= r_porig; f_p <= r_porig; f_rowp <= r_porig;
            for (int b = 0; b < NB; b++) begin
              a_full[b] <= '0; a_pend[b] <= '0; w_full[b] <= '0; w_pend[b] <= '0;
            end
            v_after_inv <= r_gbfit ? S_V_GB : S_V_ROW;
            inv_wait <= 8'd42;                    // i2f (1) + recip (38) + margin
            state <= f_x ? S_X_RUN
                   : (ARR != 0 && f_gm) ? S_G_TILE
                   : f_norm ? S_V_INV            // 1/N first (the norms)
                   : f_row  ? ((SFU == 0 && r_wfit && d_k == 0) ? S_W_RUN : S_V_ROW)   // FPGA: no overlap
                   : S_V_ROW;                     // RELU (activation): one pass of the row engine
          end
        end

        // ---------------------------------------------------- GEMM
        S_G_TILE: begin
          if (x_minner ? n0 >= x_n : m0 >= x_m) begin
            if (!wr_busy && !e_busy && oq_cnt == 0) begin qi <= qi + 1; q_daddr <= q_daddr + 64'd64; state <= S_NEXT; end   // last row written first
          end else begin
            state <= S_G_CLR;
          end
        end
        S_G_CLR: begin                          // the feed starts the tile, in the other bank; its
          cnt <= 0; f_tag <= !f_tag;            // chunks may already be loading (prefetch)
          f_base <= f_bnx; f_bnx <= LB'(f_bnx + LB'(x_k >> 4)); f_tidx <= f_tidx + 1;
          for (int i = 0; i < R; i++) begin fa_in[i] <= (i == 0); fa_sl[i] <= f_bnx; rlive[i] <= m0 + 32'(i) < x_m; end
          for (int j = 0; j < C; j++) begin fw_in[j] <= (j == 0); fw_sl[j] <= f_bnx; clive[j] <= n0 + 32'(j) < x_n; end
          state <= S_G_FEED;
        end
        S_G_FEED: if (feed_go) begin
          // a row that takes the last byte of a chunk frees its slot for chunk + NB
          for (int i = 0; i < R; i++) begin
            int kc;
            kc = int'(cnt) - i;
            if (rlive[i] && fa_in[i] && (kc & 15) == 15) a_full[fa_sl[i]][i] <= 1'b0;
            // the next step's window and slot (k + 1)
            fa_in[i] <= kc + 1 >= 0 && kc + 1 < int'(x_k);
            fa_sl[i] <= LB'(f_base + LB'((kc + 1) >> 4));
          end
          for (int j = 0; j < C; j++) begin
            int kc;
            kc = int'(cnt) - j;
            if (clive[j] && fw_in[j] && (kc & 15) == 15) w_full[fw_sl[j]][j] <= 1'b0;
            fw_in[j] <= kc + 1 >= 0 && kc + 1 < int'(x_k);
            fw_sl[j] <= LB'(f_base + LB'((kc + 1) >> 4));
          end
          // No drain: PE (i, j) adds a[i][K-1] at step K-1+i+j, so row i is final from step
          // K+i+C-1 on. The epilogue reads row i at step K+SK-1+i or later (rows in order, at most
          // one per step), and SK >= C.
          if (cnt == feed_last) begin
            f_adv <= 0; state <= S_G_EPI;
          end else cnt <= cnt + 1;
        end
        S_G_EPI: begin                          // hand the tile to the epilogue engine once it is free
          f_adv <= f_adv + 1;                   // the array advances in every non-feed cycle
          if (!e_busy) begin
            e_busy <= 1'b1; e_bank <= f_tag; e_ei <= 0; e_adv <= f_adv + 1;
            e_m0 <= m0; e_n0 <= n0; e_cols <= t_cols; e_r <= t_r; e_nrow <= (x_m - m0 < 32'(R)) ? x_m - m0 : 32'(R); e_s <= t_s; e_b <= t_b;
            e_yaddr <= e_ynx;
            state <= S_G_TILE_END;
          end
        end
        S_G_TILE_END: begin
          if (x_minner) begin
            if (m0 + R >= x_m) begin m0 <= 0; ymr <= 0; n0 <= n0 + C; end
            else begin m0 <= m0 + R; ymr <= ymr + x_rldy; end
          end else begin
            if (n0 + C >= x_n) begin n0 <= 0; m0 <= m0 + R; ymr <= ymr + x_rldy; end else n0 <= n0 + C;
          end
          state <= S_G_TILE;
        end

        // ---------------------------------------------------- v0.3 row ops
        S_V_INV: if (inv_wait == 0) begin       // 1/N = recip(i2f(N)) is out of its pipeline
          x_invn <= x_l2 ? 32'h3F800000 : inv_out; state <= v_after_inv;
        end
        S_V_GB: begin                           // gamma (then beta) into hbuf, once per descriptor
          if (v_rcv != 0 && v_left != 0) begin
            logic [31:0] li;
            li = ((v_gb == 2'd1) ? x_gb0 : x_gb1) + v_gi_con;
`ifndef TM_FPGA
            hbuf[(li / KR) < SK ? li / KR : 0][li % KR] <= vq_q[0];
`endif
            v_gi_con <= v_gi_con + 1; v_left <= v_left - 1;
          end
          if (v_left == 0) begin
            v_gi_iss <= 0; v_ileft <= x_nbr; v_gi_con <= 0; v_left <= x_nbr;
            if (v_gb == 2'd1 && x_bias) v_gb <= 2'd2;
            else state <= x_ov ? S_W_RUN : S_V_ROW;
          end
        end
        S_V_ROW: begin
          if (v_rm >= x_m) begin
            state <= S_V_NEXT;
          end else begin
            v_pass <= 0; v_gi_iss <= 0; v_ileft <= x_nbr; v_gi_con <= 0; v_left <= x_nbr; v_kind_iss <= 0;
            v_vlen <= (x_causal && x_kst + v_rm < x_n) ? x_kst + v_rm : x_n;
            lane <= '{default: ((x_fn == 2'd0) ? 32'hFF800000 : 32'h0)};
            v_arow <= v_anext; v_anext <= v_anext + 64'(x_lda);
            v_yrow <= v_ynext; v_ynext <= v_ynext + 64'(x_ldy);
            cvl[CL_MAX:1] <= '0;
            state <= S_V_PASS;
          end
        end
        S_V_PASS: begin
`ifndef TM_FPGA
          if (v_con && x_fits && v_pass == 0)   // keep the row for the later passes (FPGA: hw_*)
            hbuf[(v_gi_con / KR) < SK ? v_gi_con / KR : 0][v_gi_con % KR] <= vx;
`endif
          if (!b_wait && v_left == 0) begin
            if (v_last) begin                   // all output steps issued: next row
              v_rm <= v_rm + 1; state <= S_V_ROW;
            end else b_wait <= 1'b1;
          end
          if (b_wait) pass_end();
        end
        S_V_NEXT: if (!wr_busy && oq_cnt == 0) begin qi <= qi + 1; q_daddr <= q_daddr + 64'd64; state <= S_NEXT; end
        S_X_RUN: begin                          // v0.4: the extension unit runs the descriptor
          if (xp_start) xp_start <= 1'b0;
          else if (!xp_busy && !wr_busy && oq_cnt == 0) begin
            if (gp_rem != 0) begin               // ARR 0: the GEMM's next rows, a GEMV pass
              x_a <= x_a + 64'(gp_pr * x_lda); x_y <= x_y + 64'(gp_pr * x_ldy); xp_r <= xp_r + 64'(gp_pr << 2);
              x_m <= (gp_rem < gp_pr) ? gp_rem : gp_pr;
              gp_rem <= gp_rem - ((gp_rem < gp_pr) ? gp_rem : gp_pr);
              xp_start <= 1'b1;
            end else begin
              qi <= qi + 1; q_daddr <= q_daddr + 64'd64; state <= S_NEXT; x_run <= 1'b0;
            end
          end
        end
        S_W_RUN: begin
          // stage A: pass 0 of row a_row into region a_buf
          if (a_con) begin
`ifndef TM_FPGA
            hbuf[a_wad / KR < SK ? a_wad / KR : 0][a_wad % KR] <= vq_q[0];
`endif
            a_gi <= a_gi + 1; a_gleft <= a_gleft - 1; a_wad <= a_wad + 1;
          end
          if (a_more && a_gleft == 0 && a_drain == 0 && !a_fold && !a_ready) begin
            a_fold <= 1'b1;                     // its contributions are in: fold, A's scalar
            a_fwait <= (x_fn == 2'd0) ? 8'd5 : (x_fn == 2'd1) ? 8'd16 : 8'd75;
          end
          if (a_fold && a_fwait == 0) begin
            a_fold <= 1'b0; a_ready <= 1'b1;
            a_c1 <= (x_fn == 2'd0) ? af_mx : af_mean;
            a_sc <= af_rs;
          end
          // stage B: VG groups per step; pass ends wait for the drain and the fold
          if (b_act && !b_wait && v_left == 0) begin
            if (v_last) b_act <= 1'b0;          // all output steps issued
            else b_wait <= 1'b1;
          end
          if (b_wait) pass_end();
          // handoff: A's row (with its scalar) to B; A starts the next row in the other region
          if (a_ready && !b_act) begin
            if (x_fn == 2'd2) v_sc2 <= a_sc; else v_mx_or_mean <= a_c1;
            b_act <= 1'b1; b_buf <= a_buf; v_rm <= a_row; v_pass <= 2'd1; v_gi_con <= 0; v_left <= x_nbr;
            lane <= '{default: 32'h0};
            cvl[CL_MAX:1] <= '0;
            v_yrow <= v_ynext; v_ynext <= v_ynext + 64'(x_ldy);
            a_row <= a_row + 1; a_more <= a_row + 1 < x_m; a_buf <= !a_buf; a_gi <= 0; a_gleft <= x_nbr;
            a_wad <= a_buf ? 32'd0 : x_nbr; a_ready <= 1'b0;
            a_lane <= '{default: ((x_fn == 2'd0) ? 32'hFF800000 : 32'h0)};
          end
          if (!a_more && !b_act && !a_ready) state <= S_V_NEXT;
        end

        // ---------------------------------------------------- memory subroutines
        S_RD_REQ:  if (mr_ready) state <= S_RD_WAIT;
        S_RD_WAIT: if (mr_rvalid) begin rd_data <= mr_rdata; state <= ret; end

        S_HALT: ;
        default: state <= S_HALT;
      endcase
    end
  end

  // ------------------------------------------------------------------ shared SFU (FPGA)
  // One fpu_act for the epilogue, the row engine and the extension unit (tm_sfu.sv); only one of
  // them runs a descriptor at a time, and a descriptor waits for the queue to empty before the
  // next starts, so the SFU follows the active unit. Each unit spaces its long-act steps W cycles
  // apart (e_sok, v_sok, the extension unit's own gate).
  always_ff @(posedge clk)
    if (rst) begin e_scnt <= 0; v_scnt <= 0; end
    else begin
      e_scnt <= (e_go && (x_gs || x_sp)) ? 32'(C - 1) : (e_scnt != 0) ? e_scnt - 1 : 0;
      v_scnt <= (v_con && v_sneed) ? 32'(NS - 1) : (v_scnt != 0) ? v_scnt - 1 : 0;
    end
`ifdef TM_FPGA
  logic        e_sv, v_sv, v_sv3;          // a step's act inputs (exp arguments: v_sv3) are at the lanes now
  logic [31:0] e_sx, v_sx, v_sxe, sfu_x, sfu_ya1, sfu_yl, sfu_ys, sfu_ye;
  fpu_dly #(1, 12) u_esv (.clk, .d(e_go), .q(e_sv));          // y1: 11 after the launch registers
  always_ff @(posedge clk) v_sv <= v_con && v_sneed;         // sx: the issue registers (only steps that use
                                                             // the SFU: a pass-0 step must not reload the port)
  tm_sfu_port #(C)  u_esp (.clk, .v(e_sv), .x(e_ax), .sx(e_sx), .sy(sfu_y), .y(e_ay));
  tm_sfu_port #(NS) u_vsp (.clk, .v(v_sv), .x(sx), .sx(v_sx), .sy(sfu_y), .y(v_sy));
  fpu_dly #(1, 3) u_vsv3 (.clk, .d(v_sv), .q(v_sv3));         // x - mx: 3 after the issue registers
  tm_sfu_port #(NS) u_xsp (.clk, .v(v_sv3), .x(v_xf), .sx(v_sxe), .sy(sfu_ye), .y(v_ye));
  assign sfu_x = x_gemm ? e_sx : x_xpu ? xp_sx : v_sx;
  fpu_act u_sfu (.clk, .x(sfu_x), .act(x_act), .y1(sfu_ya1), .ylong(sfu_yl), .ysp(sfu_ys),
                 .esel(x_row && x_fn == 2'd0), .xe(v_sxe), .ye(sfu_ye));
  assign sfu_y = x_sp ? sfu_ys : sfu_yl;
`endif

`ifdef TM_FPGA
  // ------------------------------------------------------------------ hbuf (FPGA: banks)
  // the writers, as in the ASIC build's hbuf assignments; they never coincide (the loader's
  // returns and padding zeros are GEMM-time and a zero waits for a return, l_zfill; the row
  // engine's are in S_V_GB / S_V_PASS, where the loader takes no returns; no overlap, S_W_RUN)
  always_comb begin
    logic [31:0] li;
    hw_e = 1'b0; hw_b = 0; hw_c = 0; hw_d = '0;
    li = ((v_gb == 2'd1) ? x_gb0 : x_gb1) + v_gi_con;
    if (l_ret && fq_k[fq_h] <= 3'd1 && fq_h1[fq_h]) begin
      hw_e = 1'b1; hw_b = 32'(fq_i[fq_h]); hw_c = 32'(fq_c[fq_h]); hw_d = mr_rdata;
    end else if (l_zfill && lh_first && !x_minner) begin
      hw_e = 1'b1; hw_b = 32'(cd_i); hw_c = 32'(cd_c);
    end else if (state == S_V_GB && v_rcv != 0 && v_left != 0) begin
      hw_e = 1'b1; hw_b = (li / KR) < SK ? li / KR : 0; hw_c = li % KR; hw_d = vq_q[0];
    end else if (state == S_V_PASS && v_con && x_fits && v_pass == 0) begin
      hw_e = 1'b1; hw_b = (v_gi_con / KR) < SK ? v_gi_con / KR : 0; hw_c = v_gi_con % KR; hw_d = vx;
    end
  end
  for (genvar b = 0; b < SK; b++) begin : g_hb
    logic [31:0] ra [HNR];
    always_comb begin
      int k;
      k = int'(cnt) - b;                                   // row / column b's K index
      ra[0] = ((k >> 4) < KR) ? 32'(k >> 4) : 32'd0;
      for (int r = 0; r < 3 * VG; r++) ra[1 + r] = hr_l[r] % KR;
    end
    tm_lutram #(.W(128), .D(KR), .NR(HNR)) u_m (.clk, .we(hw_e && hw_b == 32'(b)), .wa(hw_c), .wd(hw_d),
                                               .ra, .rd(hb_q[b]));
    assign hf_q[b] = hb_q[b][0];
  end
  // the row engine's beats (addresses here, not in its big always_comb, which takes the data:
  // no false combinational loop through the memories)
  always_comb
    for (int k = 0; k < VG; k++) begin
      hr_l[k]          = ((x_ov && b_buf) ? x_nbr : 32'd0) + v_gi_con + 32'(k);   // v_rbase
      hr_l[VG + k]     = x_gb0 + v_gi_con + 32'(k);
      hr_l[2 * VG + k] = x_gb1 + v_gi_con + 32'(k);
    end
  always_comb
    for (int r = 0; r < 3 * VG; r++)
      hr_q[r] = hb_q[(hr_l[r] / KR) < SK ? hr_l[r] / KR : 0][1 + r];

  // the row reader's FIFO: one write port (returns), three read ports (head, head + 1, + 2)
  logic [31:0]  vq_ra [3];
  for (genvar k = 0; k < 3; k++) begin : g_vqa
    assign vq_ra[k] = 32'(LQ'(vq_h + LQ'(k)));        // wraps at VQ
  end
  tm_lutram #(.W(128), .D(VQ), .NR(3)) u_vq (.clk, .we(v_ret && !rst), .wa(32'(vq_r)), .wd(mr_rdata),
                                            .ra(vq_ra), .rd(vq_q));

  // the output queue: rows as they emerge (oq_dp), address and length as a slot is taken (oq_tl:
  // the row engine's, the epilogue's or the extension unit's, never two at once)
  logic [31:0]     oq_ra [1];
  logic [8*RB-1:0] oq_drd [1];
  logic [95:0]     oq_alrd [1], oq_alw;
  assign oq_ra[0] = 32'(oq_hd);
  always_comb
    if (xp_lq_go)  oq_alw = {xp_lq_addr, xp_lq_len};
    else if (e_go) oq_alw = {r_addr, r_len};
    else           oq_alw = {v_yrow + (64'(v_gi_con) << (32'd2 + 32'(x_esl))), v_n << (32'd2 + 32'(x_esl))};
  tm_lutram #(.W(8 * RB), .D(OQ), .NR(1)) u_oqd (.clk, .we(e_emerge && !rst), .wa(32'(oq_dp)),
                                                .wd(x_gemm ? e_row : x_xpu ? xp_row : v_row), .ra(oq_ra), .rd(oq_drd));
  tm_lutram #(.W(96), .D(OQ), .NR(1)) u_oqal (.clk, .we(((v_con && v_last) || e_go || xp_lq_go) && !rst),
                                             .wa(32'(oq_tl)), .wd(oq_alw), .ra(oq_ra), .rd(oq_alrd));
  assign oq_dh = oq_drd[0];
  assign {oq_ah, oq_lh} = oq_alrd[0];
`endif
endmodule
