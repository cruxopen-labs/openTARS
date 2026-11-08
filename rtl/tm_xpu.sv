// TARS-mini v0.4 extension unit: ELT (elementwise), ROPE, FDOT (fp32 dot products, NT and NN) and
// the row QUANT (per-token INT8). One descriptor at a time, beside the row engine (which keeps
// softmax / LayerNorm / RMSNorm / L2NORM / activations).
//
// Structure: a reader walks the descriptor's beats in a fixed order (addresses from running
// pointers, no multiply) and tags each with what it is and, on a step's last beat, the step's
// meaning (launch address and length, lane group, pass end). Returns arrive in order into a FIFO
// with those tags, so the consumer needs no walker of its own: it collects a step's beats and
// fires the step. Steps are one 4-element group:
//   ELT   y = act(A) op B (MUL, ADD, SUB) or C + act(A) * B; B / C delayed by the activation's
//         latency (a circular-buffer delay line); latency act + 3 (+ 3 for MULADD)
//   ROPE  y = x c + -(xp s) / x c + xp s on the rotary groups, ftz(x) elsewhere; latency 6
//   FDOT  products a * b into 16 lane accumulators (NT: lane k mod 16; NN: lane = column within
//         the 16-column chunk, a broadcast); a pass ends with a drain, then NT folds the lanes
//         pairwise (0+1, 2+3, ...) into one output, NN reads the 16 lanes out as 4 groups
//   QUANT pass 0: running max of |x| per slot; then amax, inv = 127 * recip(amax), s = amax *
//         fp32(1/127); the row scale s goes out (raw fp32), pass 1 re-reads x: q8(x * inv)
//   GEMV  a GEMM with M <= MB rows (decode: M = 1; a prefill chunk: M tokens): x (M rows of K INT8)
//         into M 256-beat buffers once, then W streamed row after row (contiguous, one 16-byte beat
//         a cycle, the read port's rate): per row 16 INT8 products a beat summed exactly into the
//         row's INT32 accumulator, so each W beat serves M rows (16 M MACs a cycle); at a channel's
//         last beat the contract's epilogue (i2f, row scale, column scale, bias, activation,
//         conversion) on one lane, the M rows' sums one a cycle (the first launches with the beat,
//         a sequencer launches the others); M output elements per channel. On the systolic array
//         M rows use R x C MACs; here the weights stream once at the port's 16 bytes a cycle
// Every output goes through the output queue: an element step takes a slot when it issues and its
// data reaches the conversion stage pl cycles later; a readout (FDOT result, QUANT scale) takes a
// slot as it enters the conversion stage. One descriptor uses one of the two, or (QUANT) both with
// pl = 0, so its data emerges at one fixed latency after the launch: x_row (registered) holds the
// entry launched lat + 1 cycles ago (lat = pl + conversion), the convention of the core's epilogue.
// A launch (lq_go) is the cycle the step's values are in the issue registers (or a readout's in
// ro_v); the core takes a queue slot for it then.
module tm_xpu #(
  parameter int VQ = 128,                 // reads in flight / buffered (power of two)
  parameter int RB = 64,                  // output segment bytes
  parameter int MB = 8,                   // GEMV: rows at most (the core sends M <= MB, M <= K / 16)
  parameter int RW = 1                    // GEMV: W in requests of 16 RW bytes (the port's wide reads; 1, 2 or 4)
) (
  input  logic         clk,
  input  logic         rst,
  // descriptor (held while busy)
  input  logic         start,
  input  logic [2:0]   op,                // 0 ELT, 1 ROPE, 2 FDOT NT, 3 FDOT NN, 4 QUANT, 5 GEMV
  input  logic [1:0]   qm,                // GEMV: 1 per-tensor scale, 2 per-channel, 3 per-channel + row scale
  input  logic         bias,              // GEMV: bias at c_base
  input  logic [63:0]  r_base,            // GEMV: the row scale (QM 3)
  input  logic [1:0]   eop,               // ELT: 0 MUL, 1 ADD, 2 MULADD, 3 SUB
  input  logic [1:0]   bmode,             // ELT B: dense, row vector, per row, scalar
  input  logic         arow,              // ELT A: one row for every m
  input  logic [2:0]   act,
  input  logic [1:0]   out,               // 0 fp32, 1 bf16, 2 int8, 3 fp16 (ELT only)
  input  logic         half_in,           // ELT A / FDOT NN B stored in packed binary16
  input  logic [31:0]  osc,
  input  logic [31:0]  m_n, n_n, k_n,     // M, N, K (ROPE: K = R)
  input  logic [31:0]  g_n,               // ROPE rows per position
  input  logic [63:0]  a_base, b_base, c_base, y_base, s_base,
  input  logic [31:0]  lda, ldb, ldy,
  output logic         busy,
  output logic [31:0]  lat,               // launch-to-data latency of this descriptor
  // reads (returns in order)
  output logic         rq_go,
  output logic [63:0]  rq_addr,
  input  logic         rq_ready,
  input  logic         rq_ret,
  input  logic [127:0] rq_data,
  output logic         rq_wide,           // RW > 1: this request is a 16 RW-byte one (rq_hi: its upper 16 (RW - 1) bytes)
  input  logic [128*((RW > 1) ? RW - 1 : 1)-1:0] rq_hi,
  // output queue
  input  logic         oq_free,            // a slot is free counting this cycle's launch (decided now, launched next cycle)
  output logic         lq_go,
  output logic [63:0]  lq_addr,
  output logic [31:0]  lq_len,
  output logic [8*RB-1:0] x_row          // the entry launched lat + 1 cycles ago
`ifdef TM_FPGA
  // the core's shared SFU (tm_sfu.sv): ELT's four lanes take it in turn (a step every 4 cycles,
  // + 3 latency), GEMV's lane 0 directly
  , output logic [31:0] sfu_x
  , input  logic [31:0] sfu_y
  // the row engine's units, idle while this unit runs (tm_core): its 16 accumulator adders (a + b,
  // 3 cycles), its fold's adder tree ((0+1), (2+3), ..., 12 cycles) and its recip (38 cycles)
  , output logic [31:0] sh_la [16], sh_lb [16]
  , input  logic [31:0] sh_ly [16]
  , output logic [31:0] sh_fl [16]
  , input  logic [31:0] sh_fsum
  , output logic [31:0] sh_rx
  , input  logic [31:0] sh_ry
`endif
);
  import tm_fp_pkg::*;
  localparam int LQ = $clog2(VQ);
  localparam logic [2:0] O_ELT = 3'd0, O_ROPE = 3'd1, O_NT = 3'd2, O_NN = 3'd3, O_Q = 3'd4, O_GV = 3'd5;
  localparam logic [2:0] K_A = 3'd0, K_B = 3'd1, K_C = 3'd2, K_XP = 3'd3, K_COS = 3'd4, K_SIN = 3'd5,
                         K_S = 3'd6, K_X = 3'd7;   // GEMV: K_XP = the row-scale beat, K_S scales, K_X x, K_B W, K_C bias

  // ------------------------------------------------------------------ per-descriptor constants
  logic [1:0]  esl;                       // log2 output element size
`ifdef TM_FPGA
  localparam int SW = 3;                  // ELT: the shared SFU's lane skew
`else
  localparam int SW = 0;
`endif
  logic [31:0] es, cl, pl, act_l, act_e, lat_c;
  logic        long_act, sp_act;
  always_comb begin
    esl      = (out == 2'd0) ? 2'd2 : (out == 2'd2) ? 2'd0 : 2'd1;
    es       = 32'd1 << esl;
    cl       = (out == 2'd0) ? 32'd0 : (out == 2'd2) ? 32'd4 : 32'd1;
    long_act = act == 3'd3 || act == 3'd4 || act == 3'd5 || act == 3'd7;
    sp_act   = act == 3'd6;
    act_l    = long_act ? 32'd122 : sp_act ? 32'(tm_fp_pkg::SP_LAT) : 32'd1;
    act_e    = act_l + ((long_act || sp_act) ? 32'(SW) : 32'd0);    // ELT
    pl       = (op == O_ELT) ? act_e + ((eop == 2'd2) ? 32'd6 : 32'd3) : (op == O_ROPE) ? 32'd6 :
               (op == O_GV) ? act_l + 32'd16 : 32'd0;
    lat_c    = pl + cl;
  end
  always_ff @(posedge clk) lat <= lat_c;  // descriptor constant: registered (the core's tap select)

  // ------------------------------------------------------------------ reader
  // walker: row m, j (NT: column n; NN: 16-column chunk; QUANT: pass), k (NN), g (group), s
  // (beat within the step)
  logic        r_act;                     // still has beats to issue
  logic [31:0] rm, rj, rk, rg;
  logic [2:0]  rs;
  logic [63:0] ra_row, rb_row, rc_row, ry_row, rb_n, rb_k, rtab, rtab2, rs_p;   // rtab2 = rtab + 2 R (sin)
  logic        r_lo, r_rot;               // ROPE: rg < R / 8 (first half), 4 rg < R (a rotary group)
  logic [31:0] rg_nx;                     // rg after this beat
  logic [31:0] ry_g, rgc;                 // output offset within the row; rows left in this table row
  logic [31:0] ngr;                       // groups per row of the walked operand
  // FDOT on the FPGA: a pass end does not hold the next pass (the fold and the readouts run behind
  // it); NN reads an A beat every 4 k
`ifdef TM_FPGA
  localparam bit FX = 1'b1;
`else
  localparam bit FX = 1'b0;
`endif
  logic        nn_ab;                     // NN: this k has an A beat
  // FDOT, 16 products a cycle (128-byte reads; shapes that fit, else the 4-lane path): four lane
  // banks taken in turn, so a lane's 3-cycle add is done when its bank comes round and each lane
  // still sums its k in order. NT: 4 keys at a time, A's row in the GEMV's x buffer (row 0, idle
  // here: 1024 bits an entry); a group's beats wait in the
  // FIFO and are read by address (key, then the next 64 bytes), so the K rows stream contiguously.
  // NN: 64 columns at a time, bank = 16 of them
  localparam bit FW = FX && RW == 8;
  logic        fw_nt, fw_nn, ra_ph, b_wd, c_wd;      // ra_ph: the walker is past A's row
  logic [31:0] e_n;                       // NT: 128-byte beats per row
  // the beat being issued and its tags
  logic [63:0] b_addr;
  logic [2:0]  b_kind;
  logic        b_last, b_launch, b_pend, b_first, b_rot, b_neg, b_q1;
  logic [1:0]  b_lg, b_ws;
  logic [63:0] b_laddr, b_roaddr;
  logic [7:0]  b_llen;
  logic [2:0]  nbeats;
  // ELT with a scalar or per-row B (bmode 2, 3): the B beat of a step is read only when it differs
  // from the last one read (bk_a); the consumer keeps that beat (cbw) for the steps without one
  logic        bk_v, b_skip;
  logic [59:0] bk_a;
  logic [LQ:0] cnt;                       // issued and not popped
  logic        rd_go;
  // the candidate register between the walker and the port: the walker computes the next beat
  // (address and tags) while the current one waits for the port, so the port's handshake does
  // not reach the walker's compares and adds
  logic        cv;
  logic [63:0] c_addr, c_laddr, c_roaddr;
  logic        c_wide;
  logic [2:0]  c_kind;
  logic        c_last, c_launch, c_pend, c_first, c_rot, c_neg, c_q1;
  logic        i_w;                       // the step has 16 products
  logic [1:0]  i_bk;                      // ... into this lane bank
  logic [31:0] ia2 [12], ib2 [12];        // operands of products 4 .. 15
  logic        hfl;                       // NN, 16 products: the head beat's second 64 bytes are next
  logic [1:0] hseg;                       // packed NN: four 16-half groups per 128-byte beat
  logic       h_done, b_half, c_half;      // narrow read's upper 8 bytes contain the four values
  logic [31:0] nn_end;
  logic        g_run, g_go, g_end, g_hf;  // NT, 16 products: a group runs; starts; its last step; which 64 bytes
  logic [1:0]  g_k;                       // the key
  logic [3:0]  g_c, aw_ra;                // the 128-byte beat of the row; the x buffer's entry read (a cycle ahead)
  logic [6:0]  aw_w;                      // A's next 16-byte beat
  logic [LQ-1:0] g_a, g_e, g_e3;          // the FIFO entry of this step; E; 3 E
  logic [511:0] hv, av;                   // the step's 64 bytes of B and of A
  localparam logic [3:0] FO0 = 4'd9;      // NT, 16 products: f_cnt when the first bank's sum is out of the fold
  logic [31:0] fo [4];                    // NT, 16 products: the group's sums
  logic        f_v;                       // a fold / readout runs behind the passes
  logic [LQ:0] pop_n;
  logic [3:0]  f_cnt;
  logic [63:0] f_a;
  logic [1:0]  c_lg, c_ws;
  logic [7:0]  c_llen;
  logic        wadv;                      // the walker hands its beat to the candidate register
  logic [31:0] h4, gp, tg;                // ROPE: half-width in groups, partner group, table group
  logic        rot;
  logic [1:0]  gv_sub;                    // GEMV: the channel's next beat (0 scale, 1 bias, 2 W)
  logic        gv_ls, gv_lb;              // GEMV: this channel reads a scale beat, a bias beat
  logic [31:0] kg;                        // K / 16
  // GEMV, wide: W in 16 RW-byte beats (K a multiple of 16 RW, W 16 RW-byte aligned, and K / 16 RW
  // >= M: the rows' launches fit a channel's beats); kgw: W beats per channel
  localparam int LRW = (RW > 1) ? $clog2(RW) : 0;
  localparam int RWH = (RW > 1) ? RW - 1 : 1;
  logic        gw;
  logic [31:0] kgw;
  logic [31:0] nxp;                       // GEMV, QM 3: row-scale beats (the scales from r_base, any word of a beat)

  always_comb begin
    kg     = k_n >> 4;
    gw     = RW > 1 && op == O_GV && k_n % (16 * RW) == 0 && b_base % (16 * RW) == 0 &&
             (m_n == 1 || (k_n >> (4 + LRW)) >= m_n);
    kgw    = gw ? k_n >> (4 + LRW) : k_n >> 4;
    nxp    = (32'(r_base[3:2]) + m_n + 32'd3) >> 2;
    gv_ls  = (qm == 2'd1) ? rk == 0 : rk[1:0] == 2'd0;
    gv_lb  = bias && rk[1:0] == 2'd0;
    gv_sub = (rs == 3'd0 && gv_ls) ? 2'd0 : (rs <= 3'd1 && gv_lb) ? 2'd1 : 2'd2;
    ngr   = ((op == O_NT) ? k_n : n_n) >> 2;
    h4    = k_n >> 3;                     // R / 8: groups in half the rotary width
    e_n   = k_n >> 5;
    fw_nt = FW && op == O_NT && k_n[4:0] == 5'd0 && k_n != 0 && k_n <= 32'd512 && (k_n >> 3) <= 32'(VQ) &&
            n_n[1:0] == 2'd0 && b_base[6:0] == 7'd0 && ldb[6:0] == 7'd0 && a_base[6:0] == 7'd0 && lda[6:0] == 7'd0;
    fw_nn = FW && op == O_NN && n_n[5:0] == 6'd0 && b_base[6:0] == 7'd0 && ldb[6:0] == 7'd0;
    nn_end = fw_nn ? (half_in ? 32'd0 : 32'd1) : 32'd3;
    h_done = half_in ? hseg == 2'd3 : hfl;
    nn_ab = !FX || rk[1:0] == 2'd0;
    rot   = (op == O_ROPE) && r_rot;
    gp    = r_lo ? rg + h4 : rg - h4;
    tg    = r_lo ? rg : rg - h4;
    case (op)
      O_ELT:   nbeats = (eop == 2'd2) ? 3'd3 : 3'd2;   // A, B, C (MULADD); A, B
      O_ROPE:  nbeats = rot ? 3'd4 : 3'd1;
      O_NT:    nbeats = fw_nt ? 3'd1 : 3'd2;
      O_NN:    nbeats = (rg == 0 && nn_ab) ? 3'd2 : 3'd1;   // rg = j (B beat within the chunk row)
      default: nbeats = 3'd1;
    endcase
    // address and kind of beat rs of the current step
    b_kind = K_A; b_addr = 64'd0; b_ws = 2'd0;
    case (op)
      O_ELT: begin
        b_ws = (bmode == 2'd2) ? rb_row[3:2] : b_base[3:2];   // the B word, on every beat of the step
        if (rs == 0) begin b_kind = K_A; b_addr = (arow ? a_base : ra_row) + (64'(rg) << (half_in ? 3 : 4)); end
        else if (rs == 1) begin
          b_kind = K_B;
          case (bmode)
            2'd0: b_addr = rb_row + {28'b0, rg, 4'b0};
            2'd1: b_addr = b_base + {28'b0, rg, 4'b0};
            2'd2: b_addr = {rb_row[63:4], 4'b0};      // rb_row = B + 4 m
            default: b_addr = {b_base[63:4], 4'b0};
          endcase
        end else begin b_kind = K_C; b_addr = rc_row + (64'(rg) << (out == 2'd3 ? 3 : 4)); end
      end
      O_ROPE: begin
        case (rs)
          3'd0: begin b_kind = K_A; b_addr = ra_row + {28'b0, rg, 4'b0}; end
          3'd1: begin b_kind = K_XP; b_addr = ra_row + {28'b0, gp, 4'b0}; end
          3'd2: begin b_kind = K_COS; b_addr = rtab + {28'b0, tg, 4'b0}; end
          default: begin b_kind = K_SIN; b_addr = rtab2 + {28'b0, tg, 4'b0}; end
        endcase
      end
      O_NT: begin
        b_kind = rs == 0 ? K_A : K_B;
        b_addr = (rs == 0 ? ra_row : rb_n) + {28'b0, rg, 4'b0};
        if (fw_nt) begin                  // A's row in 16-byte beats (once a row), then a group's K rows
          b_kind = ra_ph ? K_B : K_A;
          b_addr = ra_ph ? rb_n + {25'b0, rg, 7'b0} : ra_row + {28'b0, rg, 4'b0};
        end
      end
      O_NN: begin
        b_ws = rk[1:0];
        if (rg == 0 && rs == 0 && nn_ab) begin b_kind = K_A; b_addr = ra_row + {28'b0, rk >> 2, 4'b0}; end
        else begin
          b_kind = K_B;
          b_addr = fw_nn ? rb_k + (64'(rj) << (half_in ? 7 : 8)) + (64'(rg) << 7) :
                   rb_k + (64'(rj) << (half_in ? 5 : 6)) + (64'(rg) << (half_in ? 3 : 4));
        end
      end
      O_GV: begin
        // rj 0: the row scales (QM 3, rk < ceil(M / 4)) then x's M rows of K/16 beats (row rm at
        // ra_row); rj 1: channel rk: its scale beat
        // (per-channel: every 4th channel; per-tensor: channel 0), its bias beat (every 4th), then
        // its K/16 W beats (rs 0 scale, 1 bias, 2 W; a sub-step not needed is skipped)
        b_ws = rk[1:0];                   // the channel's word (bias; the scale's unless per-tensor)
        if (rj == 0) begin
          if (qm == 2'd3 && rk < nxp) begin b_kind = K_XP; b_addr = {r_base[63:4], 4'b0} + {28'b0, rk, 4'b0}; end
          else begin b_kind = K_X; b_addr = ra_row + {28'b0, rg, 4'b0}; end
        end else if (gv_sub == 2'd0) begin b_kind = K_S; b_addr = rs_p; end
        else if (gv_sub == 2'd1) begin b_kind = K_C; b_addr = rc_row; end
        else begin b_kind = K_B; b_addr = rb_k; end
      end
      default: begin b_kind = K_A; b_addr = ra_row + {28'b0, rg, 4'b0}; end
    endcase
    b_skip  = op == O_ELT && bmode[1] && bk_v && bk_a == ((bmode == 2'd2) ? rb_row[63:4] : b_base[63:4]);
    b_last  = (op == O_GV) ? (rj == 1 && gv_sub == 2'd2) :
              (32'(rs) + 1 >= 32'(nbeats) || (rs == 3'd0 && b_skip && nbeats == 3'd2));
    // the group index after this beat (the walker's step below), for the registered ROPE flags
    if (!b_last) rg_nx = rg;
    else case (op)
      O_NT:    rg_nx = (rg + 1 < (k_n >> 2)) ? rg + 1 : 32'd0;
      O_NN:    rg_nx = (rg < 3) ? rg + 1 : 32'd0;
      default: rg_nx = (rg + 1 < (n_n >> 2)) ? rg + 1 : 32'd0;
    endcase
    b_lg    = rg[1:0];
    b_rot   = rot;
    b_neg   = r_lo;
    b_q1    = (op == O_Q) && rj == 1;
    b_first = rg == 0;                    // QUANT: the row's first group (restarts the max); GEMV: a channel's first W beat
    b_launch = (op == O_ELT) || (op == O_ROPE) || b_q1 || (op == O_GV && rg + 1 >= kgw);
    b_laddr = ry_row + 64'(ry_g);
    b_llen  = (op == O_GV) ? 8'(es) : 8'(es << 2);
    b_pend  = (op == O_NT) ? !fw_nt && rg + 1 >= ngr : (op == O_NN) ? (rk + 1 >= k_n && rg == nn_end) :
              (op == O_Q) && rj == 0 && rg + 1 >= ngr;
    b_wd    = (fw_nt || fw_nn) && b_kind == K_B;
    b_half = b_addr[3];
    if ((op == O_ELT && ((half_in && b_kind == K_A) || (out == 2'd3 && b_kind == K_C))) ||
        (op == O_NN && half_in && b_kind == K_B && !fw_nn)) b_addr[3:0] = 4'd0;
    b_roaddr = (op == O_Q) ? rs_p : ry_row + 64'(ry_g);
    rd_go = cv && cnt < (LQ + 1)'(VQ);
    rq_go = rd_go;
    rq_addr = c_addr;
    rq_wide = c_wide;
  end

  // tag FIFO beside the data FIFO; the head's entry (qd_h, qt_h) is what the consumer reads
  typedef struct packed {
    logic [2:0]  k;
    logic        l, la, pe, fi, ro, ng, q1, hs;
    logic [1:0]  lg, ws;
    logic [63:0] laddr, roaddr;
    logic [7:0]  llen;
  } qtag_t;
  qtag_t        qt_w, qt_h;
  logic [127:0] qd_h;
  logic [128*RWH-1:0] qh_h;               // a wide W beat's upper 16 (RW - 1) bytes
  assign qt_w = '{k: c_kind, l: c_last, la: c_launch, pe: c_pend, fi: c_first, ro: c_rot, ng: c_neg,
                 q1: c_q1, hs: c_half, lg: c_lg, ws: c_ws, laddr: c_laddr, roaddr: c_roaddr, llen: c_llen};
  logic [LQ-1:0] q_t, q_r, q_h;
  logic [LQ:0]  q_rcv;                    // returned and not popped
`ifdef TM_FPGA
  // FPGA: distributed RAM, one write port each (tags as the reader issues, data as it returns)
  logic [31:0]  q_ra [1];
  logic [$bits(qtag_t)-1:0] qt_rd [1];
  logic [127:0] qd_rd [1];
  logic [128*RWH-1:0] qh_rd [1];
  assign q_ra[0] = g_run ? 32'(g_a) : 32'(q_h);
  tm_lutram #(.W($bits(qtag_t)), .D(VQ), .NR(1)) u_qt (.clk, .we(rds && !rst), .wa(32'(q_t)), .wd(qt_w),
                                                      .ra(q_ra), .rd(qt_rd));
  tm_lutram #(.W(128), .D(VQ), .NR(1)) u_qd (.clk, .we(rq_ret && !rst), .wa(32'(q_r)), .wd(rq_data),
                                            .ra(q_ra), .rd(qd_rd));
  if (RW > 1) begin : g_qh
    tm_lutram #(.W(128 * RWH), .D(VQ), .NR(1)) u_qh (.clk, .we(rq_ret && !rst), .wa(32'(q_r)), .wd(rq_hi),
                                              .ra(q_ra), .rd(qh_rd));
  end else begin : g_nqh
    assign qh_rd[0] = '0;
  end
  assign qt_h = qt_rd[0];
  assign qd_h = qd_rd[0];
  assign qh_h = qh_rd[0];
`else
  logic [127:0] qd [VQ];
  logic [128*RWH-1:0] qh [VQ];
  qtag_t        qt [VQ];
  assign qt_h = qt[q_h];
  assign qd_h = qd[q_h];
  assign qh_h = RW > 1 ? qh[q_h] : '0;
`endif

  // ------------------------------------------------------------------ consumer state
  logic [31:0]  ca [4], cb [4], cc [4], cxp [4], ccos [4], csin [4];
  logic [127:0] cbw;                      // B beat (for the word select)
  logic         c_wait;                   // a pass end is in progress
  logic [7:0]   w_cnt;
  logic [2:0]   w_ph;
  logic         pop, fire, h_last, h_launch;
  logic         s_ok;                     // ELT with a long act: the shared SFU's lanes are free
  logic [1:0]   s_cnt;
  logic [31:0]  inv_q;                    // QUANT: this row's 127 / amax
  // datapath issue registers (cycle 0)
  logic [31:0]  ia [4], ib [4], ic [4], ixp [4], ics [4], isn [4];
  logic         i_rot, i_neg, i_acc, i_launch, i_q0, i_qfi;
  logic [1:0]   i_lg;
  logic [63:0]  i_laddr;
  logic [31:0]  i_llen;
  logic [31:0]  nn_a;                     // NN: A[m][k] for this k's four steps
  logic [31:0]  a_now [4];                // the step's A beat (the head, or collected)
  // readouts
  logic         ro_go;                    // a readout launches this cycle
  logic [31:0]  ro_v [4];
  logic         ro_raw;
  logic [1:0]   ro_q;                     // NN: which group of lanes
  logic [63:0]  ro_addr, ro_la;           // the next readout's address; the launched one's

  logic [127:0] hd;
  logic [31:0] hd_value [4], wide_half [16];
  logic [1023:0] hwide;
  logic [255:0] hpart;
  logic [2:0]   hk;
  always_comb begin
    hd = qd_h;
    hk = qt_h.k;
    h_last = qt_h.l;
    h_launch = qt_h.la;
    // a step fires on its last beat; the beat waits if the step cannot fire yet
    s_ok = SW == 0 || op != O_ELT || !(long_act || sp_act) || s_cnt == 0;
    fire = q_rcv != 0 && h_last && !c_wait && (!h_launch || (oq_free && gl_rem == '0)) && !ro_go && s_ok && !fw_nt;
    pop  = q_rcv != 0 && !g_run && (!h_last || (fire && (!(fw_nn && hk == K_B) || h_done)));
    g_e  = LQ'(e_n);
    g_e3 = LQ'(e_n + (e_n << 1));
    g_go  = fw_nt && !g_run && !c_wait && q_rcv != 0 && hk == K_B && (LQ + 1)'(q_rcv) >= (LQ + 1)'(e_n << 2);
    g_end = g_run && g_k == 2'd3 && g_hf && 32'(g_c) + 1 >= e_n;
    pop_n = g_end ? (LQ + 1)'(e_n << 2) : (LQ + 1)'(pop);
    hv = (hfl || (g_run && g_hf)) ? qh_h[895:384] : {qh_h[383:0], hd};
    av = g_hf ? gv_x[0][1023:512] : gv_x[0][511:0];
    aw_ra = (g_run && g_k == 2'd3 && g_hf) ? g_c + 4'd1 : g_c;
    hwide = 1024'({qh_h, hd});
    hpart = hwide[256 * hseg +: 256];
    for (int i = 0; i < 16; i++) wide_half[i] = from_fp16(hpart[16 * i +: 16]);
    for (int i = 0; i < 4; i++) begin
      hd_value[i] = ((op == O_ELT && ((half_in && hk == K_A) || (out == 2'd3 && hk == K_C))) ||
                     (op == O_NN && half_in && hk == K_B)) ?
                    from_fp16(hd[64 * int'(qt_h.hs) + 16 * i +: 16]) : hd[32 * i +: 32];
      a_now[i] = (hk == K_A) ? hd_value[i] : ca[i];
    end
  end

  // ------------------------------------------------------------------ GEMV
  // x buffer (written as x's beats pop, read at each W beat's fire), the step's operands, the
  // channel's scale / bias words, the exact dot product and accumulator, the epilogue lane
  logic [127:0] csb, ccb;                 // the collected scale and bias beats
  localparam int LMB = (MB > 1) ? $clog2(MB) : 1;
  logic [7:0]   xw, gidx;                 // x beats written (of row xm); W beat within the channel
  logic [LMB-1:0] xm;
  logic [1:0]   xpi;                      // row-scale beats taken
  // x's rows in NXB banks of 16-byte beats (beat i in bank i mod NXB): a wide W beat meets NXB
  // consecutive x beats, a narrow one sits in its lane of the NXB (the others are 0)
  localparam int NXB = (RW > 2) ? RW : 2;
  localparam int LNX = $clog2(NXB);
  localparam int G3W = 18 + $clog2(4 * NXB);
  localparam int LXD = $clog2(256 / NXB);
  logic [128*NXB-1:0] gv_w, gv_x [MB];    // issue registers (cycle 0): the W beat (wide: 16 RW bytes), x's (buffer reads)
  logic [127:0] gx_b [MB][NXB];           // x's beats, one a bank
  logic [7:0]   gx_i;                     // the x beat read (wide: the group of NXB)

  logic         i_gv, i_gf, i_ge;         // a GEMV step issues; it is the channel's first, last
  logic [31:0]  gv_rs, gv_sc, gv_bi;      // row scale of the row in the epilogue, the channel's scale and bias
  logic [31:0]  rsw [MB];                 // the rows' scales (1.0 unless QM 3)
  (* use_dsp = "yes" *) logic signed [15:0] gpr [MB][16*NXB];
  logic signed [17:0] g2 [MB][4*NXB];
  logic signed [G3W-1:0] g3 [MB];
  logic signed [31:0] gacc [MB], gsum [MB], ga_x;
  // the launch sequencer (rows 1 .. M-1 of a channel) and the row tag of each launch, delayed to
  // the epilogue's stages: cycle 4 after its launch the row's sum enters i2f, cycle 6 its row
  // scale meets it
  logic [LMB:0]   gl_rem;
  logic [LMB-1:0] gl_j, gl_ja;
  logic [63:0]    gl_a, gl_la;
  logic           gl_go;
  logic           tg_v [1:6];
  logic [LMB-1:0] tg_j [1:6];
  logic         g1v, g1f, g1e, g2v, g2f, g2e, g3v, g3f, g3e, g4e;
  logic         ga_s, ga_z;
  logic [31:0]  ga_u, gf0, gyr, gsc9, gy0, gbi12, gy1a, gy0d, gv_y;
  assign gx_i = (fire && op == O_GV) ? (qt_h.fi ? 8'd0 : gidx) : 8'd0;
  for (genvar b = 0; b < MB; b++) begin : g_xb
    // x row b, bank j: beats j, j + NXB, ... (wide: the group at gx_i; narrow: beat gx_i's group)
    for (genvar j = 0; j < NXB; j++) begin : g_bk
      tm_bram #(.W(128), .D(256 / NXB)) u_x (.clk,
        .we((pop && op == O_GV && hk == K_X && 32'(xm) == b && 32'(xw[LNX-1:0]) == j) ||
            (b == 0 && FW && pop && fw_nt && hk == K_A && 32'(aw_w[LNX-1:0]) == j)),
        .wa((b == 0 && fw_nt) ? LXD'(aw_w >> LNX) : LXD'(xw >> LNX)), .wd(hd),
        .ra((b == 0 && fw_nt) ? LXD'(aw_ra) : gw ? LXD'(gx_i) : LXD'(gx_i >> LNX)), .rd(gx_b[b][j]));
      assign gv_x[b][128 * j +: 128] = gx_b[b][j];
    end
  end
  always_ff @(posedge clk) begin
    // 1: sixteen INT8 products a row; 2: four sums of four; 3: the beat's sum; 4: the accumulator
    for (int b = 0; b < MB; b++) begin
      for (int i = 0; i < 16 * NXB; i++) gpr[b][i] <= $signed(gv_w[8 * i +: 8]) * $signed(gv_x[b][8 * i +: 8]);
      for (int j = 0; j < 4 * NXB; j++)
        g2[b][j] <= 18'(gpr[b][4 * j]) + 18'(gpr[b][4 * j + 1]) + 18'(gpr[b][4 * j + 2]) + 18'(gpr[b][4 * j + 3]);
      begin
        logic signed [G3W-1:0] t;
        t = '0;
        for (int j = 0; j < 4 * NXB; j++) t = t + G3W'(g2[b][j]);
        g3[b] <= t;
      end
      if (g3v) gacc[b] <= g3f ? 32'(g3[b]) : gacc[b] + 32'(g3[b]);
      if (g4e) gsum[b] <= gacc[b];        // the channel's sums, kept for the launches of rows 1 ..
    end
    g1v <= i_gv; g1f <= i_gf; g1e <= i_ge;
    g2v <= g1v; g2f <= g1f; g2e <= g1e;
    g3v <= g2v; g3f <= g2f; g3e <= g2e;
    g4e <= g3v && g3e;
    // the epilogue from the launched row's final sum (cycle 4 after its launch): i2f in two stages
    ga_s <= ga_x[31]; ga_z <= ga_x == 0; ga_u <= ga_x[31] ? 32'(-ga_x) : 32'(ga_x);
    gf0  <= ga_z ? 32'h0 : i2f_tail(ga_s, ga_u);
    // the row tags
    tg_v[1] <= (i_launch && op == O_GV) || gl_go;
    tg_j[1] <= gl_go ? gl_ja : '0;
    for (int k = 2; k <= 6; k++) begin tg_v[k] <= tg_v[k-1]; tg_j[k] <= tg_j[k-1]; end
  end
  // row 0's sum is final in the cycle its launch reaches stage 4 (g4e); later rows' come from gsum
  assign ga_x  = (tg_v[4] && tg_j[4] != '0) ? gsum[tg_j[4]] : gacc[0];
  assign gv_rs = rsw[tg_v[6] ? tg_j[6] : '0];
  logic [31:0] gsc0, gbi0;
  assign gsc0 = gv_sc;
  assign gbi0 = gv_bi;
  fpu_mul u_gr (.clk, .a(gf0), .b(gv_rs), .y(gyr));                          // 9: row scale
  fpu_dly #(32, 9) u_gs9 (.clk, .d(gsc0), .q(gsc9));
  fpu_mul u_gm (.clk, .a(gyr), .b(gsc9), .y(gy0));                            // 12: column scale
  fpu_dly #(32, 12) u_gb12 (.clk, .d(gbi0), .q(gbi12));
  fpu_add u_ga (.clk, .a(gy0), .b(gbi12), .y(gy1a));                          // 15: bias
  fpu_dly #(32, 3) u_gy0d (.clk, .d(gy0), .q(gy0d));
  // 16: the result, registered, and the lane-0 input select as one registered bit (the descriptor
  // decode stays off the activation's first stage)
  logic gsel;
  always_ff @(posedge clk) begin
    gv_y <= bias ? gy1a : gy0d;
    gsel <= op == O_GV;
  end

  // ------------------------------------------------------------------ datapath
  // ELT: activation, B / C delay lines, op
  logic [31:0] e_y1 [4], e_yl [4], e_ys [4], e_ya [4], e_bd [4], e_cd [4], e_m1 [4], e_ab [4], e_mac [4];
  logic [31:0] e_out [4];
  // ROPE
  logic [31:0] r_p1 [4], r_p2 [4], r_sum [4], r_x6 [4], r_out [4];
  logic        r_rot6, r_neg3;
  // FDOT products
  logic [31:0] f_pr [4];
`ifdef TM_FPGA
  logic        i_fire;                    // the issue registers hold a step
  logic [31:0] s_y [4], s_px;
  always_ff @(posedge clk) begin
    i_fire <= fire;
    if (rst) s_cnt <= 2'd0;
    else if (fire && op == O_ELT && (long_act || sp_act)) s_cnt <= 2'(SW);
    else if (s_cnt != 0) s_cnt <= s_cnt - 2'd1;
  end
  tm_sfu_port #(4) u_sp (.clk, .v(i_fire), .x(ia), .sx(s_px), .sy(sfu_y), .y(s_y));
  assign sfu_x = gsel ? gv_y : s_px;
`else
  assign s_cnt = 2'd0;
`endif
  for (genvar i = 0; i < 4; i++) begin : g_s
`ifdef TM_FPGA
    fpu_act1 u_act (.clk, .x((i == 0 && gsel) ? gv_y : ia[i]), .act, .y1(e_y1[i]));
    assign e_yl[i] = (i == 0 && gsel) ? sfu_y : s_y[i];
    assign e_ys[i] = e_yl[i];
`else
    fpu_act u_act (.clk, .x((i == 0 && gsel) ? gv_y : ia[i]), .act, .y1(e_y1[i]), .ylong(e_yl[i]), .ysp(e_ys[i]));
`endif
    assign e_ya[i] = long_act ? e_yl[i] : sp_act ? e_ys[i] : e_y1[i];
    tm_vdly u_bd (.clk, .rst, .d(ib[i]), .n(act_e), .q(e_bd[i]));
    tm_vdly u_cd (.clk, .rst, .d(ic[i]), .n(act_e + 32'd3), .q(e_cd[i]));
    fpu_mul u_m1 (.clk, .a(e_ya[i]), .b(e_bd[i]), .y(e_m1[i]));
    fpu_add u_ab (.clk, .a(e_ya[i]), .b(eop == 2'd3 ? fneg(e_bd[i]) : e_bd[i]), .y(e_ab[i]));
    fpu_add u_mac (.clk, .a(e_cd[i]), .b(e_m1[i]), .y(e_mac[i]));
    assign e_out[i] = (eop == 2'd2) ? e_mac[i] : (eop == 2'd0) ? e_m1[i] : e_ab[i];
    fpu_mul u_p1 (.clk, .a(ia[i]), .b(ics[i]), .y(r_p1[i]));
    fpu_mul u_p2 (.clk, .a(ixp[i]), .b(isn[i]), .y(r_p2[i]));
    fpu_add u_rs (.clk, .a(r_p1[i]), .b(r_neg3 ? fneg(r_p2[i]) : r_p2[i]), .y(r_sum[i]));
    fpu_dly #(32, 6) u_x6 (.clk, .d(ftz(ia[i])), .q(r_x6[i]));
    assign r_out[i] = r_rot6 ? r_sum[i] : r_x6[i];
    fpu_mul u_fp (.clk, .a(ia[i]), .b(ib[i]), .y(f_pr[i]));
  end
  logic r_rot3;
  fpu_dly #(1, 3) u_rot3 (.clk, .d(i_rot), .q(r_rot3));
  fpu_dly #(1, 3) u_rot6 (.clk, .d(r_rot3), .q(r_rot6));
  fpu_dly #(1, 3) u_neg3 (.clk, .d(i_neg), .q(r_neg3));

  // FDOT accumulators: a product reaches lane (lg * 4 + i) 3 cycles after its step; the same lane
  // is touched at most every 4 cycles (NT: every 4th step, 2 beats each; NN: 5 beats per k), so
  // the 3-cycle add completes before the next contribution
  logic [31:0] lane_s [16], lane_y [16], lane_c [16], f_p2 [12];
  logic [1:0]  bk3, bk6, l_rb;            // the banks of the add's read and write; the bank read
  logic [1:0]  l_fb;                      // at a pass end: the bank going to the fold
  logic [15:0] lv [4];
  logic        w3;
  logic [15:0] lane_in;
  logic [2:0]  lane_p [16];
  logic        acc3;
  logic [1:0]  lg3;
  fpu_dly #(1, 3) u_acc3 (.clk, .d(i_acc), .q(acc3));
  fpu_dly #(2, 3) u_lg3 (.clk, .d(i_lg), .q(lg3));
  fpu_dly #(1, 3) u_w3 (.clk, .d(i_w), .q(w3));
  fpu_dly #(2, 3) u_bk3 (.clk, .d(i_bk), .q(bk3));
  fpu_dly #(2, 3) u_bk6 (.clk, .d(bk3), .q(bk6));
  if (FW) begin : g_fp
    for (genvar i = 0; i < 12; i++) begin : g_m
      fpu_mul u (.clk, .a(ia2[i]), .b(ib2[i]), .y(f_p2[i]));
    end
  end else begin : g_nfp
    assign f_p2 = '{default: 32'h0};
  end
  // the lanes: four banks in a small memory each (the bank is the address); a bank entry counts as
  // zero until its first sum (lv), so a pass end clears 16 bits, not 16 words
  for (genvar l = 0; l < 16; l++) begin : g_ln
    logic [31:0] ra [1], rd [1];
    assign ra[0] = 32'(l_rb);
    tm_lutram #(.W(32), .D(4), .NR(1)) u_l (.clk, .we(lane_p[l][2]), .wa(32'(bk6)), .wd(lane_y[l]), .ra(ra), .rd(rd));
    assign lane_s[l] = lv[l_rb][l] ? rd[0] : 32'h0;
  end
  assign l_rb = acc3 ? bk3 : l_fb;
  always_comb begin
    lane_in = '0;                         // (a lane not written ignores its adder)
    if (acc3) lane_in = w3 ? 16'hFFFF : 16'(16'h000F << (int'(lg3) * 4));
    for (int l = 0; l < 16; l++) lane_c[l] = (w3 && l >= 4) ? f_p2[l - 4] : f_pr[l % 4];
  end
`ifdef TM_FPGA
  assign sh_la = lane_s;
  assign sh_lb = lane_c;
  assign lane_y = sh_ly;
`else
  for (genvar l = 0; l < 16; l++) begin : g_acc
    fpu_add u (.clk, .a(lane_s[l]), .b(lane_c[l]), .y(lane_y[l]));
  end
`endif
  // NT fold: (0+1), (2+3), ... four levels, 12 cycles
  logic [31:0] fl [16], t8 [8], t4 [4], t2 [2], fsum;
`ifdef TM_FPGA
  assign sh_fl = fl;
  assign fsum = sh_fsum;
`else
  for (genvar i = 0; i < 8; i++) begin : g_f8
    fpu_add u (.clk, .a(fl[2 * i]), .b(fl[2 * i + 1]), .y(t8[i]));
  end
  for (genvar i = 0; i < 4; i++) begin : g_f4
    fpu_add u (.clk, .a(t8[2 * i]), .b(t8[2 * i + 1]), .y(t4[i]));
  end
  for (genvar i = 0; i < 2; i++) begin : g_f2
    fpu_add u (.clk, .a(t4[2 * i]), .b(t4[2 * i + 1]), .y(t2[i]));
  end
  fpu_add u_fs (.clk, .a(t2[0]), .b(t2[1]), .y(fsum));
`endif
  // QUANT: running max of |x| per slot, then the row's scalars
  logic [31:0] qmax [4], qm01, qm23, amax, q_rc, q_inv, q_s;
`ifdef TM_FPGA
  assign sh_rx = amax;
  assign q_rc = sh_ry;                                               // 38
`else
  fpu_recip u_qr (.clk, .x(amax), .y(q_rc));                         // 38
`endif
  fpu_mul u_qi (.clk, .a(Q127), .b(q_rc), .y(q_inv));                // 41
  fpu_mul u_qs (.clk, .a(amax), .b(INV127), .y(q_s));               // 3

  // ------------------------------------------------------------------ conversion stage
  // input: 4 values entering now (element data pl cycles after its launch, or a readout), raw
  // (fp32 bits whatever the format); fp32 +0, bf16 +1, int8 +4 (q8 with osc, QUANT: inv)
  logic [31:0] cv_in [4], cv_osc, cv_yq [4], cv_raw4 [4], cv_in1 [4], cv_in4 [4];
  logic        cv_raw, raw1, raw4;
  logic [15:0] cv_bf [4], cv_half [4];
  logic [7:0]  cv_q8 [4];
  logic [31:0] lsel [4];
  always_comb begin
    for (int i = 0; i < 4; i++) begin
      case (op)
        O_ELT:   lsel[i] = e_out[i];
        O_ROPE:  lsel[i] = r_out[i];
        O_GV:    lsel[i] = e_ya[i];      // lane 0: the channel's output
        default: lsel[i] = ia[i];        // QUANT pass 1: x at the issue registers (pl = 0)
      endcase
      cv_in[i] = ro_go ? ro_v[i] : lsel[i];
    end
    cv_raw = ro_go && ro_raw;
    cv_osc = (op == O_Q) ? inv_q : osc;
  end
  for (genvar i = 0; i < 4; i++) begin : g_cv
    always_ff @(posedge clk) cv_bf[i] <= to_bf16(cv_in[i]);
    always_ff @(posedge clk) cv_half[i] <= to_fp16(cv_in[i]);
    fpu_mul u_q (.clk, .a(cv_in[i]), .b(cv_osc), .y(cv_yq[i]));
    always_ff @(posedge clk) cv_q8[i] <= q8_tail(cv_yq[i]);
    fpu_dly #(32, 1) u_d1 (.clk, .d(cv_in[i]), .q(cv_in1[i]));
    fpu_dly #(32, 4) u_d4 (.clk, .d(cv_in[i]), .q(cv_in4[i]));
  end
  fpu_dly #(1, 1) u_r1 (.clk, .d(cv_raw), .q(raw1));
  fpu_dly #(1, 4) u_r4 (.clk, .d(cv_raw), .q(raw4));
  logic [127:0] cv_row, cv_f;
  logic [63:0]  cv_b;
  logic [31:0]  cv_q;
  always_comb begin
    for (int i = 0; i < 4; i++) begin
      cv_f[32 * i +: 32] = cv_in[i];
      cv_b[16 * i +: 16] = out == 2'd3 ? cv_half[i] : cv_bf[i];
      cv_q[8 * i +: 8]   = cv_q8[i];
    end
    cv_row = (out == 2'd0) ? cv_f
           : (out != 2'd2) ? (raw1 ? {96'b0, cv_in1[0]} : {64'b0, cv_b})
           : (raw4 ? {96'b0, cv_in4[0]} : {96'b0, cv_q});
  end
  always_ff @(posedge clk) x_row <= (8 * RB)'(cv_row);

  // ------------------------------------------------------------------ control
  logic        rds;                        // a beat is issued
  assign rds  = rd_go && rq_ready;
  assign wadv = r_act && (!cv || rds);
  assign lq_go   = i_launch || ro_go || gl_go;
  assign lq_addr = ro_go ? ro_la : gl_go ? gl_la : i_laddr;
  assign lq_len  = ro_go ? (ro_raw ? 32'd4 : (op == O_NT) ? es : es << 2) : gl_go ? es : i_llen;
  assign busy    = r_act || cv || q_rcv != 0 || cnt != 0 || c_wait || f_v || g_run || i_launch || ro_go || gl_go || gl_rem != '0;

  always_ff @(posedge clk) begin
    if (rst) begin
      l_fb <= 2'd0; r_act <= 1'b0; cv <= 1'b0; cnt <= 0; q_t <= 0; q_r <= 0; q_h <= 0; q_rcv <= 0; c_wait <= 1'b0;
      ro_go <= 1'b0; i_acc <= 1'b0; i_launch <= 1'b0; f_v <= 1'b0;
      lv <= '{default: 16'h0}; lane_p <= '{default: 3'b0}; g_run <= 1'b0; hfl <= 1'b0; hseg <= 2'd0;
      gl_go <= 1'b0; gl_rem <= '0;
    end else begin
      // -------------------------------------------------------- reader
      if (rds) begin                       // the candidate goes out: its tags into the FIFO
`ifndef TM_FPGA
        qt[q_t] <= qt_w;
`endif
        q_t <= q_t + 1'b1;
      end
      if (start) cv <= 1'b0;
      else if (wadv) begin
        cv <= 1'b1; c_addr <= b_addr; c_kind <= b_kind; c_last <= b_last && !(fw_nt && b_kind == K_A); c_launch <= b_launch && b_last;
        c_wide <= (gw && rj == 1 && b_kind == K_B) || b_wd;
        c_pend <= b_pend && b_last; c_first <= b_first; c_rot <= b_rot; c_neg <= b_neg; c_q1 <= b_q1;
        c_lg <= b_lg; c_ws <= b_ws; c_laddr <= b_laddr; c_llen <= b_llen; c_roaddr <= b_roaddr; c_half <= b_half;
      end else if (rds) cv <= 1'b0;
      if (start) bk_v <= 1'b0;
      else if (wadv && op == O_ELT && b_kind == K_B) begin bk_v <= 1'b1; bk_a <= b_addr[63:4]; end
      if (start) begin
        r_act <= m_n != 0; ra_ph <= 1'b0; hfl <= 1'b0; hseg <= 2'd0; aw_w <= 7'd0;
        rm <= 0; rj <= 0; rk <= 0; rg <= 0; rs <= 0;
        ra_row <= a_base; rb_row <= b_base; rc_row <= c_base;
        ry_row <= y_base; ry_g <= 0; rb_n <= b_base; rb_k <= b_base; rtab <= b_base; rgc <= g_n;
        rtab2 <= b_base + {31'b0, k_n, 1'b0}; r_lo <= (k_n >> 3) != 0; r_rot <= k_n != 0;
        rs_p <= s_base;
        lv <= '{default: 16'h0};
        xw <= 0; xm <= '0; xpi <= '0;      // GEMV: x from beat 0 of row 0; row scales 1.0 unless QM 3
        for (int b = 0; b < MB; b++) rsw[b] <= 32'h3F800000;
      end else if (wadv) begin
        r_lo <= rg_nx < h4; r_rot <= (rg_nx << 2) < k_n;
        if (op == O_GV && !b_last) begin    // GEMV: the row scale, x, a scale or a bias beat
          if (rj == 0) begin
            if (qm == 2'd3 && rk < nxp) rk <= rk + 1;
            else if (rg + 1 < kg) rg <= rg + 1;
            else if (rm + 1 < m_n) begin rg <= 0; rm <= rm + 1; ra_row <= ra_row + 64'(lda); end
            else begin rg <= 0; rm <= 0; rj <= 1; rk <= 0; rs <= 0; end
          end else if (gv_sub == 2'd0) begin rs <= 3'd1; rs_p <= rs_p + 64'd16; end
          else begin rs <= 3'd2; rc_row <= rc_row + 64'd16; end
        end else if (!b_last) rs <= (rs == 3'd0 && b_skip) ? 3'd2 : rs + 3'd1;
        else begin
          rs <= (op == O_GV && rg + 1 < kgw) ? 3'd2 : 3'd0;   // GEMV: the channel's next W beat
          // advance the walker by one step
          case (op)
            O_ELT, O_ROPE: begin
              ry_g <= ry_g + (es << 2);
              if (rg + 1 < (n_n >> 2)) rg <= rg + 1;
              else begin
                rg <= 0; ry_g <= 0; rm <= rm + 1; r_act <= rm + 1 < m_n;
                ra_row <= ra_row + 64'(lda); ry_row <= ry_row + 64'(ldy); rc_row <= rc_row + 64'(ldy);
                rb_row <= rb_row + ((bmode == 2'd2) ? 64'd4 : 64'(ldb));
                if (rgc <= 1) begin rgc <= g_n; rtab <= rtab + {30'b0, k_n, 2'b0}; rtab2 <= rtab2 + {30'b0, k_n, 2'b0}; end
                else rgc <= rgc - 1;
              end
            end
            O_NT: if (fw_nt) begin
              if (rg + 1 < (ra_ph ? e_n : k_n >> 2)) rg <= rg + 1;
              else begin
                rg <= 0;
                if (!ra_ph) ra_ph <= 1'b1;
                else begin
                  rb_n <= rb_n + 64'(ldb);
                  if (rk != 32'd3) rk <= rk + 1;
                  else begin
                    rk <= 0; ry_g <= ry_g + (es << 2);
                    if (rj + 4 < n_n) rj <= rj + 4;
                    else begin
                      rj <= 0; rb_n <= b_base; ry_g <= 0; rm <= rm + 1; r_act <= rm + 1 < m_n; ra_ph <= 1'b0;
                      ra_row <= ra_row + 64'(lda); ry_row <= ry_row + 64'(ldy);
                    end
                  end
                end
              end
            end else begin
              if (rg + 1 < (k_n >> 2)) rg <= rg + 1;
              else begin
                rg <= 0; ry_g <= ry_g + es;
                if (rj + 1 < n_n) begin rj <= rj + 1; rb_n <= rb_n + 64'(ldb); end
                else begin
                  rj <= 0; rb_n <= b_base; ry_g <= 0; rm <= rm + 1; r_act <= rm + 1 < m_n;
                  ra_row <= ra_row + 64'(lda); ry_row <= ry_row + 64'(ldy);
                end
              end
            end
            O_NN: begin
              if (rg < nn_end) rg <= rg + 1;
              else begin
                rg <= 0;
                if (rk + 1 < k_n) begin rk <= rk + 1; rb_k <= rb_k + 64'(ldb); end
                else begin
                  rk <= 0; rb_k <= b_base; ry_g <= ry_g + (fw_nn ? es << 6 : es << 4);
                  if (rj + 1 < (fw_nn ? n_n >> 6 : n_n >> 4)) rj <= rj + 1;
                  else begin
                    rj <= 0; ry_g <= 0; rm <= rm + 1; r_act <= rm + 1 < m_n;
                    ra_row <= ra_row + 64'(lda); ry_row <= ry_row + 64'(ldy);
                  end
                end
              end
            end
            O_GV: begin
              if (rg + 1 < kgw) rg <= rg + 1;
              else begin                     // the channel's last W beat
                rg <= 0; ry_g <= ry_g + es; rk <= rk + 1; r_act <= rk + 1 < n_n;
              end
              rb_k <= rb_k + (gw ? 64'(16 * RW) : 64'd16);
            end
            default: begin                  // QUANT: pass 0 then pass 1 over the row
              if (rj == 1) ry_g <= ry_g + 32'd4;
              if (rg + 1 < (n_n >> 2)) rg <= rg + 1;
              else begin
                rg <= 0;
                if (rj == 0) rj <= 1;
                else begin
                  rj <= 0; ry_g <= 0; rm <= rm + 1; r_act <= rm + 1 < m_n;
                  ra_row <= ra_row + 64'(lda); ry_row <= ry_row + 64'(ldy); rs_p <= rs_p + 64'd4;
                end
              end
            end
          endcase
        end
      end
      if (rq_ret) begin
`ifndef TM_FPGA
        qd[q_r] <= rq_data;
        qh[q_r] <= rq_hi;
`endif
        q_r <= q_r + 1'b1;
      end
      cnt   <= cnt + (LQ + 1)'(rds) - pop_n;
      q_rcv <= q_rcv + (LQ + 1)'(rq_ret) - pop_n;
      if (pop || g_end) q_h <= q_h + LQ'(pop_n);
      if (pop && fw_nt && hk == K_A) aw_w <= aw_w + 7'd1;
      if (g_end) aw_w <= 7'd0;
      if (fire && fw_nn && hk == K_B) begin hfl <= !hfl; hseg <= hseg + 2'd1; end
      // NT, 16 products: a group's steps, one a cycle (key; second 64 bytes; next 128-byte beat)
      if (g_go) begin
        g_run <= 1'b1; g_a <= q_h; g_k <= 2'd0; g_hf <= 1'b0; g_c <= 4'd0; ro_addr <= qt_h.roaddr;
      end else if (g_end) begin
        g_run <= 1'b0; c_wait <= 1'b1; w_ph <= 3'd4; w_cnt <= 8'd0; g_c <= 4'd0;
      end else if (g_run) begin
        g_k <= g_k + 2'd1;
        if (g_k != 2'd3) g_a <= g_a + g_e;
        else begin
          g_a <= g_a - g_e3 + LQ'(g_hf);
          g_hf <= !g_hf;
          if (g_hf) g_c <= g_c + 4'd1;
        end
      end

      // -------------------------------------------------------- consumer: collect the beats
      if (pop) begin
        for (int i = 0; i < 4; i++) begin
          case (hk)
            K_A:   ca[i] <= hd_value[i];
            K_B:   cb[i] <= hd_value[i];
            K_C:   cc[i] <= hd_value[i];
            K_XP:  cxp[i] <= hd[32 * i +: 32];
            K_COS: ccos[i] <= hd[32 * i +: 32];
            default: csin[i] <= hd[32 * i +: 32];
          endcase
        end
        if (hk == K_B) cbw <= hd;
        if (op == O_GV) begin
          if (hk == K_X) begin
            if (9'(xw) + 9'd1 == 9'(kg)) begin xw <= 8'd0; xm <= xm + LMB'(1); end
            else xw <= xw + 8'd1;
          end
          if (hk == K_S) csb <= hd;
          if (hk == K_C) ccb <= hd;
          if (hk == K_XP) begin
            // word i of the beat is row 4 xpi + i - (the first row's word in its beat)
            for (int i = 0; i < 4; i++)
              if (4 * int'(xpi) + i >= int'(r_base[3:2]) && 4 * int'(xpi) + i - int'(r_base[3:2]) < MB)
                rsw[4 * int'(xpi) + i - int'(r_base[3:2])] <= hd[32 * i +: 32];
            xpi <= xpi + 2'd1;
          end
        end
      end
      // the step's issue registers (cycle 0 of the datapath); the last beat is still in hd
      i_acc <= 1'b0; i_launch <= 1'b0; i_q0 <= 1'b0; i_gv <= 1'b0; i_w <= 1'b0; i_bk <= 2'd0;
      gl_go <= 1'b0;
      if (gl_rem != '0 && oq_free) begin
        gl_go <= 1'b1; gl_ja <= gl_j; gl_la <= gl_a;
        gl_j <= gl_j + LMB'(1); gl_a <= gl_a + 64'(ldy); gl_rem <= gl_rem - (LMB + 1)'(1);
      end
      if (fire && op == O_GV && h_launch) begin
        gl_rem <= (LMB + 1)'(m_n - 32'd1); gl_j <= LMB'(1); gl_a <= qt_h.laddr + 64'(ldy);
      end
      if (fire && op == O_GV) begin         // a W beat: the step's operands (x's beat is read now)
        if (gw) gv_w <= (128 * NXB)'({qh_h, hd});
        else begin
          gv_w <= '0;
          gv_w[128 * int'(gx_i[LNX-1:0]) +: 128] <= hd;
        end
        i_gv <= 1'b1; i_gf <= qt_h.fi; i_ge <= h_launch;
        gidx <= qt_h.fi ? 8'd1 : gidx + 8'd1;
        if (h_launch) begin
          gv_sc <= csb[32 * ((qm == 2'd1) ? 0 : int'(qt_h.ws)) +: 32];
          gv_bi <= ccb[32 * qt_h.ws +: 32];
        end
      end
      if (fire) begin
        for (int i = 0; i < 4; i++) begin
          logic [31:0] b_e;
          b_e = (hk == K_B) ? hd_value[i] : cb[i];
          ia[i] <= a_now[i];
          case (op)
            O_ELT: begin
              ib[i] <= (bmode[1]) ? (hk == K_B ? hd[32 * qt_h.ws +: 32] : cbw[32 * qt_h.ws +: 32]) : b_e;
              ic[i] <= (hk == K_C) ? hd_value[i] : cc[i];
            end
            O_ROPE: begin
              ixp[i] <= cxp[i]; ics[i] <= ccos[i]; isn[i] <= (hk == K_SIN) ? hd[32 * i +: 32] : csin[i];
            end
            O_NT: ib[i] <= b_e;
            O_NN: begin
              // step 0 of a k follows the A beat (collected): its word k mod 4, held for steps 1-3
              ia[i] <= (qt_h.lg == 2'd0) ? ca[qt_h.ws] : nn_a;
              ib[i] <= b_e;
            end
            default: ;
          endcase
        end
        if (op == O_NN && qt_h.lg == 2'd0) nn_a <= ca[qt_h.ws];
        i_rot <= qt_h.ro; i_neg <= qt_h.ng; i_lg <= qt_h.lg;
        i_acc <= op == O_NT || op == O_NN;
        i_launch <= h_launch; i_laddr <= qt_h.laddr; i_llen <= 32'(qt_h.llen);
        i_q0 <= op == O_Q && !qt_h.q1; i_qfi <= qt_h.fi;
        if (qt_h.pe && (!fw_nn || h_done)) begin c_wait <= 1'b1; w_ph <= 3'd0; w_cnt <= 8'd0; ro_addr <= qt_h.roaddr; end
        if (fw_nn) begin                    // A[m][k] on all 16, B's 64 bytes
          logic [31:0] aw1;
          aw1 = (qt_h.lg == 2'd0) ? ca[qt_h.ws] : nn_a;
          i_w <= 1'b1; i_bk <= half_in ? hseg : {qt_h.lg[0], hfl};
          for (int i = 0; i < 4; i++) ib[i] <= half_in ? wide_half[i] : hv[32 * i +: 32];
          for (int i = 0; i < 12; i++) begin
            ia2[i] <= aw1; ib2[i] <= half_in ? wide_half[i + 4] : hv[32 * (i + 4) +: 32];
          end
        end
      end
      if (g_run) begin
        i_acc <= 1'b1; i_w <= 1'b1; i_bk <= g_k; i_lg <= 2'd0;
        for (int i = 0; i < 4; i++) begin ia[i] <= av[32 * i +: 32]; ib[i] <= hv[32 * i +: 32]; end
        for (int i = 0; i < 12; i++) begin ia2[i] <= av[32 * (i + 4) +: 32]; ib2[i] <= hv[32 * (i + 4) +: 32]; end
      end
      // QUANT pass 0: running max of |x| per slot, from the issue registers (a cycle after the fire)
      if (i_q0)
        for (int i = 0; i < 4; i++) qmax[i] <= i_qfi ? fabs(ia[i]) : fmax(qmax[i], fabs(ia[i]));
      // lane accumulators
      for (int l = 0; l < 16; l++) begin
        lane_p[l] <= {lane_p[l][1:0], lane_in[l]};
        if (lane_p[l][2]) lv[bk6][l] <= 1'b1;
      end

      // -------------------------------------------------------- pass ends, readouts
      ro_go <= 1'b0;
      if (FX && f_v) begin                  // behind the next pass: NT the fold (12 cycles), then its sum;
        if (f_cnt != 4'd15) f_cnt <= f_cnt + 4'd1;      // NN the four lane groups
        if (fw_nt && f_cnt >= FO0 && f_cnt < FO0 + 4'd4) fo[2'(f_cnt - FO0)] <= fsum;   // a sum a cycle
        if (oq_free && !ro_go && (op == O_NN || f_cnt >= (fw_nt ? FO0 + 4'd4 : 4'd13))) begin
          ro_go <= 1'b1; ro_raw <= 1'b0; ro_la <= f_a;
          if (fw_nt) begin
            ro_v[0] <= fo[ro_q]; f_a <= f_a + 64'(es); ro_q <= ro_q + 2'd1;
            if (ro_q == 2'd3) f_v <= 1'b0;
          end else if (op == O_NT) begin ro_v[0] <= fsum; f_v <= 1'b0; end
          else begin
            for (int i = 0; i < 4; i++) ro_v[i] <= fl[int'(ro_q) * 4 + i];
            f_a <= f_a + 64'(es << 2);
            if (ro_q == 2'd3) f_v <= 1'b0;
            ro_q <= ro_q + 2'd1;
          end
        end
      end
      if (c_wait) begin
        w_cnt <= w_cnt + 8'd1;
        case (op)
          O_NT: case (w_ph)
            3'd0: if (w_cnt >= 8'd9 && !f_v) begin   // the last product is in its lane
              fl <= lane_s; lv[0] <= 16'h0; w_ph <= 3'd1; w_cnt <= 8'd0;
              if (FX) begin c_wait <= 1'b0; f_v <= 1'b1; f_cnt <= 4'd0; f_a <= ro_addr; end
            end
            3'd1: if (w_cnt == 8'd12) w_ph <= 3'd2;   // the fold is out
            3'd4: if (w_cnt >= 8'd9 && !f_v) begin     // 16 products: the banks into the fold, one a cycle
              fl <= lane_s; lv[0] <= 16'h0; w_ph <= 3'd5; l_fb <= 2'd1;
            end
            3'd5: begin fl <= lane_s; lv[1] <= 16'h0; w_ph <= 3'd6; l_fb <= 2'd2; end
            3'd6: begin fl <= lane_s; lv[2] <= 16'h0; w_ph <= 3'd7; l_fb <= 2'd3; end
            3'd7: begin
              fl <= lane_s; lv[3] <= 16'h0; c_wait <= 1'b0; l_fb <= 2'd0;
              f_v <= 1'b1; f_cnt <= 4'd0; f_a <= ro_addr; ro_q <= 2'd0;
            end
            default: if (oq_free && !ro_go) begin
              ro_go <= 1'b1; ro_raw <= 1'b0; ro_v[0] <= fsum; ro_la <= ro_addr; c_wait <= 1'b0;
            end
          endcase
          O_NN: case (w_ph)
            3'd0: if (w_cnt >= 8'd9 && !f_v) begin
              fl <= lane_s; lv[0] <= 16'h0; ro_q <= 2'd0; w_ph <= 3'd2;
              if (fw_nn) l_fb <= 2'd1;
              else if (FX) begin c_wait <= 1'b0; f_v <= 1'b1; f_cnt <= 4'd0; f_a <= ro_addr; end
            end
            3'd3: begin                                // 16 products: the next bank, 16 columns
              fl <= lane_s; lv[l_fb] <= 16'h0; w_ph <= 3'd2; l_fb <= l_fb + 2'd1;
            end
            default: if (oq_free && !ro_go) begin
              ro_go <= 1'b1; ro_raw <= 1'b0;
              for (int i = 0; i < 4; i++) ro_v[i] <= fl[int'(ro_q) * 4 + i];
              ro_la <= ro_addr; ro_addr <= ro_addr + 64'(es << 2);
              if (ro_q == 2'd3) begin
                if (fw_nn && l_fb != 2'd0) w_ph <= 3'd3;
                else begin c_wait <= 1'b0; l_fb <= 2'd0; end
              end
              ro_q <= ro_q + 2'd1;
            end
          endcase
          default: case (w_ph)                       // QUANT
            3'd0: begin                              // the last max is in at w_cnt 1; two fmax levels
              if (w_cnt == 8'd1) begin qm01 <= fmax(qmax[0], qmax[1]); qm23 <= fmax(qmax[2], qmax[3]); end
              if (w_cnt == 8'd2) begin amax <= fmax(qm01, qm23); w_ph <= 3'd1; w_cnt <= 8'd0; end
            end
            3'd1: if (w_cnt == 8'd42) w_ph <= 3'd2;   // recip (38) + fmul (3)
            default: if (oq_free && !ro_go) begin
              ro_go <= 1'b1; ro_raw <= 1'b1; ro_v[0] <= q_s; ro_la <= ro_addr; inv_q <= q_inv; c_wait <= 1'b0;
            end
          endcase
        endcase
      end
    end
  end

endmodule
