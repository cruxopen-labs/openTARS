// TARS-mini on the KV260 (Zynq UltraScale+): the core behind an AXI-Lite register block, four AXI4
// read masters to the PS DDR (r0..r3: S_AXI_HP0/HP3/HP2/HP1_FPD, 128 bits at the HP clock, each
// behind a clock-crossing FIFO: tm_hpr) and an AXI4 write master (256 bits at the core's clock;
// the interconnect converts it and shares S_AXI_HP1_FPD with r3).
//
// Address translation: the core's flat address space (MEM_BYTES) is split in pages of 2^PAGE_BITS
// bytes (2 MB: Linux hugepages); the page table (NPT entries, written through AXI-Lite) holds each
// page's physical page number. The board software allocates hugepages, writes their physical
// addresses here, and never needs a physically contiguous buffer.
//
// Reads: the core asks for one 16-byte eighth at a time, or (wide) a whole 128-byte beat; the
// bridge works in 128-byte beats: requests gather into bursts (up to 8 beats, never across a 1 KB
// block; several interleaved streams at once; the eighths of a beat asked one after the other
// share it), each burst goes to the read port of its 1 KB block (physical address bits 11:10: the
// four ports split a stream evenly), and the data comes back to the core in request order (a
// reorder buffer of two 64-byte halves, each written once a cycle).
//
// Ordering: the core expects a read to see every write it issued before (its port model: a write
// is done when accepted). The bridge keeps the address of every write not yet acknowledged (B) and
// holds a read of such 16 bytes (other reads go on); it reports idle only when every read
// has returned and every write is acknowledged, so the host sees the results in DDR once STATUS
// says halted and idle.
//
// Registers (byte offsets, 32 bits):
//   0x000 ID        "TARS" (0x54415253)          0x004 VERSION  0x00030400
//   0x008 CTRL      [0] start (write 1: a pulse)  [1] core reset (level, 1 = held in reset)
//   0x00C STATUS    [0] halted [1] done [7:4] err_code [8] bridge idle [9] read error (sticky)
//                   [10] write error (sticky) [11] core in reset
//   0x010 OP_IDX    0x014 Q_ADDR_LO  0x018 Q_ADDR_HI  0x01C Q_LEN
//   0x020 CYCLES_LO 0x024 CYCLES_HI (the core's busy cycles)
//   0x028 RD_BEATS (128-byte beats)  0x02C WR_BEATS (32-byte beats) (since the last core reset)
//   0x030 CLEAR     (write: clears the sticky errors)
//   0x034 PARAMS    [7:0] R [15:8] C [19:16] VG [27:24] PAGE_BITS - 16
//   0x038 MEM_MB    MEM_BYTES >> 20
//   performance (since the last core reset): 0x040 RD_BURSTS (AR sent) 0x044 RD_HAZARD (cycles a
//   read waits for a write to its 16 bytes) 0x048 AR_WAIT (cycles a burst waits for its port's AR
//   queue) 0x04C WR_WAIT (cycles the core's write waits) 0x050 RD_WAIT (cycles the core's read
//   waits, any reason) 0x054 RD_OCC_LO 0x058 RD_OCC_HI (sum over cycles of the read beats in
//   flight: / RD_BEATS = the mean read latency in cycles; read LO first)
//   0x05C PROFILE_CAP = 0x50524601 if PROFILE enabled, otherwise 0.
//   0x100 + 16*class + 4*metric: 10 classes, 32-bit counters since core reset.
//   Classes: overhead, GEMV, ELT, FDOT_NT, FDOT_NN, ROPE, QUANT, NORM, SOFTMAX, ACT.
//   Metrics: busy cycles, read-request wait, write-request wait, read/write hazard.
//   Waits overlap; they are observations during a class, not disjoint stall reasons.
//   0x2000 + 4 i    page table entry i: physical page number (physical address >> PAGE_BITS)
module tm_kv260_top #(
  parameter int     R         = 8,
  parameter int     C         = 2,
  parameter int     VG        = 1,
  parameter int     NB        = 4,
  parameter int     OT        = 64,
  parameter int     HB        = 256,
  parameter int     OQ        = 32,
  parameter int     LG        = 4,
  parameter int     VQMIN     = 32,
  parameter int     GB        = 4,               // GEMV rows (with 128-byte W beats: 4 x 128 INT8 MACs a cycle)
  parameter bit     PROFILE   = 1,               // observational counters; disable for area comparison
  parameter longint MEM_BYTES = 64'h4000_0000,   // 1 GB
  parameter int     PAGE_BITS = 21,              // 2 MB pages
  parameter int     NPT       = 512,             // page table entries (MEM_BYTES >> PAGE_BITS)
  parameter int     PA_BITS   = 40
) (
  input  logic          clk,
  input  logic          rstn,
  input  logic          hclk,                    // the HP read ports' clock
  input  logic          hrstn,
  // AXI-Lite slave (registers)
  input  logic [13:0]   s_axil_awaddr,
  input  logic          s_axil_awvalid,
  output logic          s_axil_awready,
  input  logic [31:0]   s_axil_wdata,
  input  logic [3:0]    s_axil_wstrb,
  input  logic          s_axil_wvalid,
  output logic          s_axil_wready,
  output logic [1:0]    s_axil_bresp,
  output logic          s_axil_bvalid,
  input  logic          s_axil_bready,
  input  logic [13:0]   s_axil_araddr,
  input  logic          s_axil_arvalid,
  output logic          s_axil_arready,
  output logic [31:0]   s_axil_rdata,
  output logic [1:0]    s_axil_rresp,
  output logic          s_axil_rvalid,
  input  logic          s_axil_rready,
  // AXI4 read masters (DDR), hclk: r0 (HP0), r1 (HP3), r2 (HP2), r3 (HP1, with the writes)
  output logic [PA_BITS-1:0] r0_axi_araddr,
  output logic [7:0]    r0_axi_arlen,
  output logic [2:0]    r0_axi_arsize,
  output logic [1:0]    r0_axi_arburst,
  output logic [3:0]    r0_axi_arcache,
  output logic [2:0]    r0_axi_arprot,
  output logic [0:0]    r0_axi_arid,
  output logic          r0_axi_arlock,
  output logic [3:0]    r0_axi_arqos,
  output logic          r0_axi_arvalid,
  input  logic          r0_axi_arready,
  input  logic [127:0]  r0_axi_rdata,
  input  logic [1:0]    r0_axi_rresp,
  input  logic          r0_axi_rlast,
  input  logic [0:0]    r0_axi_rid,
  input  logic          r0_axi_rvalid,
  output logic          r0_axi_rready,
  output logic [PA_BITS-1:0] r1_axi_araddr,
  output logic [7:0]    r1_axi_arlen,
  output logic [2:0]    r1_axi_arsize,
  output logic [1:0]    r1_axi_arburst,
  output logic [3:0]    r1_axi_arcache,
  output logic [2:0]    r1_axi_arprot,
  output logic [0:0]    r1_axi_arid,
  output logic          r1_axi_arlock,
  output logic [3:0]    r1_axi_arqos,
  output logic          r1_axi_arvalid,
  input  logic          r1_axi_arready,
  input  logic [127:0]  r1_axi_rdata,
  input  logic [1:0]    r1_axi_rresp,
  input  logic          r1_axi_rlast,
  input  logic [0:0]    r1_axi_rid,
  input  logic          r1_axi_rvalid,
  output logic          r1_axi_rready,
  output logic [PA_BITS-1:0] r2_axi_araddr,
  output logic [7:0]    r2_axi_arlen,
  output logic [2:0]    r2_axi_arsize,
  output logic [1:0]    r2_axi_arburst,
  output logic [3:0]    r2_axi_arcache,
  output logic [2:0]    r2_axi_arprot,
  output logic [0:0]    r2_axi_arid,
  output logic          r2_axi_arlock,
  output logic [3:0]    r2_axi_arqos,
  output logic          r2_axi_arvalid,
  input  logic          r2_axi_arready,
  input  logic [127:0]  r2_axi_rdata,
  input  logic [1:0]    r2_axi_rresp,
  input  logic          r2_axi_rlast,
  input  logic [0:0]    r2_axi_rid,
  input  logic          r2_axi_rvalid,
  output logic          r2_axi_rready,
  output logic [PA_BITS-1:0] r3_axi_araddr,
  output logic [7:0]    r3_axi_arlen,
  output logic [2:0]    r3_axi_arsize,
  output logic [1:0]    r3_axi_arburst,
  output logic [3:0]    r3_axi_arcache,
  output logic [2:0]    r3_axi_arprot,
  output logic [0:0]    r3_axi_arid,
  output logic          r3_axi_arlock,
  output logic [3:0]    r3_axi_arqos,
  output logic          r3_axi_arvalid,
  input  logic          r3_axi_arready,
  input  logic [127:0]  r3_axi_rdata,
  input  logic [1:0]    r3_axi_rresp,
  input  logic          r3_axi_rlast,
  input  logic [0:0]    r3_axi_rid,
  input  logic          r3_axi_rvalid,
  output logic          r3_axi_rready,
  // AXI4 write master (DDR), clk
  output logic [PA_BITS-1:0] m_axi_awaddr,
  output logic [7:0]    m_axi_awlen,
  output logic [2:0]    m_axi_awsize,
  output logic [1:0]    m_axi_awburst,
  output logic [3:0]    m_axi_awcache,
  output logic [2:0]    m_axi_awprot,
  output logic [0:0]    m_axi_awid,
  output logic          m_axi_awlock,
  output logic [3:0]    m_axi_awqos,
  output logic          m_axi_awvalid,
  input  logic          m_axi_awready,
  output logic [255:0]  m_axi_wdata,
  output logic [31:0]   m_axi_wstrb,
  output logic          m_axi_wlast,
  output logic          m_axi_wvalid,
  input  logic          m_axi_wready,
  input  logic [1:0]    m_axi_bresp,
  input  logic [0:0]    m_axi_bid,
  input  logic          m_axi_bvalid,
  output logic          m_axi_bready
);
  logic        rp_go, rp_err;                    // returned read data is taken this cycle; an error response among it
  localparam int LPT = $clog2(NPT);

  // ------------------------------------------------------------------ registers
  logic        ctl_rst, start_p, err_r, err_w;
  logic [63:0] q_addr;
  logic [31:0] q_len, rd_beats, wr_beats, cyc_hi_snap;
  logic [31:0] p_bursts, p_haz, p_arw, p_ww, p_rw, lat_hi_snap;
  logic [47:0] p_lat;
  logic        halted, done;
  logic [3:0]  err_code;
  logic [31:0] op_idx;
  logic [63:0] cycles;
  logic        br_idle;
  logic        core_rst;
  logic [3:0]  profile_class;
  logic [31:0] profile_count [10][4];
  assign core_rst = !rstn || ctl_rst;

  // the page table: written here, read by the read and the write address paths
  logic        pt_we;
  logic [31:0] pt_wa, pt_wd;
  logic [31:0] pt_ra [2];
  logic [31:0] pt_rd [2];
  tm_lutram #(.W(32), .D(NPT), .NR(2)) u_pt (.clk, .we(pt_we), .wa(pt_wa), .wd(pt_wd), .ra(pt_ra), .rd(pt_rd));

  // write: address and data both present, one response
  logic        aw_ok;
  logic [13:0] wa;
  assign aw_ok          = s_axil_awvalid && s_axil_wvalid && !s_axil_bvalid;
  assign s_axil_awready = aw_ok;
  assign s_axil_wready  = aw_ok;
  assign s_axil_bresp   = 2'b00;
  assign wa             = {s_axil_awaddr[13:2], 2'b00};
  assign pt_we          = aw_ok && wa[13];
  assign pt_wa          = 32'(wa[12:2]);
  assign pt_wd          = s_axil_wdata;

  always_ff @(posedge clk) begin
    if (!rstn) begin
      s_axil_bvalid <= 1'b0; ctl_rst <= 1'b1; start_p <= 1'b0; q_addr <= '0; q_len <= '0;
      err_r <= 1'b0; err_w <= 1'b0;
    end else begin
      start_p <= 1'b0;
      if (s_axil_bvalid && s_axil_bready) s_axil_bvalid <= 1'b0;
      if (aw_ok) begin
        s_axil_bvalid <= 1'b1;
        if (!wa[13])
          case (wa[12:0])
            13'h008: begin start_p <= s_axil_wdata[0]; ctl_rst <= s_axil_wdata[1]; end
            13'h014: q_addr[31:0]  <= s_axil_wdata;
            13'h018: q_addr[63:32] <= s_axil_wdata;
            13'h01C: q_len <= s_axil_wdata;
            13'h030: begin err_r <= 1'b0; err_w <= 1'b0; end
            default: ;
          endcase
      end
      if (rp_go && rp_err) err_r <= 1'b1;
      if (m_axi_bvalid && m_axi_bresp[1]) err_w <= 1'b1;
    end
  end

  // read: one cycle
  always_ff @(posedge clk) begin
    if (!rstn) begin
      s_axil_rvalid <= 1'b0; s_axil_rdata <= '0; cyc_hi_snap <= '0; lat_hi_snap <= '0;
    end else begin
      if (s_axil_rvalid && s_axil_rready) s_axil_rvalid <= 1'b0;
      if (s_axil_arvalid && !s_axil_rvalid) begin
        s_axil_rvalid <= 1'b1;
        case ({s_axil_araddr[13:2], 2'b00})
          14'h000: s_axil_rdata <= 32'h5441_5253;
          14'h004: s_axil_rdata <= 32'h0003_0400;
          14'h008: s_axil_rdata <= {30'b0, ctl_rst, 1'b0};
          14'h00C: s_axil_rdata <= {20'b0, core_rst, err_w, err_r, br_idle, err_code, 2'b0, done, halted};
          14'h010: s_axil_rdata <= op_idx;
          14'h014: s_axil_rdata <= q_addr[31:0];
          14'h018: s_axil_rdata <= q_addr[63:32];
          14'h01C: s_axil_rdata <= q_len;
          14'h020: begin s_axil_rdata <= cycles[31:0]; cyc_hi_snap <= cycles[63:32]; end
          14'h024: s_axil_rdata <= cyc_hi_snap;
          14'h028: s_axil_rdata <= rd_beats;
          14'h02C: s_axil_rdata <= wr_beats;
          14'h034: s_axil_rdata <= {4'b0, 4'(PAGE_BITS - 16), 4'b0, 4'(VG), 8'(C), 8'(R)};
          14'h038: s_axil_rdata <= 32'(MEM_BYTES >> 20);
          14'h040: s_axil_rdata <= p_bursts;
          14'h044: s_axil_rdata <= p_haz;
          14'h048: s_axil_rdata <= p_arw;
          14'h04C: s_axil_rdata <= p_ww;
          14'h050: s_axil_rdata <= p_rw;
          14'h054: begin s_axil_rdata <= p_lat[31:0]; lat_hi_snap <= 32'(p_lat[47:32]); end
          14'h058: s_axil_rdata <= lat_hi_snap;
          14'h05C: s_axil_rdata <= PROFILE ? 32'h5052_4601 : 32'd0;
          default: begin
            if (PROFILE && s_axil_araddr[13:8] == 6'h01 && s_axil_araddr[7:4] < 4'd10)
              s_axil_rdata <= profile_count[s_axil_araddr[7:4]][s_axil_araddr[3:2]];
            else s_axil_rdata <= 32'h0;
          end
        endcase
      end
    end
  end
  assign s_axil_arready = !s_axil_rvalid;
  assign s_axil_rresp   = 2'b00;

  // ------------------------------------------------------------------ the core
  logic          mr_req, mr_ready, mr_rvalid, mw_req, mw_ready;
  logic [63:0]   mr_addr, mw_addr;
  logic          mr_wide;
  logic [127:0]  mr_rdata, mw_data;
  logic [895:0]  mr_rhi;
  logic [15:0]   mw_strb;
  tm_core #(.R(R), .C(C), .MEM_BYTES(MEM_BYTES), .NB(NB), .OT(OT), .HB(HB), .VG(VG), .LG(LG), .OQ(OQ),
            .VQMIN(VQMIN), .GB(GB), .RW(8), .ARR(0)) u_core (
    .clk, .rst(core_rst), .start(start_p), .q_addr, .q_len, .halted, .done, .err_code, .op_idx, .cycles, .profile_class,
    .mr_req, .mr_addr, .mr_wide, .mr_ready, .mr_rvalid, .mr_rdata, .mr_rhi,
    .mw_req, .mw_addr, .mw_data, .mw_strb, .mw_ready);

  // ------------------------------------------------------------------ the bridge
  localparam int LA  = $clog2(MEM_BYTES) - 7;   // a read (128-byte) beat's address in the core's space
  localparam int LW  = $clog2(MEM_BYTES) - 5;   // a write (32-byte) beat's
  localparam int LH  = $clog2(MEM_BYTES) - 4;   // a 16-byte eighth's (the hazard checks)
  localparam int BB  = (PAGE_BITS < 10) ? PAGE_BITS : 10;   // a read burst stays in one 2^BB block (1 KB: one port's)
  localparam int BBW = (PAGE_BITS < 12) ? PAGE_BITS : 12;   // a write burst in one 2^BBW block (4 KB, a page)
  localparam int MAXB = 8;                        // beats per read burst at most (1 KB)
  localparam int LN = $clog2(MAXB) + 1;
  localparam int GT = 4;                          // a gathered burst waits this many idle cycles for its next beat
  localparam int NW = 16;                         // write bursts gathered or in flight (the read hazard check)
  localparam int LNW = $clog2(NW);
  localparam int MW = 16;                         // beats per write burst at most
  localparam int LMW = $clog2(MW);
  localparam int LNM = LMW + 1;                  // a write burst's beat count
  initial if (PAGE_BITS < 7) $fatal(1, "PAGE_BITS < 7: a 128-byte beat could cross a page");

  function automatic logic [PA_BITS-1:0] xlate(input logic [63:0] a, input logic [31:0] ppn, input int lo);
    logic [63:0] o;
    o = 64'(a[PAGE_BITS-1:0]) >> lo << lo;
    return PA_BITS'({ppn, o[PAGE_BITS-1:0]});
  endfunction
  // [a, a + na) and [b, b + nb) overlap, in 16-byte units (hazards: a write and a read of
  // different 16 bytes of a beat do not wait for each other)
  function automatic logic ovq(input logic [LH-1:0] a, input logic [7:0] na, input logic [LH-1:0] b,
                               input logic [7:0] nb);
    return LH'(a - b) < LH'(nb) || LH'(b - a) < LH'(na);
  endfunction

  // writes: gathered into bursts (slots of a ring, NW, in core order): a write joins the open
  // slot when it is to its last beat (the strobes merge: an element at a time from the GEMV unit
  // fills one beat) or to the next one (up to MW beats, one 2^BBW block), and opens a new slot
  // otherwise. The open slot's last beat is a register; earlier beats are in a BRAM. A slot
  // closes when a write does not join it, when it has not grown for GT cycles, or when a read
  // waits for it; then AW goes out (once no read the core issued before overlaps it), then its
  // beats on W, and the slot is free at its response (B: in order, one ID). A write never joins
  // the open slot at the 16 bytes a read in the input register waits on: that read is older than
  // the write and must not see it.
  logic [LW-1:0]  wb_b [NW];                   // a slot's first beat, its address, its beats
  logic [63:0]    wb_a [NW];
  logic [LNM-1:0]  wb_n [NW];
  logic [LH-1:0]  wb_h [NW];                   // ... the 16-byte units it writes: from wb_h, wb_m of them
  logic [7:0]     wb_m [NW];
  logic [LH-1:0]  w_h16, in_h16;               // the write's 16-byte unit; the read's (its first)
  logic [7:0]     in_nq;                       // the read's eighths (1, or 8: a wide one)
  logic [NW-1:0]  ws_v, ws_nv;                 // live (opened, no response yet); after this cycle
  logic [LNW:0]   w_hd, w_aw, w_rp, w_tl;      // oldest live; next AW; next to read for W; next to open
  logic           wo_v;                        // a slot is open
  logic [LNW-1:0] w_op;
  logic [2:0]     wo_age;
  logic [255:0]   lb_d;                        // the open slot's last beat
  logic [31:0]    lb_s, w_s32;
  logic [255:0]   w_d32;                       // the core's 16-byte write placed in its half of the beat
  logic [LW-1:0]  w_beat;
  logic           w_take, w_blk, j_same, j_next, w_cls, aw_go;
  logic [LNW-1:0] aw_s;
  logic [15:0]  rd_out;                        // read beats asked for (AR queued), not yet returned
  logic         w_war;

  // reads: a request (a 16-byte eighth, or wide: a whole 128-byte beat) is taken into an input
  // register (nothing combinational from the core's address back to mr_ready) and gets the next
  // place in the request queue (request order). From there beats gather into bursts: NS streams
  // at once (a unit reading A, B and C beat by beat makes three interleaved streams; one burst a
  // stream, not one a beat: a PS port takes a limited number of bursts in flight), each burst up
  // to MAXB consecutive beats in one 2^BB block. A request joins a stream at its last beat (the
  // eighths not asked yet: eighths read one after the other share the beat) or at the next beat.
  // A burst goes out when it is full, when its next beat would leave the block, when it has not
  // grown for GT cycles, or when a request finds no stream and every stream is taken; it goes to
  // its block's port (ar_p). Each beat of a burst has a reorder-buffer slot (a ring, NR, in the
  // order the beats were added), kept in the burst's slot list (one of NBB, BRAM); each port
  // returns its bursts in its AR order (one ID), as lo and hi halves; a lo half and a hi half
  // (from any ports) land at their slots each cycle, and the core gets the requests in order,
  // each its eighth (or the whole beat) of its slot. A slot is free again once no stream can add
  // to it and the last request on it is delivered (rq_last). A read waits while a write the core
  // issued before it (in_old: the writes in flight when the read was taken, and one taken in the
  // same cycle) to the same 16 bytes is not acknowledged; other reads go on.
  localparam int NS  = 4;
  localparam int NBB = 32;                     // bursts gathered, on AR or in flight
  localparam int LBB = $clog2(NBB);
  localparam int NR  = 256;                    // reorder buffer: beats
  localparam int LR  = $clog2(NR);
  localparam int NQ  = 256;                    // requests taken and not delivered (the core keeps at most 128 in flight)
  localparam int LQ  = $clog2(NQ);
  logic                in_v, in_w;
  logic [2:0]          in_q;                   // the eighth (narrow)
  logic [63:0]         in_addr;
  logic [LA-1:0]       in_b;
  logic [7:0]          in_m;                   // the eighths it asks for
  logic [NW-1:0]       in_old;
  logic                g_v [NS];               // a burst being gathered
  logic [LA-1:0]       g_b [NS], g_nx [NS];    // its first beat; the beat that would extend it
  logic [LN-1:0]       g_n [NS];
  logic [2:0]          g_age [NS];             // cycles since the burst last grew
  logic [PA_BITS-1:0]  g_pa [NS];
  logic [LBB-1:0]      g_bf [NS];              // its slot list
  logic [LR-1:0]       g_ls [NS];              // its last beat's slot
  logic [7:0]          g_lm [NS];              // ... and the eighths asked of it
  logic                ar_v;                   // the burst for a port's AR queue (held until taken)
  logic [PA_BITS-1:0]  ar_a;
  logic [2:0]          ar_l;
  logic [LA-1:0]       ar_b;
  logic [LN-1:0]       ar_n;
  logic [LBB-1:0]      ar_bf;
  logic [1:0]          ar_p;
  logic                ar_acc;
  logic [NBB-1:0]      bf_free;
  logic [LBB-1:0]      fl_q [4][NBB];          // per port: slot lists of its bursts on the bus, AR order
  logic [LBB:0]        fl_hd [4], fl_tl [4];
  logic [LN-1:0]       r_k [4];                // beat within the port's returning burst
  logic [LQ:0]         s_wr, s_rd;             // requests taken; delivered
  logic [LR:0]         a_tl, a_hd;             // slots given out; freed
  logic                in_haz, ar_free, in_go, g_close, hs, hn, fg_ok, bf_ok, sl_ok, ev, newb;
  localparam int LS  = $clog2(NS);
  logic [LS-1:0]       hi, fg, cl;
  logic [LBB-1:0]      nbf;
  logic [LR-1:0]       slot;                   // the taken request's slot
  // the ports (tm_hpr): AR queue in, 64-byte halves out (lo, hi of each beat, in order); each
  // cycle the reorder buffer takes a lo half from one port (channel L) and a hi half from one
  // (channel H)
  logic                hp_arv [4], hp_arr [4], hp_rv [4], hp_rr [4], hp_rh [4], hp_rl [4], hp_re [4];
  logic [511:0]        hp_rd [4];
  logic                cl_go, ch_go;           // channel L / H takes a half
  logic [1:0]          cl_s, ch_s, cl_pri, ch_pri;   // its port; the one preferred next

  assign pt_ra[0] = 32'(in_addr[PAGE_BITS +: LPT]);
  assign pt_ra[1] = 32'(wb_a[aw_s][PAGE_BITS +: LPT]);

  assign in_b   = in_addr[LA+6:7];
  assign in_h16 = in_addr[LA+6:4];
  assign in_nq  = in_w ? 8'd8 : 8'd1;
  assign in_m   = in_w ? 8'hFF : 8'h01 << in_q;
  always_comb begin
    in_haz = 1'b0;
    for (int i = 0; i < NW; i++)
      if (in_old[i] && ovq(in_h16, in_nq, wb_h[i], wb_m[i])) in_haz = 1'b1;
    // the stream this read joins (hs: at its last beat; hn: at the next one); a free stream; a free slot list
    hs = 1'b0; hn = 1'b0; hi = '0; fg_ok = 1'b0; fg = '0;
    for (int i = NS - 1; i >= 0; i--) begin
      if (g_v[i] && in_b == g_nx[i] - LA'(1) && (g_lm[i] & in_m) == 8'h00) begin hs = 1'b1; hn = 1'b0; hi = LS'(i); end
      else if (!hs && g_v[i] && in_b == g_nx[i] && g_n[i] != LN'(MAXB) && in_b[BB-8:0] != '0) begin hn = 1'b1; hi = LS'(i); end
      if (!g_v[i]) begin fg_ok = 1'b1; fg = LS'(i); end
    end
    bf_ok = 1'b0; nbf = '0;
    for (int i = NBB - 1; i >= 0; i--) if (bf_free[i]) begin bf_ok = 1'b1; nbf = LBB'(i); end
  end
  assign newb    = !hs;                        // the request adds a beat (a slot)
  assign slot    = hs ? g_ls[hi] : a_tl[LR-1:0];
  assign sl_ok   = (LR + 1)'(a_tl - a_hd) != (LR + 1)'(NR);
  assign ar_acc  = ar_v && hp_arr[ar_p];
  assign ar_free = !ar_v || ar_acc;
  assign in_go   = in_v && !in_haz && rd_out < 16'hFF00 && (LQ + 1)'(s_wr - s_rd) != (LQ + 1)'(NQ) &&
                   (hs || (sl_ok && (hn || (fg_ok && bf_ok))));
  assign mr_ready = !core_rst && (!in_v || in_go);
  // a read with no stream and none free makes room (ev): the stream idle longest goes out
  assign ev = in_v && !in_haz && !hs && !hn && !fg_ok;
  always_comb begin
    logic [2:0] best;
    g_close = 1'b0; cl = '0; best = '0;
    for (int i = NS - 1; i >= 0; i--)
      if (g_v[i] && !(in_go && (hs || hn) && 32'(hi) == i) &&
          (32'(g_age[i]) >= GT - 1 || g_n[i] == LN'(MAXB) || g_nx[i][BB-8:0] == '0 || (ev && g_age[i] >= best))) begin
        g_close = ar_free; cl = LS'(i); best = g_age[i];
      end
  end

  // the ports; the halves they return, a lo and a hi a cycle (the preferred port first, then in
  // turn), each with its slot from its burst's slot list (two copies: one read a channel)
  for (genvar p = 0; p < 4; p++) begin : g_hp
    assign hp_arv[p] = ar_v && ar_p == 2'(p);
    assign hp_rr[p]  = hp_rv[p] && ((cl_go && cl_s == 2'(p) && !hp_rh[p]) || (ch_go && ch_s == 2'(p) && hp_rh[p]));
  end
  tm_hpr #(.PA(PA_BITS)) u_hp0 (.clk, .rst(!rstn), .ar_v(hp_arv[0]), .ar_rdy(hp_arr[0]), .ar_a(ar_a), .ar_n1(ar_l),
    .r_v(hp_rv[0]), .r_rdy(hp_rr[0]), .r_d(hp_rd[0]), .r_hi(hp_rh[0]), .r_last(hp_rl[0]), .r_err(hp_re[0]),
    .hclk, .hrst(!hrstn),
    .m_araddr(r0_axi_araddr), .m_arlen(r0_axi_arlen), .m_arsize(r0_axi_arsize), .m_arburst(r0_axi_arburst),
    .m_arcache(r0_axi_arcache), .m_arprot(r0_axi_arprot), .m_arid(r0_axi_arid), .m_arlock(r0_axi_arlock),
    .m_arqos(r0_axi_arqos), .m_arvalid(r0_axi_arvalid), .m_arready(r0_axi_arready), .m_rdata(r0_axi_rdata),
    .m_rresp(r0_axi_rresp), .m_rlast(r0_axi_rlast), .m_rvalid(r0_axi_rvalid), .m_rready(r0_axi_rready));
  tm_hpr #(.PA(PA_BITS)) u_hp1 (.clk, .rst(!rstn), .ar_v(hp_arv[1]), .ar_rdy(hp_arr[1]), .ar_a(ar_a), .ar_n1(ar_l),
    .r_v(hp_rv[1]), .r_rdy(hp_rr[1]), .r_d(hp_rd[1]), .r_hi(hp_rh[1]), .r_last(hp_rl[1]), .r_err(hp_re[1]),
    .hclk, .hrst(!hrstn),
    .m_araddr(r1_axi_araddr), .m_arlen(r1_axi_arlen), .m_arsize(r1_axi_arsize), .m_arburst(r1_axi_arburst),
    .m_arcache(r1_axi_arcache), .m_arprot(r1_axi_arprot), .m_arid(r1_axi_arid), .m_arlock(r1_axi_arlock),
    .m_arqos(r1_axi_arqos), .m_arvalid(r1_axi_arvalid), .m_arready(r1_axi_arready), .m_rdata(r1_axi_rdata),
    .m_rresp(r1_axi_rresp), .m_rlast(r1_axi_rlast), .m_rvalid(r1_axi_rvalid), .m_rready(r1_axi_rready));
  tm_hpr #(.PA(PA_BITS)) u_hp2 (.clk, .rst(!rstn), .ar_v(hp_arv[2]), .ar_rdy(hp_arr[2]), .ar_a(ar_a), .ar_n1(ar_l),
    .r_v(hp_rv[2]), .r_rdy(hp_rr[2]), .r_d(hp_rd[2]), .r_hi(hp_rh[2]), .r_last(hp_rl[2]), .r_err(hp_re[2]),
    .hclk, .hrst(!hrstn),
    .m_araddr(r2_axi_araddr), .m_arlen(r2_axi_arlen), .m_arsize(r2_axi_arsize), .m_arburst(r2_axi_arburst),
    .m_arcache(r2_axi_arcache), .m_arprot(r2_axi_arprot), .m_arid(r2_axi_arid), .m_arlock(r2_axi_arlock),
    .m_arqos(r2_axi_arqos), .m_arvalid(r2_axi_arvalid), .m_arready(r2_axi_arready), .m_rdata(r2_axi_rdata),
    .m_rresp(r2_axi_rresp), .m_rlast(r2_axi_rlast), .m_rvalid(r2_axi_rvalid), .m_rready(r2_axi_rready));
  tm_hpr #(.PA(PA_BITS)) u_hp3 (.clk, .rst(!rstn), .ar_v(hp_arv[3]), .ar_rdy(hp_arr[3]), .ar_a(ar_a), .ar_n1(ar_l),
    .r_v(hp_rv[3]), .r_rdy(hp_rr[3]), .r_d(hp_rd[3]), .r_hi(hp_rh[3]), .r_last(hp_rl[3]), .r_err(hp_re[3]),
    .hclk, .hrst(!hrstn),
    .m_araddr(r3_axi_araddr), .m_arlen(r3_axi_arlen), .m_arsize(r3_axi_arsize), .m_arburst(r3_axi_arburst),
    .m_arcache(r3_axi_arcache), .m_arprot(r3_axi_arprot), .m_arid(r3_axi_arid), .m_arlock(r3_axi_arlock),
    .m_arqos(r3_axi_arqos), .m_arvalid(r3_axi_arvalid), .m_arready(r3_axi_arready), .m_rdata(r3_axi_rdata),
    .m_rresp(r3_axi_rresp), .m_rlast(r3_axi_rlast), .m_rvalid(r3_axi_rvalid), .m_rready(r3_axi_rready));
  always_comb begin
    cl_go = 1'b0; cl_s = cl_pri; ch_go = 1'b0; ch_s = ch_pri;
    for (int k = 3; k >= 0; k--) begin
      if (hp_rv[2'(cl_pri + 2'(k))] && !hp_rh[2'(cl_pri + 2'(k))]) begin cl_go = 1'b1; cl_s = 2'(cl_pri + 2'(k)); end
      if (hp_rv[2'(ch_pri + 2'(k))] && hp_rh[2'(ch_pri + 2'(k))]) begin ch_go = 1'b1; ch_s = 2'(ch_pri + 2'(k)); end
    end
  end
  assign rp_go  = cl_go || ch_go;
  assign rp_err = (cl_go && hp_re[cl_s]) || (ch_go && hp_re[ch_s]);

  // slot lists: written as beats join a burst, read as their halves return (one cycle, registered)
  logic          sl_we;
  logic [LBB+LN-2:0] sl_wa, sll_ra, slh_ra;
  logic [LR-1:0] sll_q, slh_q;
  assign sl_we  = in_go && newb;
  assign sl_wa  = hn ? {g_bf[hi], (LN - 1)'(g_n[hi])} : {nbf, (LN - 1)'(0)};
  assign sll_ra = {fl_q[cl_s][fl_hd[cl_s][LBB-1:0]], (LN - 1)'(r_k[cl_s])};
  assign slh_ra = {fl_q[ch_s][fl_hd[ch_s][LBB-1:0]], (LN - 1)'(r_k[ch_s])};
  tm_bram #(.W(LR), .D(NBB * MAXB)) u_sll (.clk, .we(sl_we), .wa(sl_wa), .wd(a_tl[LR-1:0]), .ra(sll_ra), .rd(sll_q));
  tm_bram #(.W(LR), .D(NBB * MAXB)) u_slh (.clk, .we(sl_we), .wa(sl_wa), .wd(a_tl[LR-1:0]), .ra(slh_ra), .rd(slh_q));
  // the request queue (slot, eighth, wide) and, per slot, its last request (distributed RAM)
  logic [31:0]   rq_ra [1], lq_ra [1];
  logic [LR+3:0] rq_rd [1];
  logic [LQ:0]   lq_rd [1];
  assign rq_ra[0] = 32'(s_rd[LQ-1:0]);
  assign lq_ra[0] = 32'(a_hd[LR-1:0]);
  tm_lutram #(.W(LR + 4), .D(NQ), .NR(1)) u_rq (.clk, .we(in_go), .wa(32'(s_wr[LQ-1:0])), .wd({in_w, in_q, slot}),
                                               .ra(rq_ra), .rd(rq_rd));
  tm_lutram #(.W(LQ + 1), .D(NR), .NR(1)) u_lq (.clk, .we(in_go), .wa(32'(slot)), .wd(s_wr), .ra(lq_ra), .rd(lq_rd));
  // reorder buffer, two halves: a returned half (registered with its slot) is written; the next
  // request's slot is read once both halves are there, and its eighth (or the whole beat) goes to
  // the core a cycle later
  logic          r1l_v, r1h_v, d1_v, d1_w;
  logic [2:0]    d1_q;
  logic [511:0]  r1l_d, r1h_d, rbl_q, rbh_q;
  logic [NR-1:0] rob_vl, rob_vh;
  logic          d_go, a_free, a_open;
  logic [LQ:0]   a_dq;
  assign d_go  = s_wr != s_rd && rob_vl[rq_rd[0][LR-1:0]] && rob_vh[rq_rd[0][LR-1:0]];
  tm_bram #(.W(512), .D(NR)) u_robl (.clk, .we(r1l_v), .wa(sll_q), .wd(r1l_d), .ra(rq_rd[0][LR-1:0]), .rd(rbl_q));
  tm_bram #(.W(512), .D(NR)) u_robh (.clk, .we(r1h_v), .wa(slh_q), .wd(r1h_d), .ra(rq_rd[0][LR-1:0]), .rd(rbh_q));
  // the oldest slot is free once no stream can add to it (it is not a stream's last beat) and its
  // last request is delivered
  always_comb begin
    a_open = 1'b0;
    for (int i = 0; i < NS; i++) if (g_v[i] && g_ls[i] == a_hd[LR-1:0]) a_open = 1'b1;
    a_dq   = s_rd - lq_rd[0];
    a_free = a_hd != a_tl && !a_open && a_dq != '0 && 32'(a_dq) <= NQ && !(in_go && hs && g_ls[hi] == a_hd[LR-1:0]);
  end

  always_ff @(posedge clk)
    if (!rstn) begin
      in_v <= 1'b0; in_old <= '0; ar_v <= 1'b0; bf_free <= '1; cl_pri <= '0; ch_pri <= '0;
      for (int p = 0; p < 4; p++) begin fl_hd[p] <= '0; fl_tl[p] <= '0; r_k[p] <= '0; end
      s_wr <= '0; s_rd <= '0; a_tl <= '0; a_hd <= '0; rob_vl <= '0; rob_vh <= '0; r1l_v <= 1'b0; r1h_v <= 1'b0;
      d1_v <= 1'b0;
      for (int i = 0; i < NS; i++) g_v[i] <= 1'b0;
    end else begin
      if (mr_req && mr_ready) begin
        in_v <= 1'b1; in_addr <= mr_addr; in_q <= mr_addr[6:4]; in_w <= mr_wide; in_old <= ws_nv;
      end else begin
        if (in_go) in_v <= 1'b0;
        in_old <= in_old & ws_nv;
      end
      for (int i = 0; i < NS; i++)
        if (g_v[i] && g_age[i] != '1) g_age[i] <= g_age[i] + 3'd1;
      if (g_close) begin
        ar_v <= 1'b1; ar_a <= g_pa[cl]; ar_l <= 3'(g_n[cl] - LN'(1)); ar_b <= g_b[cl]; ar_n <= g_n[cl];
        ar_bf <= g_bf[cl]; ar_p <= g_pa[cl][11:10]; g_v[cl] <= 1'b0;
      end else if (ar_acc) ar_v <= 1'b0;
      if (in_go) begin
        s_wr <= s_wr + (LQ + 1)'(1);
        if (newb) a_tl <= a_tl + (LR + 1)'(1);
        if (hs) begin
          g_lm[hi] <= g_lm[hi] | in_m; g_age[hi] <= '0;
        end else if (hn) begin
          g_n[hi] <= g_n[hi] + LN'(1); g_nx[hi] <= g_nx[hi] + LA'(1); g_age[hi] <= '0;
          g_ls[hi] <= a_tl[LR-1:0]; g_lm[hi] <= in_m;
        end else begin
          g_v[fg] <= 1'b1; g_b[fg] <= in_b; g_nx[fg] <= in_b + LA'(1); g_n[fg] <= LN'(1); g_age[fg] <= '0;
          g_pa[fg] <= xlate(in_addr, pt_rd[0], 7); g_bf[fg] <= nbf; g_ls[fg] <= a_tl[LR-1:0]; g_lm[fg] <= in_m;
        end
      end
      // bursts on the bus: per port, their slot lists in AR order; a list is free once its last
      // beat's slot is read
      if (ar_acc) begin
        fl_q[ar_p][fl_tl[ar_p][LBB-1:0]] <= ar_bf; fl_tl[ar_p] <= fl_tl[ar_p] + (LBB + 1)'(1);
      end
      r1l_v <= cl_go; r1l_d <= hp_rd[cl_s];
      r1h_v <= ch_go; r1h_d <= hp_rd[ch_s];
      if (cl_go) cl_pri <= cl_s + 2'd1;
      if (ch_go) begin                     // a beat's hi half: the beat is complete
        ch_pri <= ch_s + 2'd1;
        if (hp_rl[ch_s]) begin
          r_k[ch_s] <= '0; fl_hd[ch_s] <= fl_hd[ch_s] + (LBB + 1)'(1);
          bf_free[fl_q[ch_s][fl_hd[ch_s][LBB-1:0]]] <= 1'b1;
        end else r_k[ch_s] <= r_k[ch_s] + LN'(1);
      end
      if (in_go && !hs && !hn) bf_free[nbf] <= 1'b0;
      // delivery; freeing slots
      if (r1l_v) rob_vl[sll_q] <= 1'b1;
      if (r1h_v) rob_vh[slh_q] <= 1'b1;
      d1_v <= d_go;
      d1_w <= rq_rd[0][LR+3];
      d1_q <= rq_rd[0][LR+2:LR];
      if (d_go) s_rd <= s_rd + (LQ + 1)'(1);
      if (a_free) begin rob_vl[a_hd[LR-1:0]] <= 1'b0; rob_vh[a_hd[LR-1:0]] <= 1'b0; a_hd <= a_hd + (LR + 1)'(1); end
    end
  assign mr_rvalid     = d1_v;
  assign mr_rdata      = d1_w ? rbl_q[127:0] : d1_q[2] ? rbh_q[128 * d1_q[1:0] +: 128] : rbl_q[128 * d1_q[1:0] +: 128];
  assign mr_rhi        = {rbh_q, rbl_q[511:128]};

  // the slot on AW waits while a read the core issued before it overlaps it in the input register
  // (not younger: !in_old), gathered or on its way to a port's AR queue. This only clears: a read
  // of its units taken after it waits for it (in_haz) and does not move on.
  assign aw_s = w_aw[LNW-1:0];
  always_comb begin
    w_war = (in_v && !in_old[aw_s] && ovq(in_h16, in_nq, wb_h[aw_s], wb_m[aw_s])) ||
            (ar_v && ovq({ar_b, 3'b000}, 8'(ar_n) << 3, wb_h[aw_s], wb_m[aw_s]));
    for (int i = 0; i < NS; i++)
      if (g_v[i] && ovq({g_b[i], 3'b000}, 8'(g_n[i]) << 3, wb_h[aw_s], wb_m[aw_s])) w_war = 1'b1;
  end
  assign aw_go         = w_aw != w_tl && !(wo_v && aw_s == w_op) && !w_war;
  assign mw_ready      = !core_rst && (LNW + 1)'(w_tl - w_hd) != (LNW + 1)'(NW);
  assign w_take        = mw_req && mw_ready;
  assign w_beat        = mw_addr[LW+4:5];
  assign w_d32         = mw_addr[4] ? {mw_data, 128'b0} : {128'b0, mw_data};
  assign w_s32         = mw_addr[4] ? {mw_strb, 16'b0} : {16'b0, mw_strb};
  assign w_h16         = mw_addr[LW+4:4];
  assign w_blk         = in_v && LH'(w_h16 - in_h16) < LH'(in_nq);
  assign j_same        = wo_v && !w_blk && w_beat == wb_b[w_op] + LW'(wb_n[w_op] - LNM'(1));
  assign j_next        = wo_v && !w_blk && w_beat == wb_b[w_op] + LW'(wb_n[w_op]) && wb_n[w_op] != LNM'(MW) &&
                         w_beat[BBW-6:0] != '0;
  // the open slot closes without a write: idle GT cycles, or a read in the input register waits for it
  assign w_cls         = wo_v && !(w_take && (j_same || j_next)) &&
                         (w_take || 32'(wo_age) >= GT - 1 || (in_v && in_old[w_op] &&
                          ovq(in_h16, in_nq, wb_h[w_op], wb_m[w_op])));
  assign m_axi_awvalid = aw_go;
  assign m_axi_awaddr  = xlate(wb_a[aw_s], pt_rd[1], 5);
  assign m_axi_awlen   = 8'(wb_n[aw_s] - LNM'(1));
  assign m_axi_bready  = 1'b1;

  // beats: the open slot's last beat goes into the BRAM when the next one starts or the slot
  // closes; W reads them back (registered) into a small FIFO, slots in order once their AW is out
  logic                 wd_we, wd_rv, wd_rl;
  logic [LNW+LMW-1:0]   wd_wa, wd_ra;
  logic [287:0]         wd_q;
  logic [LMW-1:0]       w_rk;
  logic [255:0]         wf_d [4];
  logic [31:0]          wf_s [4];
  logic                 wf_l [4];
  logic [2:0]           wf_n;
  logic [1:0]           wf_h, wf_t;
  logic                 w_rd;
  assign wd_we = (w_take && wo_v && !j_same) || (w_cls && !w_take);
  assign wd_wa = {w_op, LMW'(wb_n[w_op] - LNM'(1))};
  assign w_rd  = w_rp != w_aw && 32'(wf_n) + 32'(wd_rv) < 3;
  assign wd_ra = {w_rp[LNW-1:0], w_rk};
  tm_bram #(.W(288), .D(NW * MW)) u_wd (.clk, .we(wd_we), .wa(wd_wa), .wd({lb_s, lb_d}), .ra(wd_ra), .rd(wd_q));
  assign m_axi_wvalid  = wf_n != 0;
  assign m_axi_wdata   = wf_d[wf_h];
  assign m_axi_wstrb   = wf_s[wf_h];
  assign m_axi_wlast   = wf_l[wf_h];

  always_comb begin
    ws_nv = ws_v;
    if (w_take && !(j_same || j_next)) ws_nv[w_tl[LNW-1:0]] = 1'b1;
    if (m_axi_bvalid) ws_nv[w_hd[LNW-1:0]] = 1'b0;
  end
  always_ff @(posedge clk) begin
    if (!rstn) begin
      ws_v <= '0; w_hd <= '0; w_aw <= '0; w_rp <= '0; w_tl <= '0; wo_v <= 1'b0; w_rk <= '0;
      wd_rv <= 1'b0; wf_n <= '0; wf_h <= '0; wf_t <= '0; rd_out <= '0;
    end else begin
      ws_v <= ws_nv;
      if (wo_v && !(w_take && (j_same || j_next)) && wo_age != '1) wo_age <= wo_age + 3'd1;
      if (w_take) begin
        if (j_same) begin
          for (int b = 0; b < 32; b++) if (w_s32[b]) lb_d[8 * b +: 8] <= w_d32[8 * b +: 8];
          lb_s <= lb_s | w_s32; wo_age <= '0;
          // the unit range grows: up (the next unit), or down (the lower unit of the slot's first beat)
          if (w_h16 + LH'(1) == wb_h[w_op]) begin wb_h[w_op] <= w_h16; wb_m[w_op] <= wb_m[w_op] + 8'd1; end
          else if (LH'(w_h16 - wb_h[w_op]) >= LH'(wb_m[w_op])) wb_m[w_op] <= 8'(w_h16 - wb_h[w_op]) + 8'd1;
        end else if (j_next) begin
          lb_d <= w_d32; lb_s <= w_s32; wb_n[w_op] <= wb_n[w_op] + LNM'(1); wo_age <= '0;
          wb_m[w_op] <= 8'(w_h16 - wb_h[w_op]) + 8'd1;
        end else begin                    // a new slot (the open one, if any, closes: its last beat is written)
          wb_b[w_tl[LNW-1:0]] <= w_beat; wb_a[w_tl[LNW-1:0]] <= mw_addr; wb_n[w_tl[LNW-1:0]] <= LNM'(1);
          wb_h[w_tl[LNW-1:0]] <= w_h16; wb_m[w_tl[LNW-1:0]] <= 8'd1;
          lb_d <= w_d32; lb_s <= w_s32; wo_v <= 1'b1; w_op <= w_tl[LNW-1:0]; wo_age <= '0;
          w_tl <= w_tl + (LNW + 1)'(1);
        end
      end else if (w_cls) wo_v <= 1'b0;
      if (m_axi_awvalid && m_axi_awready) w_aw <= w_aw + (LNW + 1)'(1);
      // W: read the beats of slot w_rp (its AW is out), one a cycle while the FIFO has room
      wd_rv <= w_rd;
      wd_rl <= w_rd && 32'(w_rk) + 1 == 32'(wb_n[w_rp[LNW-1:0]]);
      if (w_rd) begin
        if (32'(w_rk) + 1 == 32'(wb_n[w_rp[LNW-1:0]])) begin w_rk <= '0; w_rp <= w_rp + (LNW + 1)'(1); end
        else w_rk <= w_rk + LMW'(1);
      end
      if (wd_rv) begin wf_d[wf_t] <= wd_q[255:0]; wf_s[wf_t] <= wd_q[287:256]; wf_l[wf_t] <= wd_rl; wf_t <= wf_t + 2'd1; end
      if (m_axi_wvalid && m_axi_wready) wf_h <= wf_h + 2'd1;
      wf_n <= wf_n + 3'(wd_rv) - 3'(m_axi_wvalid && m_axi_wready);
      if (m_axi_bvalid) w_hd <= w_hd + (LNW + 1)'(1);
      rd_out <= rd_out + (ar_acc ? 16'(ar_l) + 16'd1 : 16'd0) - 16'(ch_go);
    end
  end
  always_comb begin
    br_idle = w_hd == w_tl && !wo_v && wf_n == 0 && !wd_rv && !in_v && !ar_v && rd_out == 0 && s_wr == s_rd && !r1l_v && !r1h_v && !d1_v;
    for (int i = 0; i < NS; i++) if (g_v[i]) br_idle = 1'b0;
  end

  // performance counters (since the last core reset)
  for (genvar i = 0; i < 10; i++) begin : g_profile
    if (PROFILE) begin : g_enabled
      always_ff @(posedge clk) begin
        if (core_rst) begin
          for (int j = 0; j < 4; j++) profile_count[i][j] <= '0;
        end else if (profile_class == 4'(i)) begin
          profile_count[i][0] <= profile_count[i][0] + 32'd1;
          profile_count[i][1] <= profile_count[i][1] + 32'(mr_req && !mr_ready);
          profile_count[i][2] <= profile_count[i][2] + 32'(mw_req && !mw_ready);
          profile_count[i][3] <= profile_count[i][3] + 32'(in_v && in_haz);
        end
      end
    end else begin : g_disabled
      for (genvar j = 0; j < 4; j++) assign profile_count[i][j] = 32'd0;
    end
  end
  always_ff @(posedge clk) begin
    if (core_rst) begin
      rd_beats <= '0; wr_beats <= '0; p_bursts <= '0; p_haz <= '0; p_arw <= '0; p_ww <= '0;
      p_rw <= '0; p_lat <= '0;
    end else begin
      rd_beats <= rd_beats + 32'(ch_go);
      wr_beats <= wr_beats + (m_axi_bvalid ? 32'(wb_n[w_hd[LNW-1:0]]) : 32'd0);
      p_bursts <= p_bursts + 32'(ar_acc);
      p_haz    <= p_haz + 32'(in_v && in_haz);
      p_arw    <= p_arw + 32'(ar_v && !ar_acc);
      p_ww     <= p_ww + 32'(mw_req && !mw_ready);
      p_rw     <= p_rw + 32'(mr_req && !mr_ready);
      p_lat    <= p_lat + 48'(rd_out);
    end
  end

  assign m_axi_awsize  = 3'd5;                 // 32 bytes
  assign m_axi_awburst = 2'b01;
  assign m_axi_awcache = 4'b0011;              // normal, non-cacheable, bufferable (the host flushes)
  assign m_axi_awprot  = 3'b010;               // non-secure data
  assign m_axi_awid    = 1'b0;
  assign m_axi_awlock  = 1'b0;
  assign m_axi_awqos   = 4'd0;
endmodule

// One HP read port: bursts of 64-byte beats from the core's clock (an AR queue) go out on a
// 128-bit AXI4 port at hclk as bursts of four times the 16-byte beats; the returned beats are
// packed four by four into 64-byte beats (with the burst's last flag and any error response) and
// come back across the clocks in order. rready holds while the return queue is full, so nothing
// is ever dropped.
// One HP read port: bursts of 128-byte beats from the core's clock (an AR queue) go out on a
// 128-bit AXI4 port at hclk as bursts of eight times the 16-byte beats; the returned beats are
// packed four by four into 64-byte halves (r_hi: the beat's upper half; r_last on the burst's
// last half; any error response) and come back across the clocks in order. rready holds while
// the return queue is full, so nothing is ever dropped.
module tm_hpr #(
  parameter int PA = 40
) (
  input  logic           clk,
  input  logic           rst,
  input  logic           ar_v,
  output logic           ar_rdy,
  input  logic [PA-1:0]  ar_a,                 // 128-byte aligned
  input  logic [2:0]     ar_n1,                // 128-byte beats - 1
  output logic           r_v,
  input  logic           r_rdy,
  output logic [511:0]   r_d,
  output logic           r_hi,
  output logic           r_last,
  output logic           r_err,
  input  logic           hclk,
  input  logic           hrst,
  output logic [PA-1:0]  m_araddr,
  output logic [7:0]     m_arlen,
  output logic [2:0]     m_arsize,
  output logic [1:0]     m_arburst,
  output logic [3:0]     m_arcache,
  output logic [2:0]     m_arprot,
  output logic [0:0]     m_arid,
  output logic           m_arlock,
  output logic [3:0]     m_arqos,
  output logic           m_arvalid,
  input  logic           m_arready,
  input  logic [127:0]   m_rdata,
  input  logic [1:0]     m_rresp,
  input  logic           m_rlast,
  input  logic           m_rvalid,
  output logic           m_rready
);
  logic          af_full, rf_full;
  logic [PA+2:0] af_q;
  logic [514:0]  rf_q;
  logic [1:0]    k;                            // 16-byte beat within the 64-byte half
  logic          h;                            // the half being packed
  logic [383:0]  pk;
  logic          pe;                           // an error response among them
  logic          push;
  tm_afifo #(.W(PA + 3), .D(16), .BRAM(1)) u_ar (.wclk(clk), .wrst(rst), .we(ar_v), .wd({ar_n1, ar_a}),
    .full(af_full), .rclk(hclk), .rrst(hrst), .rv(m_arvalid), .rd(af_q), .rr(m_arready));
  assign ar_rdy    = !af_full;
  assign m_araddr  = af_q[PA-1:0];
  assign m_arlen   = {2'b00, af_q[PA+2:PA], 3'b111};
  assign m_arsize  = 3'd4;                     // 16 bytes
  assign m_arburst = 2'b01;
  assign m_arcache = 4'b0011;                  // normal, non-cacheable, bufferable (the host flushes)
  assign m_arprot  = 3'b010;                   // non-secure data
  assign m_arid    = 1'b0;
  assign m_arlock  = 1'b0;
  assign m_arqos   = 4'd0;
  assign m_rready  = !rf_full;
  assign push      = m_rvalid && m_rready && k == 2'd3;
  always_ff @(posedge hclk)
    if (hrst) begin k <= '0; h <= 1'b0; pe <= 1'b0; end
    else if (m_rvalid && m_rready) begin
      k <= k + 2'd1;
      if (k != 2'd3) pk[128 * k +: 128] <= m_rdata;
      pe <= (k == 2'd3) ? 1'b0 : pe || m_rresp[1];
      if (k == 2'd3) h <= !h;
    end
  tm_afifo #(.W(515), .D(64), .BRAM(1)) u_r (.wclk(hclk), .wrst(hrst), .we(push),
    .wd({pe || m_rresp[1], m_rlast, h, m_rdata, pk}), .full(rf_full), .rclk(clk), .rrst(rst), .rv(r_v), .rd(rf_q),
    .rr(r_rdy));
  assign r_d    = rf_q[511:0];
  assign r_hi   = rf_q[512];
  assign r_last = rf_q[513];
  assign r_err  = rf_q[514];
endmodule
