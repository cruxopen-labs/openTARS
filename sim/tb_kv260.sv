// Testbench for tm_kv260_top: the board's view of the core. An AXI4 slave memory: four 128-bit
// read ports on their own clock (hclk, 2.5 times the core's, as on the board: HP0, HP3, HP2, HP1), one
// 256-bit write port on the core's clock (random backpressure on every channel, random read and
// write-response latency; a write lands in memory only when its response is given, a read takes
// the memory as it is when its beat is first presented) and an
// AXI-Lite master that does what the board software does: fill the page table (here a scrambled
// mapping of small pages into a high physical window), reset, start, poll STATUS until the core
// has halted and the bridge is idle. Same plusargs and output as tb_core (+img in the core's
// address order, +out likewise), so tars_mini.rtlsim runs the core tests through it (TM_TOP=kv260).
module tb_kv260 #(
  parameter int R = 8,
  parameter int C = 8,
  parameter int NB = 8,
  parameter int OT = 128,
  parameter int LG = 4,
  parameter int HB = 1024,
  parameter int VG = 4,
  parameter int OQ = 128,
  parameter int VQMIN = 32,
  parameter longint MEM_BYTES = 64'h40_0000,
  parameter int PAGE_BITS = 12
);
  localparam int     WORDS = int'(MEM_BYTES / 16);
  localparam int     NP    = int'(MEM_BYTES >> PAGE_BITS);    // pages
  localparam int     NPT   = (NP < 2) ? 2 : NP;
  localparam longint PBASE = 64'h8_0000_0000;                 // the physical window

  logic clk = 1'b0, rstn = 1'b0, hclk = 1'b0;
  always #5 clk = ~clk;
  always #2 hclk = ~hclk;

  logic [127:0] pmem [WORDS];                                 // physical memory (the window)
  logic [127:0] cmem [WORDS];                                 // the image in the core's order (loading)
  int unsigned  stall = 0, wstall = 0, lat = 2, seed = 1;   // wstall: the write channel's alone
  int unsigned  nout = 0, blat = 0, artrace = 0;   // artrace: print 3000 AR bursts after this many   // board-like: at most nout read bursts in flight (0: any), blat more cycles of latency
  longint unsigned maxcyc = 64'd50_000_000, now = 0, hnow = 0;

  function automatic int perm(input int i);                   // core page -> physical page
    return (i * 5 + 3) % NP;
  endfunction
  function automatic longint widx(input logic [39:0] pa);
    return longint'((pa - 40'(PBASE)) >> 4);
  endfunction
  function automatic longint cidx(input longint w);           // core word -> physical word
    longint pw;
    pw = longint'(1) << (PAGE_BITS - 4);
    return longint'(perm(int'(w / pw))) * pw + (w % pw);
  endfunction

  // ---------------------------------------------------------------- the DUT
  logic [13:0]  awaddr, araddr;
  logic         awvalid, awready, wvalid, wready, bvalid, bready, arvalid, arready, rvalid, rready;
  logic [31:0]  wdata, rdata;
  logic [3:0]   wstrb;
  logic [1:0]   bresp, rresp;
  logic [39:0]  r_araddr [4], m_awaddr;
  logic [7:0]   r_arlen [4], m_awlen;
  logic [2:0]   r_arsize [4], m_awsize, r_arprot [4], m_awprot;
  logic [1:0]   r_arburst [4], m_awburst, m_bresp;
  logic [3:0]   r_arcache [4], m_awcache, r_arqos [4], m_awqos;
  logic [0:0]   r_arid [4], m_awid, m_bid;
  logic         r_arlock [4], r_arvalid [4], r_arready [4], r_rlast [4], r_rvalid [4], r_rready [4];
  logic         m_awlock, m_awvalid, m_awready, m_wlast, m_wvalid, m_wready, m_bvalid, m_bready;
  logic [127:0] r_rdata [4];
  logic [255:0] m_wdata;                     // 32-byte write beats: memory words 2 w, 2 w + 1
  logic [31:0]  m_wstrb;

  tm_kv260_top #(.R(R), .C(C), .VG(VG), .NB(NB), .OT(OT), .HB(HB), .OQ(OQ), .LG(LG), .VQMIN(VQMIN),
                 .MEM_BYTES(MEM_BYTES), .PAGE_BITS(PAGE_BITS), .NPT(NPT), .PA_BITS(40), .GB(4)) dut (
    .clk, .rstn, .hclk, .hrstn(rstn),
    .s_axil_awaddr(awaddr), .s_axil_awvalid(awvalid), .s_axil_awready(awready), .s_axil_wdata(wdata),
    .s_axil_wstrb(wstrb), .s_axil_wvalid(wvalid), .s_axil_wready(wready), .s_axil_bresp(bresp),
    .s_axil_bvalid(bvalid), .s_axil_bready(bready), .s_axil_araddr(araddr), .s_axil_arvalid(arvalid),
    .s_axil_arready(arready), .s_axil_rdata(rdata), .s_axil_rresp(rresp), .s_axil_rvalid(rvalid),
    .s_axil_rready(rready),
    .r0_axi_araddr(r_araddr[0]), .r0_axi_arlen(r_arlen[0]), .r0_axi_arsize(r_arsize[0]),
    .r0_axi_arburst(r_arburst[0]), .r0_axi_arcache(r_arcache[0]), .r0_axi_arprot(r_arprot[0]),
    .r0_axi_arid(r_arid[0]), .r0_axi_arlock(r_arlock[0]), .r0_axi_arqos(r_arqos[0]),
    .r0_axi_arvalid(r_arvalid[0]), .r0_axi_arready(r_arready[0]), .r0_axi_rdata(r_rdata[0]),
    .r0_axi_rresp(2'b00), .r0_axi_rlast(r_rlast[0]), .r0_axi_rid(1'b0), .r0_axi_rvalid(r_rvalid[0]),
    .r0_axi_rready(r_rready[0]),
    .r1_axi_araddr(r_araddr[1]), .r1_axi_arlen(r_arlen[1]), .r1_axi_arsize(r_arsize[1]),
    .r1_axi_arburst(r_arburst[1]), .r1_axi_arcache(r_arcache[1]), .r1_axi_arprot(r_arprot[1]),
    .r1_axi_arid(r_arid[1]), .r1_axi_arlock(r_arlock[1]), .r1_axi_arqos(r_arqos[1]),
    .r1_axi_arvalid(r_arvalid[1]), .r1_axi_arready(r_arready[1]), .r1_axi_rdata(r_rdata[1]),
    .r1_axi_rresp(2'b00), .r1_axi_rlast(r_rlast[1]), .r1_axi_rid(1'b0), .r1_axi_rvalid(r_rvalid[1]),
    .r1_axi_rready(r_rready[1]),
    .r2_axi_araddr(r_araddr[2]), .r2_axi_arlen(r_arlen[2]), .r2_axi_arsize(r_arsize[2]),
    .r2_axi_arburst(r_arburst[2]), .r2_axi_arcache(r_arcache[2]), .r2_axi_arprot(r_arprot[2]),
    .r2_axi_arid(r_arid[2]), .r2_axi_arlock(r_arlock[2]), .r2_axi_arqos(r_arqos[2]),
    .r2_axi_arvalid(r_arvalid[2]), .r2_axi_arready(r_arready[2]), .r2_axi_rdata(r_rdata[2]),
    .r2_axi_rresp(2'b00), .r2_axi_rlast(r_rlast[2]), .r2_axi_rid(1'b0), .r2_axi_rvalid(r_rvalid[2]),
    .r2_axi_rready(r_rready[2]),
    .r3_axi_araddr(r_araddr[3]), .r3_axi_arlen(r_arlen[3]), .r3_axi_arsize(r_arsize[3]),
    .r3_axi_arburst(r_arburst[3]), .r3_axi_arcache(r_arcache[3]), .r3_axi_arprot(r_arprot[3]),
    .r3_axi_arid(r_arid[3]), .r3_axi_arlock(r_arlock[3]), .r3_axi_arqos(r_arqos[3]),
    .r3_axi_arvalid(r_arvalid[3]), .r3_axi_arready(r_arready[3]), .r3_axi_rdata(r_rdata[3]),
    .r3_axi_rresp(2'b00), .r3_axi_rlast(r_rlast[3]), .r3_axi_rid(1'b0), .r3_axi_rvalid(r_rvalid[3]),
    .r3_axi_rready(r_rready[3]),
    .m_axi_awaddr(m_awaddr), .m_axi_awlen(m_awlen), .m_axi_awsize(m_awsize),
    .m_axi_awburst(m_awburst), .m_axi_awcache(m_awcache), .m_axi_awprot(m_awprot), .m_axi_awid(m_awid),
    .m_axi_awlock(m_awlock), .m_axi_awqos(m_awqos), .m_axi_awvalid(m_awvalid), .m_axi_awready(m_awready),
    .m_axi_wdata(m_wdata), .m_axi_wstrb(m_wstrb), .m_axi_wlast(m_wlast), .m_axi_wvalid(m_wvalid),
    .m_axi_wready(m_wready), .m_axi_bresp(m_bresp), .m_axi_bid(m_bid), .m_axi_bvalid(m_bvalid),
    .m_axi_bready(m_bready));

  // ---------------------------------------------------------------- AXI4 slave memory
  longint unsigned b_due [$], last_b = 0;
  longint unsigned r_due [4][$], last_r [4];
  logic [39:0]     r_ad [4][$], aw_q [$];
  bit              r_last [4][$];
  logic [255:0]    w_dq [$];
  logic [31:0]     w_sq [$];
  logic [39:0]     b_ad [$];
  int unsigned     bad_axi = 0, rd_bursts = 0, wb_out = 0, w_k = 0;
  int unsigned     rb_out [4];
  logic [7:0]      aw_l [$];
  int              b_n [$];
  int unsigned     max_rd = 0;                              // the core's reads in flight (16-byte requests or wide ones)
  always @(posedge clk) if (int'(9'(dut.s_wr - dut.s_rd)) + int'(dut.in_v) > max_rd) max_rd = int'(9'(dut.s_wr - dut.s_rd)) + int'(dut.in_v);
  // +optrace: one line per descriptor boundary (the core's index, cycle, read beats so far, the
  // core's wide and narrow read requests, cycles the core's read request waited)
  int unsigned optrace = 0;
  longint unsigned t_w = 0, t_n = 0, t_rw = 0;
  logic [31:0] t_qi = '1;
  always @(posedge clk) begin
    if (dut.mr_req && dut.mr_ready) begin if (dut.mr_wide) t_w++; else t_n++; end
    if (dut.mr_req && !dut.mr_ready) t_rw++;
    if (optrace != 0 && dut.u_core.qi != t_qi) begin
      $display("OPT %0d %0d %0d %0d %0d %0d", dut.u_core.qi, now, dut.rd_beats, t_w, t_n, t_rw);
      t_qi <= dut.u_core.qi;
    end
  end
  assign m_bresp = 2'b00;
  assign m_bid   = 1'b0;
  // the read ports (hclk): in order, a beat a cycle at most, held until taken; latencies in hclk
  // cycles (the plusargs are in core cycles: x 2.5)
  initial for (int p = 0; p < 4; p++) begin rb_out[p] = 0; last_r[p] = 0; end
  always @(posedge hclk) begin
    hnow <= hnow + 1;
    for (int p = 0; p < 4; p++) begin
      if (!rstn) begin
        r_rvalid[p] <= 1'b0; r_rlast[p] <= 1'b0; r_arready[p] <= 1'b0;
      end else begin
        if (r_rvalid[p] && r_rready[p]) begin
          if (r_last[p][0]) rb_out[p]--;
          void'(r_due[p].pop_front()); void'(r_ad[p].pop_front()); void'(r_last[p].pop_front());
        end
        if (!r_rvalid[p] || r_rready[p]) begin
          if (r_due[p].size() != 0 && r_due[p][0] <= hnow) begin   // (the taken beat is popped already)
            r_rvalid[p] <= 1'b1; r_rlast[p] <= r_last[p][0];
            r_rdata[p]  <= pmem[widx(r_ad[p][0])];
          end else begin r_rvalid[p] <= 1'b0; r_rlast[p] <= 1'b0; end
        end
        if (r_arvalid[p] && r_arready[p]) begin
          longint unsigned d;
          // a burst of whole 128-byte beats (INCR, 16-byte transfers), inside its port's 1 KB block
          if (r_arsize[p] != 3'd4 || r_arburst[p] != 2'b01 || r_araddr[p][6:0] != 0 || r_arlen[p][2:0] != 3'b111 ||
              r_araddr[p] < 40'(PBASE) || widx(r_araddr[p]) + longint'(r_arlen[p]) >= WORDS ||
              int'(r_araddr[p][9:4]) + int'(r_arlen[p]) > 63 || int'(r_araddr[p][11:10]) != p) begin bad_axi++; $display("BAD AR%0d %0d %h len %0d size %0d", p, now, r_araddr[p], r_arlen[p], r_arsize[p]); end
          rd_bursts++;
          if (artrace != 0 && rd_bursts > artrace && rd_bursts <= artrace + 2000000) $display("AR%0d %0d %h %0d", p, now, r_araddr[p], r_arlen[p]);
          rb_out[p]++;
          d = hnow + 2 + (5 * blat) / 2 + ((lat > 1) ? $urandom_range(10 * lat, 0) : 0);
          for (int k = 0; k <= int'(r_arlen[p]); k++) begin
            if (r_due[p].size() != 0 && d <= last_r[p]) d = last_r[p] + 1;
            if (k > 0 && lat > 1 && blat == 0 && $urandom_range(3, 0) == 0) d = d + 1;   // gaps inside a burst (board-like: none)
            r_due[p].push_back(d); r_ad[p].push_back(r_araddr[p] + 40'(16 * k)); r_last[p].push_back(k == int'(r_arlen[p]));
            last_r[p] = d;
          end
        end
        r_arready[p] <= ($urandom_range(99, 0) >= stall) && (nout == 0 || rb_out[p] < nout);
      end
    end
  end
  always @(posedge clk) begin
    now <= now + 1;
    m_bvalid <= 1'b0;
    if (rstn) begin
      // writes: bursts of awlen + 1 beats (INCR, inside one 4 KB block); the data follows its
      // address in order (wlast on the burst's last beat); a burst lands, all its beats, with its
      // response
      if (m_awvalid && m_awready) begin
        if (m_awsize != 3'd5 || m_awburst != 2'b01 || m_awaddr[4:0] != 0 || m_awaddr < 40'(PBASE) ||
            widx(m_awaddr) + 2 * longint'(m_awlen) + 1 >= WORDS || int'(m_awaddr[11:5]) + int'(m_awlen) > 127) begin bad_axi++; $display("BAD AW %0d %h len %0d", now, m_awaddr, m_awlen); end
        aw_q.push_back(m_awaddr); aw_l.push_back(m_awlen); wb_out++;
      end
      if (m_wvalid && m_wready) begin
        if (aw_l.size() == 0 || m_wlast != (w_k == int'(aw_l[0]))) begin bad_axi++; $display("BAD W %0d k %0d", now, w_k); end
        w_dq.push_back(m_wdata); w_sq.push_back(m_wstrb);
        if (m_wlast && aw_l.size() != 0) begin
          longint unsigned d;
          d = now + 2 + blat + ((lat > 1) ? $urandom_range(4 * lat, 0) : 0);
          if (b_due.size() != 0 && d <= last_b) d = last_b + 1;
          b_due.push_back(d); b_ad.push_back(aw_q[0]); b_n.push_back(int'(aw_l[0]) + 1);
          last_b = d;
          void'(aw_q.pop_front()); void'(aw_l.pop_front());
          w_k = 0;
        end else w_k++;
      end
      if (b_due.size() != 0 && b_due[0] <= now) begin
        for (int k = 0; k < b_n[0]; k++) begin
          for (int b = 0; b < 16; b++)
            if (w_sq[0][b]) pmem[widx(b_ad[0]) + 2 * k][8 * b +: 8] <= w_dq[0][8 * b +: 8];
          for (int b = 0; b < 16; b++)
            if (w_sq[0][16 + b]) pmem[widx(b_ad[0]) + 2 * k + 1][8 * b +: 8] <= w_dq[0][128 + 8 * b +: 8];
          void'(w_dq.pop_front()); void'(w_sq.pop_front());
        end
        m_bvalid <= 1'b1; wb_out--;
        void'(b_due.pop_front()); void'(b_ad.pop_front()); void'(b_n.pop_front());
      end
    end
    m_awready <= ($urandom_range(99, 0) >= stall) && (nout == 0 || wb_out < nout);   // wb_out: after this cycle's
    // wstall 100: the write channel shut for long stretches, so the core's output queue fills
    m_wready  <= (wstall >= 100) ? now[11:9] == 3'd0 : ($urandom_range(99, 0) >= ((wstall > stall) ? wstall : stall));
  end

  // ---------------------------------------------------------------- AXI-Lite master
  // driven and sampled at the falling edge (stable values): a handshake seen there completes at
  // the next rising edge
  task automatic lw(input logic [13:0] a, input logic [31:0] d);
    @(negedge clk);
    awaddr = a; wdata = d; wstrb = 4'hF; awvalid = 1'b1; wvalid = 1'b1; bready = 1'b1;
    do @(negedge clk); while (!awready);
    @(posedge clk);
    @(negedge clk);
    awvalid = 1'b0; wvalid = 1'b0;
    while (!bvalid) @(negedge clk);
    @(posedge clk);
    @(negedge clk);
    bready = 1'b0;
  endtask
  task automatic lr(input logic [13:0] a, output logic [31:0] d);
    @(negedge clk);
    araddr = a; arvalid = 1'b1; rready = 1'b1;
    while (!arready) @(negedge clk);
    @(posedge clk);
    @(negedge clk);
    arvalid = 1'b0;
    while (!rvalid) @(negedge clk);
    d = rdata;
    @(posedge clk);
    @(negedge clk);
    rready = 1'b0;
  endtask

  initial begin
    string img, outp;
    logic [31:0] st, opx, clo, chi, id, rdb, wrb;
    logic [63:0] q_addr;
    logic [31:0] q_len;
    longint unsigned cyc;
    awvalid = 1'b0; wvalid = 1'b0; bready = 1'b0; arvalid = 1'b0; rready = 1'b0;
    void'($value$plusargs("stall=%d", stall));
    void'($value$plusargs("wstall=%d", wstall));
    void'($value$plusargs("lat=%d", lat));
    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("nout=%d", nout));
    void'($value$plusargs("blat=%d", blat));
    void'($value$plusargs("artrace=%d", artrace));
    void'($value$plusargs("optrace=%d", optrace));
    void'($value$plusargs("maxcyc=%d", maxcyc));
    if (!$value$plusargs("img=%s", img)) $fatal(1, "+img= missing");
    if (!$value$plusargs("out=%s", outp)) $fatal(1, "+out= missing");
    if (!$value$plusargs("qaddr=%h", q_addr)) $fatal(1, "+qaddr= missing");
    if (!$value$plusargs("qlen=%h", q_len)) $fatal(1, "+qlen= missing");
    if (NP < 2 || NP % 5 == 0 || (longint'(NP) << PAGE_BITS) != MEM_BYTES)
      $fatal(1, "MEM_BYTES must be a whole number (>= 2, not a multiple of 5) of pages");
    void'($urandom(seed));
    begin                                 // the image in core order, scattered to the physical pages
      int fd, nr;
      fd = $fopen(img, "rb");
      if (fd == 0) $fatal(1, "cannot open +img");
      nr = $fread(cmem, fd);
      $fclose(fd);
      for (longint w = 0; w < WORDS; w++) pmem[cidx(w)] = cmem[w];
    end
    repeat (4) @(posedge clk);
    rstn <= 1'b1;
    repeat (2) @(posedge clk);
    lr(14'h000, id);
    if (id != 32'h5441_5253) $fatal(1, "bad ID %h", id);
    for (int i = 0; i < NP; i++)
      lw(14'h2000 + 14'(4 * i), 32'((PBASE >> PAGE_BITS) + longint'(perm(i))));
    lw(14'h014, q_addr[31:0]); lw(14'h018, q_addr[63:32]); lw(14'h01C, q_len);
    lw(14'h008, 32'd2);                   // reset the core
    lw(14'h008, 32'd0);
    lw(14'h008, 32'd1);                   // start
    cyc = 0;
    do begin
      repeat (16) @(posedge clk);
      cyc += 16;
      lr(14'h00C, st);
      if (cyc % 64'd10000000 < 32) begin $display("PROGRESS %0d", cyc); $fflush(); end
    end while (!(st[0] && st[8]) && cyc < maxcyc);
    lr(14'h010, opx); lr(14'h020, clo); lr(14'h024, chi); lr(14'h028, rdb); lr(14'h02C, wrb);
    begin
      int fd;
      fd = $fopen(outp, "wb");
      for (longint w = 0; w < WORDS; w++) $fwrite(fd, "%u", pmem[cidx(w)]);
      $fclose(fd);
    end
    $display("PERF rd_beats %0d", rdb);
    $display("PERF wr_beats %0d", wrb);
    $display("PERF bad_axi %0d", bad_axi);
    $display("PERF max_reads_in_flight %0d", max_rd);
    $display("PERF rd_bursts %0d", rd_bursts);
    if (bad_axi != 0 || st[9] || st[10]) $display("AXI ERRORS: %0d protocol, status %h", bad_axi, st);
    $display("STATUS done=%0d err=%0d op=%0d cycles=%0d%s", (bad_axi == 0) ? st[1] : 1'b0, st[7:4], opx,
             {chi, clo}, (st[0] && st[8]) ? "" : " TIMEOUT");
    $finish;
  end
endmodule
