// Testbench for tg_top: per port an AXI slave with random AR/AW/W ready, a latency queue and random
// R gaps; checks every burst's address against the expected sequence, the 4 KB rule, that valids
// hold until taken, and the counters at the end. Runs reads on ports 0/2 and writes on 1/3 at once.
`timescale 1ns/1ps
module tb_tg;
  localparam NM = 4;
  logic clk = 0, rstn = 0;
  always #1 clk = ~clk;

  logic [13:0] awaddr, araddr; logic awvalid, wvalid, arvalid, bready, rready;
  logic [31:0] wdata; wire awready, wready, bvalid, arready, rvalid; wire [1:0] bresp, rresp;
  wire [31:0] rdata;

  wire [39:0] m_araddr[NM], m_awaddr[NM]; wire [7:0] m_arlen[NM], m_awlen[NM];
  wire m_arvalid[NM], m_awvalid[NM], m_wvalid[NM], m_wlast[NM], m_rready[NM], m_bready[NM];
  logic m_arready[NM], m_awready[NM], m_wready[NM], m_rvalid[NM], m_rlast[NM], m_bvalid[NM];

  tg_top dut (.clk, .rstn,
    .s_axil_awaddr(awaddr), .s_axil_awprot(3'd0), .s_axil_awvalid(awvalid), .s_axil_awready(awready),
    .s_axil_wdata(wdata), .s_axil_wstrb(4'hF), .s_axil_wvalid(wvalid), .s_axil_wready(wready),
    .s_axil_bresp(bresp), .s_axil_bvalid(bvalid), .s_axil_bready(bready),
    .s_axil_araddr(araddr), .s_axil_arprot(3'd0), .s_axil_arvalid(arvalid), .s_axil_arready(arready),
    .s_axil_rdata(rdata), .s_axil_rresp(rresp), .s_axil_rvalid(rvalid), .s_axil_rready(rready),
`define PORT(i) \
    .m``i``_axi_araddr(m_araddr[i]), .m``i``_axi_arlen(m_arlen[i]), .m``i``_axi_arsize(), .m``i``_axi_arburst(), \
    .m``i``_axi_arcache(), .m``i``_axi_arprot(), .m``i``_axi_arid(), .m``i``_axi_arlock(), .m``i``_axi_arqos(), \
    .m``i``_axi_arvalid(m_arvalid[i]), .m``i``_axi_arready(m_arready[i]), .m``i``_axi_rdata(128'd0), \
    .m``i``_axi_rresp(2'd0), .m``i``_axi_rlast(m_rlast[i]), .m``i``_axi_rid(6'd0), .m``i``_axi_rvalid(m_rvalid[i]), \
    .m``i``_axi_rready(m_rready[i]), .m``i``_axi_awaddr(m_awaddr[i]), .m``i``_axi_awlen(m_awlen[i]), \
    .m``i``_axi_awsize(), .m``i``_axi_awburst(), .m``i``_axi_awcache(), .m``i``_axi_awprot(), .m``i``_axi_awid(), \
    .m``i``_axi_awlock(), .m``i``_axi_awqos(), .m``i``_axi_awvalid(m_awvalid[i]), .m``i``_axi_awready(m_awready[i]), \
    .m``i``_axi_wdata(), .m``i``_axi_wstrb(), .m``i``_axi_wlast(m_wlast[i]), .m``i``_axi_wvalid(m_wvalid[i]), \
    .m``i``_axi_wready(m_wready[i]), .m``i``_axi_bresp(2'd0), .m``i``_axi_bid(6'd0), .m``i``_axi_bvalid(m_bvalid[i]), \
    .m``i``_axi_bready(m_bready[i])
    `PORT(0), `PORT(1), `PORT(2), `PORT(3));

  int errors = 0;
  // ---- slave models ----
  typedef struct { int t; int n; } rq_t;
  rq_t rq[NM][$];                // read bursts: ready time, beats
  int  wq[NM][$];                // write bursts accepted (beats)
  int  bq[NM][$];                // B due times
  int  rbeat[NM];                // beats left of the current R burst
  longint exp_off[NM], ar_cnt[NM], aw_cnt[NM];
  logic [39:0] base[NM]; int msk[NM], ln[NM], ot[NM], nb[NM], md[NM];
  int now = 0;
  logic pv_ar[NM], pv_aw[NM]; logic [39:0] pa_ar[NM], pa_aw[NM];

  always @(posedge clk) begin
    now <= now + 1;
    for (int i = 0; i < NM; i++) begin
      // valid must hold, with the same address, until taken
      if (pv_ar[i] && !(m_arvalid[i] && m_araddr[i] == pa_ar[i])) begin errors++; $display("ERR AR dropped %0d", i); end
      if (pv_aw[i] && !(m_awvalid[i] && m_awaddr[i] == pa_aw[i])) begin errors++; $display("ERR AW dropped %0d", i); end
      pv_ar[i] <= m_arvalid[i] && !m_arready[i]; pa_ar[i] <= m_araddr[i];
      pv_aw[i] <= m_awvalid[i] && !m_awready[i]; pa_aw[i] <= m_awaddr[i];
      if (m_arvalid[i] && m_arready[i]) begin
        automatic longint bytes = (m_arlen[i] + 1) * 16;
        if (m_araddr[i] != base[i] + (exp_off[i] & msk[i])) begin errors++; $display("ERR AR addr %0d %h", i, m_araddr[i]); end
        if ((m_araddr[i] & 12'hFFF) + bytes > 4096) begin errors++; $display("ERR 4K %0d", i); end
        exp_off[i] += bytes; ar_cnt[i]++;
        rq[i].push_back('{now + 20 + $urandom_range(0, 60), m_arlen[i] + 1});
      end
      if (m_awvalid[i] && m_awready[i]) begin
        automatic longint bytes = (m_awlen[i] + 1) * 16;
        if (m_awaddr[i] != base[i] + (exp_off[i] & msk[i])) begin errors++; $display("ERR AW addr %0d %h", i, m_awaddr[i]); end
        exp_off[i] += bytes; aw_cnt[i]++;
        wq[i].push_back(m_awlen[i] + 1);
      end
    end
  end
  // R: bursts in order once due, beats with random gaps (rready is always 1 in the master)
  always @(posedge clk) begin
    for (int i = 0; i < NM; i++) begin
      if (m_rvalid[i] && m_rready[i]) begin
        rbeat[i]--;
        if (rbeat[i] == 0) void'(rq[i].pop_front());
      end
      if (rbeat[i] == 0 && rq[i].size() > 0 && rq[i][0].t <= now) rbeat[i] = rq[i][0].n;
      if (rbeat[i] > 0 && $urandom_range(0, 3) != 0) begin m_rvalid[i] <= 1; m_rlast[i] <= rbeat[i] == 1; end
      else begin m_rvalid[i] <= 0; m_rlast[i] <= 0; end
    end
  end
  // AR/AW/W ready, W counting, B
  int wleft[NM];
  always @(posedge clk) begin
    for (int i = 0; i < NM; i++) begin
      m_arready[i] <= $urandom_range(0, 2) != 0;
      m_awready[i] <= $urandom_range(0, 2) != 0;
      m_wready[i]  <= $urandom_range(0, 3) != 0;
      if (m_wvalid[i] && m_wready[i]) begin
        if (wleft[i] == 0) begin
          if (wq[i].size() == 0) begin errors++; $display("ERR W before AW %0d", i); end
          else wleft[i] = wq[i].pop_front();
        end
        wleft[i]--;
        if (m_wlast[i] != (wleft[i] == 0)) begin errors++; $display("ERR WLAST %0d", i); end
        if (wleft[i] == 0) bq[i].push_back(now + 10 + $urandom_range(0, 30));
      end
      if (m_bvalid[i] && m_bready[i]) void'(bq[i].pop_front());
      m_bvalid[i] <= 0;
      if (bq[i].size() > (m_bvalid[i] ? 1 : 0) && bq[i][m_bvalid[i] ? 1 : 0] <= now && $urandom_range(0, 1)) m_bvalid[i] <= 1;
    end
  end

  task automatic lwr(input int a, input int d);
    @(negedge clk); awaddr = a; wdata = d; awvalid = 1; wvalid = 1;
    do @(posedge clk); while (!awready);
    @(negedge clk); awvalid = 0; wvalid = 0; bready = 1;
    while (!bvalid) @(negedge clk);
    @(negedge clk); bready = 0;
  endtask
  task automatic lrd(input int a, output int d);
    @(negedge clk); araddr = a; arvalid = 1; rready = 1;
    do @(posedge clk); while (!arready);
    @(negedge clk); arvalid = 0;
    while (!rvalid) @(negedge clk);
    d = rdata; @(negedge clk); rready = 0;
  endtask

  task automatic run(input int lens[NM], input int ots[NM], input int nbs[NM], input int mds[NM]);
    int v;
    for (int i = 0; i < NM; i++) begin
      base[i] = 40'h8_0000_0000 + 40'(i) * 40'h20_0000; msk[i] = 32'h1F_FFFF;
      ln[i] = lens[i]; ot[i] = ots[i]; nb[i] = nbs[i]; md[i] = mds[i];
      exp_off[i] = 0; ar_cnt[i] = 0; aw_cnt[i] = 0;
      lwr(32'h100 * (i + 1) + 'h00, base[i][31:0]); lwr(32'h100 * (i + 1) + 'h04, 32'(base[i][39:32]));
      lwr(32'h100 * (i + 1) + 'h08, msk[i]); lwr(32'h100 * (i + 1) + 'h0C, ln[i]);
      lwr(32'h100 * (i + 1) + 'h10, ot[i]); lwr(32'h100 * (i + 1) + 'h14, nb[i]); lwr(32'h100 * (i + 1) + 'h18, md[i]);
    end
    lwr('h004, 32'hF1);
    do lrd('h008, v); while (v != 0);
    for (int i = 0; i < NM; i++) begin
      int cyc, beats, ls, lmin, lmax, e;
      lrd(32'h100 * (i + 1) + 'h20, cyc); lrd(32'h100 * (i + 1) + 'h24, beats); lrd(32'h100 * (i + 1) + 'h28, ls);
      lrd(32'h100 * (i + 1) + 'h30, lmin); lrd(32'h100 * (i + 1) + 'h34, lmax); lrd(32'h100 * (i + 1) + 'h38, e);
      $display("m%0d mode %0d len %0d ot %0d: cyc %0d beats %0d lat avg %0d min %0d max %0d errs %0d",
               i, md[i], ln[i] + 1, ot[i], cyc, beats, nb[i] ? ls / nb[i] : 0, lmin, lmax, e);
      if (beats != nb[i] * (ln[i] + 1)) begin errors++; $display("ERR beats %0d", i); end
      if ((md[i] ? aw_cnt[i] : ar_cnt[i]) != nb[i]) begin errors++; $display("ERR bursts %0d", i); end
      if (e != 0) errors++;
    end
  endtask

  initial begin
    int v;
    awvalid = 0; wvalid = 0; arvalid = 0; bready = 0; rready = 0;
    for (int i = 0; i < NM; i++) begin pv_ar[i] = 0; pv_aw[i] = 0; rbeat[i] = 0; wleft[i] = 0; end
    repeat (5) @(posedge clk); rstn = 1;
    lrd('h000, v); if (v != 32'h4E454754) begin errors++; $display("ERR ID %h", v); end
    run('{15, 3, 0, 255}, '{8, 4, 1, 64}, '{300, 200, 50, 40}, '{0, 1, 0, 1});
    run('{63, 0, 255, 7}, '{64, 64, 2, 16}, '{200, 1000, 30, 500}, '{1, 0, 0, 1});
    run('{1, 1, 1, 1}, '{3, 3, 3, 3}, '{0, 5, 70000, 3}, '{0, 0, 0, 0});   // 0 bursts; long run with wrap
    if (errors == 0) $display("TB PASS"); else $display("TB FAIL %0d", errors);
    $finish;
  end
  initial begin #20000000; $display("TB TIMEOUT"); $finish; end
endmodule
