// Testbench for tm_core: a DDR model of MEM_BYTES with random backpressure on both ports and a
// random read latency. Loads +img (binary: each 16-byte word with its bytes reversed), runs the queue at
// +qaddr / +qlen, writes the memory to +out and prints the end status.
module tb_core #(
  parameter int R = 8,
  parameter int C = 8,
  parameter int NB = 8,
  parameter int OT = 128,
  parameter int LG = 4,
  parameter int HB = 1024,
  parameter int VG = 4,
  parameter int OQ = 128,
  parameter int VQMIN = 32,
  parameter longint MEM_BYTES = 64'h40_0000
);
  localparam int WORDS = int'(MEM_BYTES / 16);

  logic clk = 1'b0, rst = 1'b1, start = 1'b0;
  always #5 clk = ~clk;

  logic [127:0] mem [WORDS];
  logic [63:0]  q_addr;
  logic [31:0]  q_len;
  logic         halted, done;
  logic [3:0]   err_code;
  logic [3:0]   profile_class;
  logic [31:0]  op_idx;
  logic [63:0]  cycles;
  logic         mr_req, mr_ready, mr_rvalid, mw_req, mw_ready;
  logic [63:0]  mr_addr, mw_addr;
  logic [127:0] mr_rdata, mw_data;
  logic         mr_wide;                  // (RW 1: never)
  logic [127:0] mr_rhi = '0;
  logic [15:0]  mw_strb;

  tm_core #(.R(R), .C(C), .MEM_BYTES(MEM_BYTES), .NB(NB), .OT(OT), .HB(HB), .VG(VG), .LG(LG), .OQ(OQ), .VQMIN(VQMIN)) dut (.*);

  int unsigned stall = 0, lat = 2, seed = 1;
  longint unsigned maxcyc = 64'd50_000_000;

  // read return: any number of requests in flight, each answered 1..lat cycles after it was
  // accepted, in request order, at most one per cycle
  longint unsigned now = 0, last_due = 0;
  longint unsigned q_due [$];
  logic [63:0]     q_ad  [$];
  int unsigned     max_out = 0;

`ifdef DRAM
  // DRAM timing (sim/dram/tb_dram.cpp over DRAMsim3); the bytes stay here, as below
  import "DPI-C" function void dram_init(input string cfg, input int fabric, input int rcap,
                                         input int rbuf, input int wbuf, input int wflush);
  import "DPI-C" function int  dram_rd_can();
  import "DPI-C" function int  dram_wr_can();
  import "DPI-C" function void dram_rd(input longint addr);
  import "DPI-C" function void dram_wr(input longint addr, input int strb);
  import "DPI-C" function int  dram_pop();
  import "DPI-C" function void dram_tick();
  import "DPI-C" function void dram_report();
  logic [63:0] d_ad [$];
  int unsigned d_out = 0;

  always @(posedge clk) begin
    if (!rst) begin
      mr_rvalid <= 1'b0;
      if (dram_pop() != 0) begin
        mr_rvalid <= 1'b1;
        mr_rdata  <= mem[d_ad[0][63:4]];
        void'(d_ad.pop_front());
      end
      if (mr_req && mr_ready) begin
        dram_rd(mr_addr);
        d_ad.push_back(mr_addr);
        if (d_ad.size() > max_out) max_out = d_ad.size();
      end
      if (mw_req && mw_ready) begin
        dram_wr(mw_addr, int'(mw_strb));
        for (int b = 0; b < 16; b++)
          if (mw_strb[b]) mem[mw_addr[63:4]][8 * b +: 8] <= mw_data[8 * b +: 8];
      end
      dram_tick();
      mr_ready <= dram_rd_can() != 0;
      mw_ready <= dram_wr_can() != 0;
    end
  end
`else
  always @(posedge clk) begin
    now <= now + 1;
    mr_rvalid <= 1'b0;
    if (q_due.size() != 0 && q_due[0] <= now) begin
      mr_rvalid <= 1'b1;
      mr_rdata  <= mem[q_ad[0][63:4]];
      void'(q_due.pop_front());
      void'(q_ad.pop_front());
    end
    if (mr_req && mr_ready) begin
      longint unsigned due;
      due = now + 1 + ((lat > 1) ? $urandom_range(lat - 1, 0) : 0);
      if (q_due.size() != 0 && due <= last_due) due = last_due + 1;
      q_due.push_back(due);
      q_ad.push_back(mr_addr);
      last_due = due;
      if (q_due.size() > max_out) max_out = q_due.size();
    end
    if (mw_req && mw_ready)
      for (int b = 0; b < 16; b++)
        if (mw_strb[b]) mem[mw_addr[63:4]][8 * b +: 8] <= mw_data[8 * b +: 8];
    mr_ready <= ($urandom_range(99, 0) >= stall);
    mw_ready <= ($urandom_range(99, 0) >= stall);
  end
`endif

  initial begin
    string img, outp;
    longint unsigned cyc;
    void'($value$plusargs("stall=%d", stall));
    void'($value$plusargs("lat=%d", lat));
    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("maxcyc=%d", maxcyc));
    if (!$value$plusargs("img=%s", img)) $fatal(1, "+img= missing");
    if (!$value$plusargs("out=%s", outp)) $fatal(1, "+out= missing");
    if (!$value$plusargs("qaddr=%h", q_addr)) $fatal(1, "+qaddr= missing");
    if (!$value$plusargs("qlen=%h", q_len)) $fatal(1, "+qlen= missing");
    void'($urandom(seed));
    begin                                 // binary image: 16-byte words, each word's bytes MSB first
      int fd, nr;
      fd = $fopen(img, "rb");
      if (fd == 0) $fatal(1, "cannot open +img");
      nr = $fread(mem, fd);
      $fclose(fd);
    end
`ifdef DRAM
    begin
      string cfg;
      int fabric = 40, rcap = 64, rbuf = 4, wbuf = 8, wflush = 64;
      if (!$value$plusargs("dramcfg=%s", cfg)) $fatal(1, "+dramcfg= missing");
      void'($value$plusargs("fabric=%d", fabric));
      void'($value$plusargs("rcap=%d", rcap));
      void'($value$plusargs("rbuf=%d", rbuf));
      void'($value$plusargs("wbuf=%d", wbuf));
      void'($value$plusargs("wflush=%d", wflush));
      dram_init(cfg, fabric, rcap, rbuf, wbuf, wflush);
    end
`endif
    mr_ready = 1'b0;
    mw_ready = 1'b0;
    repeat (4) @(posedge clk);
    rst <= 1'b0;
    @(posedge clk);
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;
    cyc = 0;
    while (!halted && cyc < maxcyc) begin
      @(posedge clk);
      cyc++;
      if (cyc % 64'd10000000 == 0) begin $display("PROGRESS %0d", cyc); $fflush(); end
    end
    @(posedge clk);
    begin                                 // binary dump: %u writes a word least significant byte first = address order
      int fd;
      fd = $fopen(outp, "wb");
      for (int i = 0; i < WORDS; i++) $fwrite(fd, "%u", mem[i]);
      $fclose(fd);
    end
    for (int i = 0; i < dut.NPH; i++) $display("PERF phase%0d %0d", i, dut.perf_phase[i]);
    for (int i = 0; i < dut.NPH; i++) $display("PERF rdph%0d %0d", i, dut.perf_rd_ph[i]);
    $display("PERF rd_beats %0d", dut.perf_rd_beats);
    $display("PERF wr_beats %0d", dut.perf_wr_beats);
    $display("PERF rd_accept_wait %0d", dut.perf_rd_accept_wait);
    $display("PERF rd_latency %0d", dut.perf_rd_latency);
    $display("PERF wr_accept_wait %0d", dut.perf_wr_accept_wait);
    $display("PERF feed_valid %0d", dut.perf_feed_valid);
    $display("PERF max_reads_in_flight %0d", max_out);
`ifdef DRAM
    dram_report();
`endif
    $display("STATUS done=%0d err=%0d op=%0d cycles=%0d%s", done, err_code, op_idx, cycles,
             halted ? "" : " TIMEOUT");
    $finish;
  end
endmodule
