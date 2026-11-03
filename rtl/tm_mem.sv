// A small memory with one write port and NR asynchronous read ports (FPGA: distributed RAM, one
// copy per read port; the read is combinational, like the register arrays it replaces).
// A write at the clock edge is seen by reads from the next cycle on; a read out of range reads
// entry 0 (the callers clamp their indices the same way).
module tm_lutram #(
  parameter int W  = 128,
  parameter int D  = 32,
  parameter int NR = 1
) (
  input  logic          clk,
  input  logic          we,
  input  logic [31:0]   wa,
  input  logic [W-1:0]  wd,
  input  logic [31:0]   ra [NR],
  output logic [W-1:0]  rd [NR]
);
  (* ram_style = "distributed" *) logic [W-1:0] m [D];
  always_ff @(posedge clk)
    if (we && wa < 32'(D)) m[wa] <= wd;              // out of range: no write (as an array's)
  for (genvar r = 0; r < NR; r++) begin : g_r
    assign rd[r] = m[(ra[r] < 32'(D)) ? ra[r] : 32'd0];
  end
endmodule

// A simple dual-port memory with a registered read (FPGA: block RAM): rd is the entry at ra as it
// was after the previous clock edge's write (a write and a read of the same entry in one cycle
// read the old data; the callers never do that).
module tm_bram #(
  parameter int W = 128,
  parameter int D = 512
) (
  input  logic                 clk,
  input  logic                 we,
  input  logic [$clog2(D)-1:0] wa,
  input  logic [W-1:0]         wd,
  input  logic [$clog2(D)-1:0] ra,
  output logic [W-1:0]         rd
);
  (* ram_style = "block" *) logic [W-1:0] m [D];
  always_ff @(posedge clk) begin
    if (we) m[wa] <= wd;
    rd <= m[ra];
  end
endmodule

// A FIFO across two clocks (gray-coded pointers, two-flop synchronizers; BRAM 1: block RAM,
// else distributed RAM). Write side: we only when !full. Read side, first word fall-through
// through an output register: rv says rd is valid, rr takes it. Each side has its own
// synchronous reset (both held at power-up). The gray pointers (afg_*) into the first
// synchronizer stage (afs1_*) are constrained with set_max_delay -datapath_only in the build.
module tm_afifo #(
  parameter int W    = 32,
  parameter int D    = 16,                 // a power of two
  parameter bit BRAM = 0
) (
  input  logic          wclk,
  input  logic          wrst,
  input  logic          we,
  input  logic [W-1:0]  wd,
  output logic          full,
  input  logic          rclk,
  input  logic          rrst,
  output logic          rv,
  output logic [W-1:0]  rd,
  input  logic          rr
);
  localparam int L = $clog2(D);
  function automatic logic [L:0] g(input logic [L:0] b);
    return b ^ (b >> 1);
  endfunction
  logic [L:0] wb, afg_w, rb, afg_r;
  (* ASYNC_REG = "TRUE" *) logic [L:0] afs1_r, afs2_r;   // the read pointer in the write clock
  (* ASYNC_REG = "TRUE" *) logic [L:0] afs1_w, afs2_w;   // the write pointer in the read clock
  logic empty, re;
  assign full  = afg_w == {~afs2_r[L:L-1], afs2_r[L-2:0]};
  assign empty = afg_r == afs2_w;
  assign re    = !empty && (!rv || rr);
  if (BRAM) begin : g_b
    (* ram_style = "block" *) logic [W-1:0] mb [D];
    always_ff @(posedge wclk) if (we && !full) mb[wb[L-1:0]] <= wd;
    always_ff @(posedge rclk) if (re) rd <= mb[rb[L-1:0]];
  end else begin : g_d
    (* ram_style = "distributed" *) logic [W-1:0] md [D];
    always_ff @(posedge wclk) if (we && !full) md[wb[L-1:0]] <= wd;
    always_ff @(posedge rclk) if (re) rd <= md[rb[L-1:0]];
  end
  always_ff @(posedge wclk)
    if (wrst) begin wb <= '0; afg_w <= '0; afs1_r <= '0; afs2_r <= '0; end
    else begin
      if (we && !full) begin wb <= wb + (L + 1)'(1); afg_w <= g(wb + (L + 1)'(1)); end
      afs1_r <= afg_r; afs2_r <= afs1_r;
    end
  always_ff @(posedge rclk)
    if (rrst) begin rb <= '0; afg_r <= '0; afs1_w <= '0; afs2_w <= '0; rv <= 1'b0; end
    else begin
      if (re) begin rb <= rb + (L + 1)'(1); afg_r <= g(rb + (L + 1)'(1)); end
      rv <= re || (rv && !rr);
      afs1_w <= afg_w; afs2_w <= afs1_w;
    end
endmodule
