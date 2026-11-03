// A delay line of n cycles (n held while data flows, 1 <= n <= 256): a circular buffer with a
// registered read (one write and one read port, as a small two-port SRAM; the timing flow treats
// it as one). q(t) = d(t - n).
module tm_vdly (input logic clk, input logic rst, input logic [31:0] d, input logic [31:0] n,
                output logic [31:0] q);
  logic [31:0] mem [256];
  logic [7:0]  wp;
  always_ff @(posedge clk) begin
    if (rst) wp <= 0;
    else begin mem[wp] <= d; wp <= wp + 8'd1; end
    // the entry written n - 1 cycles ago (n = 1: this cycle's input itself)
    q <= (n[8:0] == 9'd1) ? d : mem[8'(wp - n[7:0] + 8'd1)];
  end
endmodule

// The GEMV unit's x buffer: 256 beats of 16 bytes, one write port and one registered read port
// (a small two-port SRAM; a black box in the timing flow).
module tm_xbuf (input logic clk, input logic we, input logic [7:0] wa, input logic [127:0] wd,
                input logic [7:0] ra, output logic [127:0] rd);
  logic [127:0] mem [256];
  always_ff @(posedge clk) begin
    if (we) mem[wa] <= wd;
    rd <= mem[ra];
  end
endmodule
