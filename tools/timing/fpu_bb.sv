// Black-box stubs of the pipelined fp units for core timing (the units are timed stage by stage
// on their own; here their inputs are path ends and their outputs path starts).
(* blackbox *) module fpu_mul (input logic clk, input logic [31:0] a, b, output logic [31:0] y); endmodule
(* blackbox *) module fpu_add (input logic clk, input logic [31:0] a, b, output logic [31:0] y); endmodule
(* blackbox *) module fpu_exp (input logic clk, input logic [31:0] x, output logic [31:0] y); endmodule
(* blackbox *) module fpu_recip (input logic clk, input logic [31:0] x, output logic [31:0] y); endmodule
(* blackbox *) module fpu_rsqrt (input logic clk, input logic [31:0] x, output logic [31:0] y); endmodule
(* blackbox *) module fpu_gs (input logic clk, input logic [31:0] x, input logic sel, output logic [31:0] y,
                              output logic [31:0] ysig); endmodule
(* blackbox *) module fpu_softplus (input logic clk, input logic [31:0] x, output logic [31:0] y); endmodule
module fpu_mac3 (input logic clk, input logic [31:0] a, b, c, output logic [31:0] y);
  logic [31:0] m;
  fpu_mul u_m (.clk, .a, .b, .y(m));
  fpu_add u_a (.clk, .a(m), .b(c), .y);
endmodule
module fpu_act (input logic clk, input logic [31:0] x, input logic [2:0] act,
                output logic [31:0] y1, output logic [31:0] ylong, output logic [31:0] ysp);
  logic [31:0] ygs, ysig, ye, ye122;
  always_ff @(posedge clk) y1 <= tm_fp_pkg::act_simple(x, act[2] || act == 3'd3 ? 2'd0 : act[1:0]);
  fpu_gs u_gs (.clk, .x, .sel(act == 3'd3), .y(ygs), .ysig);
  fpu_exp u_e (.clk, .x, .y(ye));
  fpu_dly #(32, 71) u_ed (.clk, .d(ye), .q(ye122));
  assign ylong = (act == 3'd5) ? ysig : (act == 3'd7) ? ye122 : ygs;
  fpu_softplus u_sp (.clk, .x, .y(ysp));
endmodule
module fpu_dly #(parameter int W = 32, parameter int N = 1) (input logic clk, input logic [W-1:0] d,
                                                             output logic [W-1:0] q);
  if (N == 0) begin : g_wire
    assign q = d;
  end else begin : g_regs
    logic [W-1:0] r [N];
    always_ff @(posedge clk) r[0] <= d;
    for (genvar i = 1; i < N; i++) begin : g_r
      always_ff @(posedge clk) r[i] <= r[i - 1];
    end
    assign q = r[N-1];
  end
endmodule
// the extension unit's delay lines: a 256 x 32 two-port memory with a registered read (an SRAM
// macro in a real design; its inputs end paths, its output starts them)
(* blackbox *) module tm_vdly (input logic clk, input logic rst, input logic [31:0] d, input logic [31:0] n,
                               output logic [31:0] q); endmodule
(* blackbox *) module tm_xbuf (input logic clk, input logic we, input logic [7:0] wa, input logic [127:0] wd,
                               input logic [7:0] ra, output logic [127:0] rd); endmodule
