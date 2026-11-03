// Behavioral twins of rtl/tm_fpu.sv for fast core simulation: each unit is its tm_fp_pkg function
// followed by a delay line of the unit's latency. The pipelines never stall, so a structural unit's
// output at cycle t + L is exactly f(inputs at t): same bits, same cycles. tests/test_rtl_fp.py
// checks the structural units against the functions (all fp vectors, one per cycle), and
// tests/test_rtl.py runs some core tests on the structural build (rtlsim structural=True).
module fpu_dly #(parameter int W = 32, parameter int N = 1) (
  input  logic         clk,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
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

module fpu_mul (input logic clk, input logic [31:0] a, b, output logic [31:0] y);
  fpu_dly #(32, 3) u (.clk, .d(tm_fp_pkg::fmul(a, b)), .q(y));
endmodule

module fpu_add (input logic clk, input logic [31:0] a, b, output logic [31:0] y);
  fpu_dly #(32, 3) u (.clk, .d(tm_fp_pkg::fadd(a, b)), .q(y));
endmodule

module fpu_mac3 (input logic clk, input logic [31:0] a, b, c, output logic [31:0] y);
  logic [31:0] m;
  fpu_mul u_m (.clk, .a, .b, .y(m));
  fpu_add u_a (.clk, .a(m), .b(c), .y);
endmodule

module fpu_exp (input logic clk, input logic [31:0] x, output logic [31:0] y);
  fpu_dly #(32, 51) u (.clk, .d(tm_fp_pkg::exp_f(x)), .q(y));
endmodule

module fpu_recip (input logic clk, input logic [31:0] x, output logic [31:0] y);
  fpu_dly #(32, 38) u (.clk, .d(tm_fp_pkg::recip_f(x)), .q(y));
endmodule

module fpu_rsqrt (input logic clk, input logic [31:0] x, output logic [31:0] y);
  fpu_dly #(32, 56) u (.clk, .d(tm_fp_pkg::rsqrt_f(x)), .q(y));
endmodule

module fpu_gs (input logic clk, input logic [31:0] x, input logic sel, output logic [31:0] y,
               output logic [31:0] ysig);
  fpu_dly #(32, 122) u (.clk, .d(sel ? tm_fp_pkg::gelu_f(x) : tm_fp_pkg::silu_f(x)), .q(y));
  fpu_dly #(32, 122) u_s (.clk, .d(tm_fp_pkg::sigmoid_f(x)), .q(ysig));
endmodule

module fpu_softplus (input logic clk, input logic [31:0] x, output logic [31:0] y);
  fpu_dly #(32, tm_fp_pkg::SP_LAT) u (.clk, .d(tm_fp_pkg::softplus_f(x)), .q(y));
endmodule

// act_fn with the latency of the operation: identity / relu / relu6 one stage (y1 at +1); GELU,
// SiLU, sigmoid, exp at +122 (ylong); softplus at +SP_LAT (ysp). The caller picks by act.
// FPGA: the exp also serves the row engine's softmax (esel: ye = exp(xe) at +51), see tm_fpu.sv
module fpu_act (input logic clk, input logic [31:0] x, input logic [2:0] act,
                output logic [31:0] y1, output logic [31:0] ylong, output logic [31:0] ysp
`ifdef TM_FPGA
                , input logic esel, input logic [31:0] xe, output logic [31:0] ye
`endif
                );
  always_ff @(posedge clk) y1 <= tm_fp_pkg::act_simple(x, act[2] || act == 3'd3 ? 2'd0 : act[1:0]);
  fpu_dly #(32, 122) u_l (.clk, .d((act == 3'd3 || act == 3'd4 || act == 3'd5 || act == 3'd7)
                                    ? tm_fp_pkg::act_fn(x, act) : 32'h0), .q(ylong));
  fpu_softplus u_sp (.clk, .x, .y(ysp));
`ifdef TM_FPGA
  fpu_exp u_e (.clk, .x(xe), .y(ye));
`endif
endmodule
