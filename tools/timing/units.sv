// Timing wrappers: each tm_fp_pkg function (and the core's composite paths) between input and
// output registers, so the critical path is register to register.
module u_reg2 #(parameter int OP = 0) (input logic clk, input logic [31:0] a, b, input logic [2:0] act,
                                       output logic [31:0] y);
  import tm_fp_pkg::*;
  logic [31:0] ra, rb, f;
  logic [2:0]  rc;
  always_comb begin
    case (OP)
      0:  f = fmul(ra, rb);
      1:  f = fadd(ra, rb);
      2:  f = i2f(ra);
      3:  f = exp_f(ra);
      4:  f = recip_f(ra);
      5:  f = rsqrt_f(ra);
      6:  f = gelu_f(ra);
      7:  f = silu_f(ra);
      8:  f = act_fn(ra, rc);
      9:  f = {16'b0, to_bf16(ra)};
      10: f = {24'b0, q8(ra, rb)};
      // core composites
      11: f = act_fn(fadd(fmul(i2f(ra), rb), ra ^ rb), rc);          // GEMM epilogue lane
      12: f = fmul(exp_f(fadd(ra, fneg(rb))), ra ^ rb);               // softmax output element
      13: f = fadd(ra, exp_f(fadd(rb, fneg(ra ^ rb))));              // softmax pass-1 lane update
      14: f = rsqrt_f(fadd(fmul(ra, rb), ra ^ rb));                   // norm scalar after the fold
      15: f = recip_f(ra);                                             // softmax scalar after the fold
      // calibration: plain structures with known delays
      20: f = ra + rb;                                                 // 32-bit adder
      21: f = 32'(({8'b0, ra[23:0]} * {8'b0, rb[23:0]}) >> 16);        // 24 x 24 multiplier
      22: f = ra ^ rb;                                                 // one gate level
      default: f = fadd(fadd(fadd(ra, rb), fadd(rb, ra)), fadd(fadd(ra, ra), fadd(rb, rb)));  // fold levels
    endcase
  end
  always_ff @(posedge clk) begin ra <= a; rb <= b; rc <= act; y <= f; end
endmodule

// N-stage version: the unit's result through N registers (the RTL's latency); retiming spreads
// them through the logic. Used to check that N stages meet the per-stage budget.
module u_pipe #(parameter int OP = 0, parameter int N = 2) (input logic clk, input logic [31:0] a, b,
                                                             input logic [2:0] act, output logic [31:0] y);
  logic [31:0] f0;
  logic [31:0] d [N];
  u_reg2 #(.OP(OP)) u (.clk, .a, .b, .act, .y(f0));   // inputs registered, result registered once
  always_ff @(posedge clk) begin
    d[0] <= f0;
    for (int i = 1; i < N; i++) d[i] <= d[i - 1];
  end
  assign y = d[N - 1];
endmodule
