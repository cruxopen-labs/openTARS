// TARS-mini FPGA build: one fpu_act for the whole core (the shared SFU), see tm_core.
//
// The long activations (GELU, SiLU, sigmoid, exp: 122 cycles; softplus: SP_LAT) go through a
// single fpu_act at the core level. Only one unit (the GEMM epilogue, the row engine, the
// extension unit) runs a descriptor at a time and the descriptor's act is fixed, so the SFU just
// takes its input from the active unit. A unit with W lanes launches a step at most every W cycles
// (its own issue gate) and its tm_sfu_port feeds the step's W values to the SFU one per cycle;
// lane k enters k cycles after the step, so the whole step's results are back L + W - 1 cycles
// after it (L = the SFU's latency for the act): the unit adds W - 1 to its long-act latency.
// identity / relu / relu6 stay in the lanes (fpu_act1, one stage).

// tm_sfu_port: v marks the cycle a step's values are at x (cycle 0). The SFU sees x[0] at 0 and
// x[k] at k; sy is the SFU's result, and y[k] is lane k's result at cycle L + W - 1. W = 1 is the
// direct connection. Steps must be at least W cycles apart.
module tm_sfu_port #(parameter int W = 4) (
  input  logic        clk,
  input  logic        v,
  input  logic [31:0] x [W],
  output logic [31:0] sx,            // to the SFU
  input  logic [31:0] sy,            // from the SFU
  output logic [31:0] y [W]
);
  if (W == 1) begin : g_direct
    assign sx   = x[0];
    assign y[0] = sy;
  end else begin : g_skew
    logic [31:0] sh [1:W-1];         // sh[k]: the value for k cycles after the step (as of now: k - age)
    logic [31:0] od [1:W-1];         // the SFU's results, od[j] = sy j cycles ago
    always_ff @(posedge clk) begin
      for (int k = 1; k < W; k++)
        if (v) sh[k] <= x[k];
        else if (k < W - 1) sh[k] <= sh[(k < W - 1) ? k + 1 : k];
      od[1] <= sy;
      for (int j = 2; j < W; j++) od[j] <= od[j - 1];
    end
    assign sx = v ? x[0] : sh[1];
    for (genvar k = 0; k < W - 1; k++) begin : g_y
      assign y[k] = od[W - 1 - k];
    end
    assign y[W-1] = sy;
  end
endmodule

// identity / relu / relu6 in one stage: fpu_act's y1, for the lanes that keep it
module fpu_act1 (input logic clk, input logic [31:0] x, input logic [2:0] act, output logic [31:0] y1);
  always_ff @(posedge clk)
    y1 <= tm_fp_pkg::act_simple(x, act[2] || act == 3'd3 ? 2'd0 : act[1:0]);
endmodule
