// TARS-mini row-engine datapaths (contract v0.3 row ops and activations), pipelined for 1 GHz.
//
// vpu_lane: one element slot of a row-engine step. A step's operands enter at cycle 0 (from the
// core's issue registers); the lane produces, at fixed cycles that depend on the descriptor:
//   contribution to the pass's lane sum  softmax p1: e = exp(x - mx)     at 54
//                                        LayerNorm p1: (x - mean)^2     at  6
//                                        RMSNorm p0: x * x              at  3
//                                        (softmax p0 / LayerNorm p0 contribute x itself, at 0)
//   output element, before conversion    softmax: e * r                 at 57
//                                        LayerNorm: (d * rs) * g (+ b)  at 12
//                                        RMSNorm: (x * rs) * g (+ b)    at  9
//                                        activation                     at  1, 122 (GELU, SiLU,
//                                                                       sigmoid, exp) or SP_LAT (softplus)
//   conversion                           fp32 +0, bf16 +1, int8 +4
// FPGA: the exp is the core's shared one (xf out, ye back), so softmax's e and e * r come EXD
// cycles later (the lanes take it in turn).
// Shared units serve contributions and outputs of the same function at the same cycle offset, so
// one step per cycle never collides; `kout_*` (the step is an output step) and the step's scalars
// arrive already delayed to where they are used, so a step carries its own state down the lane.
module vpu_lane #(
  parameter int EXD = 0            // FPGA: the shared exp's lane skew (cycles after 54 for e)
) (
  input  logic        clk,
  input  logic [1:0]  fn,          // 0 softmax, 1 LayerNorm, 2 RMSNorm, 3 activation
  input  logic        bias,
  input  logic [2:0]  act,
  input  logic [1:0]  out,         // 0 fp32, 1 bf16, 2 int8
  input  logic [31:0] osc,
  input  logic        keep,        // cycle 0: the element is inside the row (v0.4 causal softmax)
  input  logic [31:0] x, g, b,     // cycle 0
  input  logic [31:0] c1,          // cycle 0: mx (softmax) or mean (LayerNorm)
  input  logic [31:0] sc_d0, sc_d3, sc_d54,   // the step's r / rs at cycles 0, 3, 54 (+ EXD)
  input  logic        kout_d0, kout_d3,       // the step is an output step, at cycles 0, 3
  output logic [31:0] contrib,     // valid at the contribution cycle above
  output logic [31:0] y            // valid at the output cycle + conversion
`ifdef TM_FPGA
  , input logic [31:0] yl          // the long activation from the core's shared SFU (tm_sfu.sv)
  , output logic [31:0] xf         // the exp's argument x - mx (cycle 3) to the shared SFU's exp
  , input logic [31:0] ye          // exp(xf), at 54 + EXD
`endif
);
  import tm_fp_pkg::*;
  logic        keep54;
  logic [31:0] E0, F, E, a1, b1, P1, g6, g3, gd, b9, b6, bd, P2, Q, P2d, pre, ya1, yalong, yasp, ybf, yq3;
  logic [7:0]  q8r;
  fpu_add u_f (.clk, .a(x), .b(fneg(c1)), .y(F));                      // 3
`ifdef TM_FPGA
  assign xf = F;
  assign E0 = ye;                                                      // 54 + EXD
`else
  fpu_exp u_e (.clk, .x(F), .y(E0));                                   // 54
`endif
  fpu_dly #(1, 54 + EXD) u_k (.clk, .d(keep), .q(keep54));
  assign E = keep54 ? E0 : 32'h0;                                      // masked: +0
  always_comb
    case (fn)
      2'd0:    begin a1 = E; b1 = sc_d54; end                          // softmax output
      2'd1:    begin a1 = F; b1 = kout_d3 ? sc_d3 : F; end             // LayerNorm
      default: begin a1 = x; b1 = kout_d0 ? sc_d0 : x; end             // RMSNorm
    endcase
  fpu_mul u_p1 (.clk, .a(a1), .b(b1), .y(P1));                         // 57 / 6 / 3
  fpu_dly #(32, 6) u_g6 (.clk, .d(g), .q(g6));
  fpu_dly #(32, 3) u_g3 (.clk, .d(g), .q(g3));
  assign gd = (fn == 2'd1) ? g6 : g3;
  fpu_mul u_p2 (.clk, .a(P1), .b(gd), .y(P2));                         // 9 / 6
  fpu_dly #(32, 9) u_b9 (.clk, .d(b), .q(b9));
  fpu_dly #(32, 6) u_b6 (.clk, .d(b), .q(b6));
  assign bd = (fn == 2'd1) ? b9 : b6;
  fpu_add u_q (.clk, .a(P2), .b(bd), .y(Q));                           // 12 / 9
  fpu_dly #(32, 3) u_p2d (.clk, .d(P2), .q(P2d));
`ifdef TM_FPGA
  fpu_act1 u_act (.clk, .x, .act, .y1(ya1));                           // 1
  assign yalong = yl;                                                  // 122 / SP_LAT + NS - 1
  assign yasp   = yl;
`else
  fpu_act u_act (.clk, .x, .act, .y1(ya1), .ylong(yalong), .ysp(yasp)); // 1 / 122 / SP_LAT
`endif
  always_comb
    case (fn)
      2'd0:    pre = P1;
      2'd1,
      2'd2:    pre = bias ? Q : P2d;
      default: pre = (act == 3'd3 || act == 3'd4 || act == 3'd5 || act == 3'd7) ? yalong
                   : (act == 3'd6) ? yasp : ya1;
    endcase
  always_ff @(posedge clk) ybf <= {16'b0, to_bf16(pre)};               // +1
  fpu_mul u_q8 (.clk, .a(pre), .b(osc), .y(yq3));                      // +3
  always_ff @(posedge clk) q8r <= q8_tail(yq3);                        // +4
  assign y = (out == 2'd0) ? pre : (out == 2'd1) ? ybf : {24'b0, q8r};
  assign contrib = (fn == 2'd0) ? E : P1;
endmodule

// vpu_fold: the end of a pass. The 16 lane values enter at cycle 0 (`start`); the contract's
// fold and the pass scalar come out at fixed cycles:
//   max   4 levels of fmax                     at  4   (softmax p0: mx)
//   sum   0+1, 2+3, ... four levels of fadd   at 12
//   mean  fmul(sum, 1/N)                       at 15   (LayerNorm p0)
//   rs    rsqrt(fadd(fmul(sum, 1/N), eps))     at 74   (LayerNorm p1, RMSNorm p0)
//   r     recip(sum)                           at 50   (softmax p1)
module vpu_fold (
  input  logic        clk,
  input  logic [31:0] lane [16],
  input  logic [31:0] invn, eps,
  output logic [31:0] mx, mean, rs, r
`ifdef TM_FPGA
  // FPGA: the recip and the sum's adder tree also serve the core while the row engine is idle
  // (1/N at dispatch; the extension unit's QUANT and NT fold): rsel puts rx on the recip
  , input  logic        rsel
  , input  logic [31:0] rx
  , output logic [31:0] fsum
`endif
);
  import tm_fp_pkg::*;
  logic [31:0] t8 [8], t4 [4], t2 [2], sum, m8 [8], m4 [4], m2 [2], m, a;
  for (genvar i = 0; i < 8; i++) begin : g8
    fpu_add u (.clk, .a(lane[2 * i]), .b(lane[2 * i + 1]), .y(t8[i]));
  end
  for (genvar i = 0; i < 4; i++) begin : g4
    fpu_add u (.clk, .a(t8[2 * i]), .b(t8[2 * i + 1]), .y(t4[i]));
  end
  for (genvar i = 0; i < 2; i++) begin : g2
    fpu_add u (.clk, .a(t4[2 * i]), .b(t4[2 * i + 1]), .y(t2[i]));
  end
  fpu_add u_s (.clk, .a(t2[0]), .b(t2[1]), .y(sum));                     // 12
  // the max fold: order-free (fmax is commutative and associative on the contract's values)
  always_ff @(posedge clk) begin
    for (int i = 0; i < 8; i++) m8[i] <= fmax(lane[2 * i], lane[2 * i + 1]);
    for (int i = 0; i < 4; i++) m4[i] <= fmax(m8[2 * i], m8[2 * i + 1]);
    for (int i = 0; i < 2; i++) m2[i] <= fmax(m4[2 * i], m4[2 * i + 1]);
    mx <= fmax(m2[0], m2[1]);                                            // 4
  end
  fpu_mul u_m (.clk, .a(sum), .b(invn), .y(m));                          // 15
  assign mean = m;
  fpu_add u_a (.clk, .a(m), .b(eps), .y(a));                             // 18
  fpu_rsqrt u_rs (.clk, .x(a), .y(rs));                                  // 74
`ifdef TM_FPGA
  assign fsum = sum;
  fpu_recip u_r (.clk, .x(rsel ? rx : sum), .y(r));                      // 50 (rx: 38)
`else
  fpu_recip u_r (.clk, .x(sum), .y(r));                                  // 50
`endif
endmodule
