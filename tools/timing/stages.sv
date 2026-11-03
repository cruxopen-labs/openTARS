// Each pipeline stage function between registers (its inputs and its result registered).
module s_stage #(parameter int OP = 0) (input logic clk, input logic [255:0] in_bits, output logic [255:0] out_bits);
  import tm_fp_pkg::*;
  import tm_fpp_pkg::*;
  logic [255:0] r, f;
  always_comb begin
    f = '0;
    case (OP)
      0: f = 256'(fadd_st1(r[31:0], r[63:32]));
      1: f = 256'(fadd_st2(fadd_s1_t'(r[$bits(fadd_s1_t)-1:0])));
      2: f = 256'(fadd_st3(fadd_s2_t'(r[$bits(fadd_s2_t)-1:0])));
      3: f = 256'(fmul_st1(r[31:0], r[63:32]));
      4: f = 256'(fmul_st2(fmul_s1_t'(r[$bits(fmul_s1_t)-1:0])));
      5: f = 256'(fmul_st3(fmul_s2_t'(r[$bits(fmul_s2_t)-1:0])));
      6: f = 256'(i2f(r[31:0]));
      7: f = 256'(to_bf16(r[31:0]));
      8: f = 256'(rint_clamp(r[31:0]));
      9: f = 256'(set_exp(r[31:0], int'($signed(r[40:32]))));
      10: f = 256'(lzc_norm32(r[31:0]));
      11: f = 256'(q8_tail(r[31:0]));
      12: f = 256'(fmax(r[31:0], r[63:32]));
      13: f = 256'(act_fn(r[31:0], 3'(r[34:32] & 3'd3)));   // identity / relu / relu6
      14: f = 256'(32'h7EF311C7 - {1'b0, 8'd127, r[22:0]});  // recip seed
      15: f = 256'(fmax(fmax(fmax(r[31:0], r[63:32]), fmax(r[95:64], r[127:96])), fmax(fmax(r[159:128], r[191:160]), fmax(r[223:192], r[255:224]))));
      default: f = 256'(r[31:0] + r[63:32]);
    endcase
  end
  always_ff @(posedge clk) begin r <= in_bits; out_bits <= f; end
endmodule
