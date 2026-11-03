// TARS-mini pipelined fp32 units for a 1 GHz clock (ASAP7 RVT TT, 650 ps logic per stage).
// Fixed latency, one operation per cycle, no stall, no reset (datapath only): the caller keeps a
// valid bit and any side data in step with the latency. Every unit computes exactly what the
// tm_fp_pkg function of the same name does (the contract's step sequence), so bits are unchanged;
// tests/test_rtl_fp.py streams the fp vectors through them.
//
//   unit        latency   composition (each fmul / fadd = 3 stages, see tm_fpp_pkg)
//   fpu_mul        3
//   fpu_add        3
//   fpu_exp       51      fmul, rint, i2f, 2 fmul, 2 fadd, 6 x (fmul, fadd), set_exp
//   fpu_recip     38      seed, 3 x (fmul, fadd, fmul, fadd), set_exp
//   fpu_rsqrt     56      seed, 3 x (3 fmul, fadd, fmul, fadd), set_exp
//   fpu_gs       122      GELU or SiLU (per operation): shared exp, +1, recip
//   fpu_act     1 / 122   identity, relu, relu6 in one stage; GELU, SiLU through fpu_gs

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
  import tm_fpp_pkg::*;
  fmul_s1_t s1;
  fmul_s2_t s2;
  always_ff @(posedge clk) begin
    s1 <= fmul_st1(a, b);
    s2 <= fmul_st2(s1);
    y  <= fmul_st3(s2);
  end
endmodule

module fpu_add (input logic clk, input logic [31:0] a, b, output logic [31:0] y);
  import tm_fpp_pkg::*;
  fadd_s1_t s1;
  fadd_s2_t s2;
  always_ff @(posedge clk) begin
    s1 <= fadd_st1(a, b);
    s2 <= fadd_st2(s1);
    y  <= fadd_st3(s2);
  end
endmodule

// y = fadd(fmul(a, b), c): c is taken 3 cycles after a, b (the caller delays it)
module fpu_mac3 (input logic clk, input logic [31:0] a, b, c, output logic [31:0] y);
  logic [31:0] m;
  fpu_mul u_m (.clk, .a, .b, .y(m));
  fpu_add u_a (.clk, .a(m), .b(c), .y);
endmodule

module fpu_exp (input logic clk, input logic [31:0] x, output logic [31:0] y);
  import tm_fp_pkg::*;
  localparam int L = 51;
  logic [31:0] xs, t, nf, mhi, mlo, r1, r, x8, mlo11;
  logic [31:0] p [7];
  logic [31:0] rr [7];
  logic signed [9:0] n4, n5;
  logic [2:0]  fl;                          // {nan, big, small} of x, delayed to the end
  logic [2:0]  fl_e;
  logic signed [9:0] n_e;
  assign xs = ftz(x);
  fpu_dly #(3, L - 1) u_fl (.clk, .d({is_nan(xs), !xs[31] && xs[30:0] >= 31'h42B20000,
                                      xs[31] && xs[30:0] >= 31'h42D00000}), .q(fl_e));
  fpu_mul u_t (.clk, .a(xs), .b(LOG2E), .y(t));                          // 3
  always_ff @(posedge clk) n4 <= 10'(rint_clamp(t));                     // 4
  always_ff @(posedge clk) begin nf <= i2f(32'(n4)); n5 <= n4; end      // 5
  fpu_mul u_hi (.clk, .a(nf), .b(NEG_LN2_HI), .y(mhi));                  // 8
  fpu_mul u_lo (.clk, .a(nf), .b(NEG_LN2_LO), .y(mlo));                  // 8
  fpu_dly #(32, 8) u_x8 (.clk, .d(xs), .q(x8));
  fpu_add u_r1 (.clk, .a(x8), .b(mhi), .y(r1));                          // 11
  fpu_dly #(32, 3) u_lo3 (.clk, .d(mlo), .q(mlo11));
  fpu_add u_r (.clk, .a(r1), .b(mlo11), .y(r));                          // 14
  // Horner: p <- fadd(fmul(p, r), C), six times, r delayed alongside
  localparam logic [31:0] C [7] = '{EXP_C0, EXP_C1, EXP_C2, EXP_C3, EXP_C4, EXP_C5, EXP_C6};
  assign p[0] = EXP_C0;
  assign rr[0] = r;
  for (genvar i = 1; i < 7; i++) begin : g_h
    fpu_mac3 u_m (.clk, .a(i == 1 ? C[0] : p[i-1]), .b(rr[i-1]), .c(C[i]), .y(p[i]));   // 14 + 6 i
    fpu_dly #(32, 6) u_r (.clk, .d(rr[i-1]), .q(rr[i]));
  end
  fpu_dly #(10, L - 6) u_n (.clk, .d(n5), .q(n_e));                      // n at cycle 50
  always_ff @(posedge clk)                                               // 51
    y <= fl_e[2] ? CANON_NAN : fl_e[1] ? 32'h7F800000 : fl_e[0] ? 32'h0 : set_exp(p[6], int'(n_e));
endmodule

module fpu_recip (input logic clk, input logic [31:0] x, output logic [31:0] y);
  import tm_fp_pkg::*;
  localparam int L = 38;
  logic [31:0] xs, m1, y1, m_e;
  logic [31:0] yi [4];
  logic [31:0] mi [4];
  logic [3:0]  fl, fl_e;                    // {nan, zero, inf, sign}
  logic [7:0]  ex_e;
  assign xs = ftz(x);
  fpu_dly #(4, L - 1) u_fl (.clk, .d({is_nan(xs), xs[30:0] == 0, is_inf(xs), xs[31]}), .q(fl_e));
  fpu_dly #(8, L - 1) u_ex (.clk, .d(xs[30:23]), .q(ex_e));
  always_ff @(posedge clk) begin                                         // 1: seed
    m1 <= {1'b0, 8'd127, xs[22:0]};
    y1 <= RECIP_MAGIC - {1'b0, 8'd127, xs[22:0]};
  end
  assign yi[0] = y1;
  assign mi[0] = m1;
  // y <- fadd(y, fmul(y, fadd(1, -fmul(m, y)))), three times, 12 cycles each
  for (genvar i = 0; i < 3; i++) begin : g_it
    logic [31:0] a, e, b, y6, y9;
    fpu_mul u_a (.clk, .a(mi[i]), .b(yi[i]), .y(a));
    fpu_add u_e (.clk, .a(F_ONE), .b(fneg(a)), .y(e));
    fpu_dly #(32, 6) u_y6 (.clk, .d(yi[i]), .q(y6));
    fpu_mul u_b (.clk, .a(y6), .b(e), .y(b));
    fpu_dly #(32, 3) u_y9 (.clk, .d(y6), .q(y9));
    fpu_add u_n (.clk, .a(y9), .b(b), .y(yi[i+1]));
    fpu_dly #(32, 12) u_m (.clk, .d(mi[i]), .q(mi[i+1]));
  end
  always_ff @(posedge clk) begin                                         // 38
    logic [31:0] s;
    s = set_exp(yi[3], -(int'(ex_e) - 127));
    y <= fl_e[3] ? CANON_NAN : fl_e[2] ? {fl_e[0], 31'h7F800000} : fl_e[1] ? {fl_e[0], 31'h0}
       : {fl_e[0], s[30:0]};
  end
endmodule

module fpu_rsqrt (input logic clk, input logic [31:0] x, output logic [31:0] y);
  import tm_fp_pkg::*;
  localparam int L = 56;
  logic [31:0] xs, m1, y1;
  logic [31:0] yi [4];
  logic [31:0] mi [4];
  logic [3:0]  fl_e;                        // {nan, zero, negative, inf}
  logic [31:0] k_e;
  int          E, k;
  assign xs = ftz(x);
  always_comb begin
    E = int'(xs[30:23]) - 127;
    k = E >>> 1;
  end
  fpu_dly #(4, L - 1) u_fl (.clk, .d({is_nan(xs), xs[30:0] == 0, xs[31], is_inf(xs)}), .q(fl_e));
  fpu_dly #(32, L - 1) u_k (.clk, .d(32'(k)), .q(k_e));
  always_ff @(posedge clk) begin                                         // 1: seed
    m1 <= {1'b0, 8'(127 + (E & 1)), xs[22:0]};
    y1 <= RSQRT_MAGIC - ({1'b0, 8'(127 + (E & 1)), xs[22:0]} >> 1);
  end
  assign yi[0] = y1;
  assign mi[0] = m1;
  // y <- fadd(y, fmul(y, fadd(0.5, -fmul(0.5, fmul(fmul(m, y), y))))), three times, 18 cycles each
  for (genvar i = 0; i < 3; i++) begin : g_it
    logic [31:0] a, b, c, e, f, y3, y12, y15;
    fpu_mul u_a (.clk, .a(mi[i]), .b(yi[i]), .y(a));                    // 3
    fpu_dly #(32, 3) u_y3 (.clk, .d(yi[i]), .q(y3));
    fpu_mul u_b (.clk, .a(a), .b(y3), .y(b));                           // 6
    fpu_mul u_c (.clk, .a(F_HALF), .b(b), .y(c));                       // 9
    fpu_add u_e (.clk, .a(F_HALF), .b(fneg(c)), .y(e));                 // 12
    fpu_dly #(32, 9) u_y12 (.clk, .d(y3), .q(y12));
    fpu_mul u_f (.clk, .a(y12), .b(e), .y(f));                          // 15
    fpu_dly #(32, 3) u_y15 (.clk, .d(y12), .q(y15));
    fpu_add u_n (.clk, .a(y15), .b(f), .y(yi[i+1]));                    // 18
    fpu_dly #(32, 18) u_m (.clk, .d(mi[i]), .q(mi[i+1]));
  end
  // specials in the reference's order: NaN, +-0 -> +-inf, x < 0 -> NaN, +inf -> +0
  always_ff @(posedge clk)                                               // 56
    y <= fl_e[3] ? CANON_NAN : fl_e[2] ? {fl_e[1], 31'h7F800000} : fl_e[1] ? CANON_NAN
       : fl_e[0] ? 32'h0 : set_exp(yi[3], -int'(k_e));
endmodule

// GELU (sel = 1) or SiLU (sel = 0), chosen per operation; they share exp, the +1 and recip:
//   GELU: u = c (x + a x^3); t = 1 - 2 recip(exp(2u) + 1); y = (x / 2)(1 + t)
//   SiLU: y = x recip(1 + exp(-x))      (fadd is commutative, so exp(-x) + 1 is the same sum)
module fpu_gs (input logic clk, input logic [31:0] x, input logic sel, output logic [31:0] y,
               output logic [31:0] ysig);   // ysig: recip(1 + exp(-x)) when sel = 0 (SiLU's factor)
  import tm_fp_pkg::*;
  localparam int L = 122;
  logic [31:0] xs, x2, x3, ax3, s, u, u2, x3d, x9, nx18, arg, ex, d, r, tq, t, ot, hx, hx119, xs110, sy;
  logic        sel18, sel_e;
  assign xs = ftz(x);
  fpu_mul u_x2 (.clk, .a(xs), .b(xs), .y(x2));                           // 3
  fpu_dly #(32, 3) u_xd3 (.clk, .d(xs), .q(x3d));
  fpu_mul u_x3 (.clk, .a(x2), .b(x3d), .y(x3));                          // 6
  fpu_mul u_ax (.clk, .a(GELU_A), .b(x3), .y(ax3));                      // 9
  fpu_dly #(32, 6) u_xd9 (.clk, .d(x3d), .q(x9));
  fpu_add u_s (.clk, .a(x9), .b(ax3), .y(s));                            // 12
  fpu_mul u_u (.clk, .a(GELU_C), .b(s), .y(u));                          // 15
  fpu_mul u_u2 (.clk, .a(F_TWO), .b(u), .y(u2));                         // 18
  fpu_dly #(32, 18) u_nx (.clk, .d(fneg(xs)), .q(nx18));
  fpu_dly #(1, 18) u_s18 (.clk, .d(sel), .q(sel18));
  assign arg = sel18 ? u2 : nx18;
  fpu_exp u_e (.clk, .x(arg), .y(ex));                                   // 69
  fpu_add u_d (.clk, .a(ex), .b(F_ONE), .y(d));                          // 72
  fpu_recip u_r (.clk, .x(d), .y(r));                                    // 110
  fpu_mul u_tq (.clk, .a(F_TWO), .b(r), .y(tq));                         // 113
  fpu_add u_t (.clk, .a(F_ONE), .b(fneg(tq)), .y(t));                    // 116
  fpu_add u_ot (.clk, .a(F_ONE), .b(t), .y(ot));                         // 119
  fpu_mul u_hx (.clk, .a(F_HALF), .b(xs), .y(hx));                       // 3
  fpu_dly #(32, 116) u_hxd (.clk, .d(hx), .q(hx119));
  fpu_dly #(32, 110) u_xs110 (.clk, .d(xs), .q(xs110));
  logic [31:0] gy, sy113, sy122;
  fpu_mul u_gy (.clk, .a(hx119), .b(ot), .y(gy));                        // 122
  fpu_mul u_sy (.clk, .a(xs110), .b(r), .y(sy113));                      // 113
  fpu_dly #(32, 9) u_syd (.clk, .d(sy113), .q(sy122));
  fpu_dly #(1, L - 18) u_sele (.clk, .d(sel18), .q(sel_e));
  assign y = sel_e ? gy : sy122;
  fpu_dly #(32, 12) u_sig (.clk, .d(r), .q(ysig));                     // 122
endmodule

// softplus (v0.4): x > 20 -> x; else relu(x) + (2 s) p(s^2), s = t recip(t + 2), t = exp(-|x|),
// p = Horner of LOG1P_C (8 multiply-add steps). 152 cycles.
module fpu_softplus (input logic clk, input logic [31:0] x, output logic [31:0] y);
  import tm_fp_pkg::*;
  logic [31:0] xs, t, t2, r, t92, sv, z, two_s, two_s146, rl152, m, sum, x152;
  logic [31:0] p [9];
  logic [31:0] zz [9];
  logic        big152;
  assign xs = ftz(x);
  fpu_exp u_t (.clk, .x(fneg(fabs(xs))), .y(t));                         // 51
  fpu_add u_t2 (.clk, .a(t), .b(F_TWO), .y(t2));                          // 54
  fpu_recip u_r (.clk, .x(t2), .y(r));                                    // 92
  fpu_dly #(32, 41) u_td (.clk, .d(t), .q(t92));
  fpu_mul u_s (.clk, .a(t92), .b(r), .y(sv));                             // 95
  fpu_mul u_z (.clk, .a(sv), .b(sv), .y(z));                              // 98
  fpu_mul u_2s (.clk, .a(F_TWO), .b(sv), .y(two_s));                      // 98
  assign p[0] = LOG1P_C[0];
  assign zz[0] = z;
  for (genvar i = 1; i < 9; i++) begin : g_h
    fpu_mac3 u_m (.clk, .a(p[i-1]), .b(zz[i-1]), .c(LOG1P_C[i]), .y(p[i]));   // 98 + 6 i
    fpu_dly #(32, 6) u_z (.clk, .d(zz[i-1]), .q(zz[i]));
  end
  fpu_dly #(32, 48) u_2sd (.clk, .d(two_s), .q(two_s146));
  fpu_mul u_m (.clk, .a(two_s146), .b(p[8]), .y(m));                      // 149
  fpu_dly #(32, 149) u_rl (.clk, .d((!xs[31] && xs[30:0] != 0) ? xs : 32'h0), .q(rl152));
  fpu_add u_y (.clk, .a(rl152), .b(m), .y(sum));                          // 152
  fpu_dly #(32, 152) u_x (.clk, .d(xs), .q(x152));
  fpu_dly #(1, 152) u_b (.clk, .d(!xs[31] && xs[30:0] > SP_HI[30:0] && !is_nan(xs)), .q(big152));
  assign y = big152 ? x152 : sum;
endmodule

// act_fn with the latency of the operation: identity / relu / relu6 one stage (y1 at +1); GELU,
// SiLU, sigmoid, exp at +122 (ylong); softplus at +SP_LAT (ysp). The caller picks by act.
`ifndef TM_FPGA
// One pipeline per function (fpu_gs, an exp, fpu_softplus); the caller holds act while
// operations are in flight (ylong is selected at the output).
module fpu_act (input logic clk, input logic [31:0] x, input logic [2:0] act,
                output logic [31:0] y1, output logic [31:0] ylong, output logic [31:0] ysp);
  import tm_fp_pkg::*;
  logic [31:0] ygs, ysig, ye, ye122;
  always_ff @(posedge clk) y1 <= act_simple(x, act[2] || act == 3'd3 ? 2'd0 : act[1:0]);
  fpu_gs u_gs (.clk, .x, .sel(act == 3'd3), .y(ygs), .ysig);
  fpu_exp u_e (.clk, .x, .y(ye));
  fpu_dly #(32, 71) u_ed (.clk, .d(ye), .q(ye122));
  assign ylong = (act == 3'd5) ? ysig : (act == 3'd7) ? ye122 : ygs;
  fpu_softplus u_sp (.clk, .x, .y(ysp));
endmodule
`else
// FPGA: one exp, one recip, and the +1 / +2 add and the x r / t r multiply shared by all five
// functions. Every function uses each shared unit at the same cycle (exp 18 -> 69, the add
// 69 -> 72, recip 72 -> 110, the multiply 110 -> 113): softplus runs 18 cycles after its ASIC
// timing (SP_LAT = 170) so that its exp lines up with GELU's. Each element carries its act code
// down the pipeline, so consecutive elements may use different functions. Every operation and
// operand order is the ASIC units' (fpu_gs, fpu_exp, fpu_softplus): the same bits.
// esel (held for a descriptor that uses no activation, the row engine's softmax): the exp takes
// xe directly, ye = exp(xe) 51 cycles later.
module fpu_act (input logic clk, input logic [31:0] x, input logic [2:0] act,
                output logic [31:0] y1, output logic [31:0] ylong, output logic [31:0] ysp,
                input logic esel, input logic [31:0] xe, output logic [31:0] ye);
  import tm_fp_pkg::*;
  logic [31:0] xs, x2, x3d, x3, ax3, x9, s, u, u2, a0, a18, arg, ex, c69, d, r;
  logic [31:0] tq, t, ot, hx, hx119, gy, xs110, t110, ma, mr, mr122, ysig, ye122, ygs;
  logic [31:0] z, two_s, two_s164, m, rl167, sum, x170;
  logic [31:0] p [9];
  logic [31:0] zz [9];
  logic [2:0]  act18, act69, act110, act122;
  logic        big170;
  always_ff @(posedge clk) y1 <= act_simple(x, act[2] || act == 3'd3 ? 2'd0 : act[1:0]);
  assign xs = ftz(x);
  // the act code alongside the element
  fpu_dly #(3, 18) u_a18 (.clk, .d(act), .q(act18));
  fpu_dly #(3, 51) u_a69 (.clk, .d(act18), .q(act69));
  fpu_dly #(3, 41) u_a110 (.clk, .d(act69), .q(act110));
  fpu_dly #(3, 12) u_a122 (.clk, .d(act110), .q(act122));
  // GELU's u = c (x + a x^3), 2u (fpu_gs)
  fpu_mul u_x2 (.clk, .a(xs), .b(xs), .y(x2));                           // 3
  fpu_dly #(32, 3) u_xd3 (.clk, .d(xs), .q(x3d));
  fpu_mul u_x3 (.clk, .a(x2), .b(x3d), .y(x3));                          // 6
  fpu_mul u_ax (.clk, .a(GELU_A), .b(x3), .y(ax3));                      // 9
  fpu_dly #(32, 6) u_xd9 (.clk, .d(x3d), .q(x9));
  fpu_add u_s (.clk, .a(x9), .b(ax3), .y(s));                            // 12
  fpu_mul u_u (.clk, .a(GELU_C), .b(s), .y(u));                          // 15
  fpu_mul u_u2 (.clk, .a(F_TWO), .b(u), .y(u2));                         // 18
  // the exp's argument: 2u (GELU), -x (SiLU, sigmoid), x (exp; fpu_exp flushes x itself, so
  // ftz(x) is the same), -|x| (softplus)
  assign a0 = (act == 3'd7) ? xs : (act == 3'd6) ? fneg(fabs(xs)) : fneg(xs);
  fpu_dly #(32, 18) u_a0 (.clk, .d(a0), .q(a18));
  assign arg = esel ? xe : (act18 == 3'd3) ? u2 : a18;
  fpu_exp u_e (.clk, .x(arg), .y(ex));                                   // 69
  assign ye = ex;
  assign c69 = (act69 == 3'd6) ? F_TWO : F_ONE;
  fpu_add u_d (.clk, .a(ex), .b(c69), .y(d));                            // 72: exp + 1, t + 2
  fpu_recip u_r (.clk, .x(d), .y(r));                                    // 110
  fpu_dly #(32, 53) u_ed (.clk, .d(ex), .q(ye122));                      // exp at 122
  // GELU: t = 1 - 2 r, y = (x / 2)(1 + t); SiLU: x r; sigmoid: r
  fpu_mul u_tq (.clk, .a(F_TWO), .b(r), .y(tq));                         // 113
  fpu_add u_t (.clk, .a(F_ONE), .b(fneg(tq)), .y(t));                    // 116
  fpu_add u_ot (.clk, .a(F_ONE), .b(t), .y(ot));                         // 119
  fpu_mul u_hx (.clk, .a(F_HALF), .b(xs), .y(hx));                       // 3
  fpu_dly #(32, 116) u_hxd (.clk, .d(hx), .q(hx119));
  fpu_mul u_gy (.clk, .a(hx119), .b(ot), .y(gy));                        // 122
  fpu_dly #(32, 110) u_xs110 (.clk, .d(xs), .q(xs110));
  fpu_dly #(32, 41) u_td (.clk, .d(ex), .q(t110));
  assign ma = (act110 == 3'd6) ? t110 : xs110;
  fpu_mul u_m (.clk, .a(ma), .b(r), .y(mr));                             // 113: x r (SiLU), s = t r (softplus)
  fpu_dly #(32, 9) u_mrd (.clk, .d(mr), .q(mr122));
  fpu_dly #(32, 12) u_sig (.clk, .d(r), .q(ysig));                       // 122
  assign ygs = (act122 == 3'd3) ? gy : mr122;
  assign ylong = (act122 == 3'd5) ? ysig : (act122 == 3'd7) ? ye122 : ygs;
  // softplus: relu(x) + (2 s) p(s^2), x > 20 -> x (fpu_softplus, 18 cycles later)
  fpu_mul u_z (.clk, .a(mr), .b(mr), .y(z));                             // 116
  fpu_mul u_2s (.clk, .a(F_TWO), .b(mr), .y(two_s));                     // 116
  assign p[0] = LOG1P_C[0];
  assign zz[0] = z;
  for (genvar i = 1; i < 9; i++) begin : g_h
    fpu_mac3 u_m (.clk, .a(p[i-1]), .b(zz[i-1]), .c(LOG1P_C[i]), .y(p[i]));   // 116 + 6 i
    fpu_dly #(32, 6) u_z (.clk, .d(zz[i-1]), .q(zz[i]));
  end
  fpu_dly #(32, 48) u_2sd (.clk, .d(two_s), .q(two_s164));
  fpu_mul u_pm (.clk, .a(two_s164), .b(p[8]), .y(m));                    // 167
  fpu_dly #(32, 167) u_rl (.clk, .d((!xs[31] && xs[30:0] != 0) ? xs : 32'h0), .q(rl167));
  fpu_add u_y (.clk, .a(rl167), .b(m), .y(sum));                         // 170
  fpu_dly #(32, 170) u_x (.clk, .d(xs), .q(x170));
  fpu_dly #(1, 170) u_b (.clk, .d(!xs[31] && xs[30:0] > SP_HI[30:0] && !is_nan(xs)), .q(big170));
  assign ysp = big170 ? x170 : sum;
endmodule
`endif
