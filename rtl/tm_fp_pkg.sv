// TARS-mini fp32 arithmetic (contract v0.2, 数值语义): roundTiesToEven only, flush-to-zero on
// inputs and results, no FMA, every NaN a mul or add produces is 0x7FC00000.
// Each function is one step of the contract's sequence and must match tars_mini/numerics.py
// bit for bit (tests/test_rtl_fp.py checks it on random and edge vectors).
package tm_fp_pkg;

  localparam logic [31:0] CANON_NAN = 32'h7FC00000;

  // IEEE binary16 storage only. Subnormals are preserved on conversion; arithmetic
  // still uses the FP32 contract. RNE on store, canonical NaN, signed zero/overflow.
  function automatic logic [15:0] to_fp16(input logic [31:0] x);
    logic [31:0] q, sig, lost, halfway;
    logic [4:0] sh;
    if (x[30:23] == 8'hff)
      return x[22:0] != 0 ? 16'h7e00 : {x[31], 5'h1f, 10'b0};
    if (x[30:23] > 8'd142) return {x[31], 5'h1f, 10'b0};
    if (x[30:23] >= 8'd113) begin
      q = ((32'(x[30:23]) - 32'd112) << 10) | (32'(x[22:0]) >> 13);
      q = q + 32'(x[12] && ((|x[11:0]) || q[0]));
    end else if (x[30:23] >= 8'd102) begin
      sh = 5'(8'd126 - x[30:23]);
      sig = {8'b0, 1'b1, x[22:0]};
      q = sig >> sh;
      lost = sig & ((32'd1 << sh) - 32'd1);
      halfway = 32'd1 << (sh - 5'd1);
      q = q + 32'(lost > halfway || (lost == halfway && q[0]));
    end else q = 32'd0;
    return {x[31], q[14:0]};
  endfunction

  function automatic logic [31:0] from_fp16(input logic [15:0] x);
    logic [7:0] e;
    logic [31:0] m;
    if (x[14:10] == 5'h1f)
      return x[9:0] != 0 ? CANON_NAN : {x[15], 8'hff, 23'b0};
    if (x[14:10] != 0) begin
      e = 8'(x[14:10]) + 8'd112;
      return {x[15], e, x[9:0], 13'b0};
    end
    e = 8'd0; m = 32'd0;
    for (int i = 0; i < 10; i++) if (x[i]) begin
      e = 8'(103 + i); m = 32'(x[9:0]) << (23 - i);
    end
    return {x[15], e, m[22:0]};
  endfunction

  // softplus latency (fpu_act ysp). The FPGA build (TM_FPGA) runs softplus 18 cycles later so
  // that it uses fpu_act's one exp and one recip in the same slots as GELU / SiLU (tm_fpu.sv)
`ifdef TM_FPGA
  localparam int SP_LAT = 170;
`else
  localparam int SP_LAT = 152;
`endif

  function automatic logic is_nan(input logic [31:0] x);
    return (x[30:23] == 8'hFF) && (x[22:0] != 0);
  endfunction

  function automatic logic is_inf(input logic [31:0] x);
    return (x[30:23] == 8'hFF) && (x[22:0] == 0);
  endfunction

  // denormal -> zero of the same sign
  function automatic logic [31:0] ftz(input logic [31:0] x);
    return (x[30:23] == 0) ? {x[31], 31'b0} : x;
  endfunction

  // ---------------------------------------------------------------- shared structures
  // Leading-zero count and normalization in one log2 tree (5 levels of OR-reduce + mux); the
  // timing-friendly form of "shift left until the top bit is 1". x == 0 gives n = 32, v = 0.
  function automatic logic [37:0] lzc_norm32(input logic [31:0] x);   // {n[5:0], v[31:0]}
    logic [31:0] v;
    logic [5:0]  n;
    v = x; n = 6'd0;
    if (v[31:16] == 16'b0) begin n[4] = 1'b1; v = {v[15:0], 16'b0}; end
    if (v[31:24] == 8'b0)  begin n[3] = 1'b1; v = {v[23:0], 8'b0}; end
    if (v[31:28] == 4'b0)  begin n[2] = 1'b1; v = {v[27:0], 4'b0}; end
    if (v[31:30] == 2'b0)  begin n[1] = 1'b1; v = {v[29:0], 2'b0}; end
    if (v[31] == 1'b0)     begin n[0] = 1'b1; v = {v[30:0], 1'b0}; end
    if (x == 32'b0) n = 6'd32;
    return {n, v};
  endfunction

  // ---------------------------------------------------------------- multiply
  function automatic logic [31:0] fmul(input logic [31:0] a_in, input logic [31:0] b_in);
    logic [31:0] a, b;
    logic        s;
    logic [23:0] ma, mb, m24;
    logic [47:0] p;
    logic        g, st, inc;
    logic signed [10:0] e;
    logic [24:0] mr;
    logic [26:0] x, xs, lost;
    logic [23:0] q;
    logic        qg, qst;
    logic [10:0] sh;
    a = ftz(a_in);
    b = ftz(b_in);
    s = a[31] ^ b[31];
    if (is_nan(a) || is_nan(b)) return CANON_NAN;
    if (is_inf(a) || is_inf(b)) begin
      if (a[30:0] == 0 || b[30:0] == 0) return CANON_NAN;     // inf * 0
      return {s, 8'hFF, 23'b0};
    end
    if (a[30:0] == 0 || b[30:0] == 0) return {s, 31'b0};
    ma = {1'b1, a[22:0]};
    mb = {1'b1, b[22:0]};
    p  = ma * mb;                                             // in [2^46, 2^48)
    e  = $signed({3'b0, a[30:23]}) + $signed({3'b0, b[30:23]}) - 11'sd127;
    if (p[47]) begin
      m24 = p[47:24]; g = p[23]; st = |p[22:0]; e = e + 11'sd1;
    end else begin
      m24 = p[46:23]; g = p[22]; st = |p[21:0];
    end
    if (e >= 11'sd1) begin
      inc = g & (st | m24[0]);
      mr  = {1'b0, m24} + 25'(inc);
      if (mr[24]) begin
        mr = mr >> 1;
        e  = e + 11'sd1;
      end
      if (e >= 11'sd255) return {s, 8'hFF, 23'b0};
      return {s, e[7:0], mr[22:0]};
    end
    // below the normal range: round on the subnormal grid; keep only a carry into 2^-126
    sh = 11'(11'sd1 - e);                                     // >= 1
    if (sh >= 11'd27) return {s, 31'b0};
    x    = {m24, g, st};
    xs   = x >> sh[4:0];
    lost = x & ~(27'h7FFFFFF << sh[4:0]);
    q    = xs[25:2];
    qg   = xs[1];
    qst  = xs[0] | (lost != 0);
    inc  = qg & (qst | q[0]);
    if ((25'(q) + 25'(inc)) == 25'h800000) return {s, 8'd1, 23'b0};   // 2^-126 survives
    return {s, 31'b0};
  endfunction

  // ---------------------------------------------------------------- add
  function automatic logic [31:0] fadd(input logic [31:0] a_in, input logic [31:0] b_in);
    logic [31:0] a, b, t;
    logic [23:0] ma, mb;
    logic [26:0] A, B, mb3;
    logic [27:0] S;
    logic        sub, s, g, st, inc;
    logic signed [10:0] e;
    logic [24:0] mr;
    logic [23:0] m24;
    logic [7:0]  d;
    logic [37:0] nz;
    a = ftz(a_in);
    b = ftz(b_in);
    if (is_nan(a) || is_nan(b)) return CANON_NAN;
    if (is_inf(a) && is_inf(b)) return (a[31] != b[31]) ? CANON_NAN : a;
    if (is_inf(a)) return a;
    if (is_inf(b)) return b;
    if (a[30:0] == 0 && b[30:0] == 0) return {a[31] & b[31], 31'b0};   // -0 only from -0 + -0
    if (a[30:0] == 0) return b;
    if (b[30:0] == 0) return a;
    if (b[30:0] > a[30:0]) begin                                      // |a| >= |b|
      t = a; a = b; b = t;
    end
    sub = a[31] ^ b[31];
    s   = a[31];
    ma  = {1'b1, a[22:0]};
    mb  = {1'b1, b[22:0]};
    d   = a[30:23] - b[30:23];                                        // >= 0 after the swap
    A   = {ma, 3'b000};
    mb3 = {mb, 3'b000};
    if (d >= 8'd27) begin
      B = 27'd1;                                                      // only a sticky bit
    end else begin
      B    = mb3 >> d[4:0];
      B[0] = B[0] | ((mb3 & ~(27'h7FFFFFF << d[4:0])) != 0);
    end
    S = sub ? ({1'b0, A} - {1'b0, B}) : ({1'b0, A} + {1'b0, B});
    if (S == 0) return 32'h00000000;                                  // exact cancellation: +0
    e = $signed({3'b0, a[30:23]});
    if (S[27]) begin
      S = {1'b0, S[27:2], S[1] | S[0]};
      e = e + 11'sd1;
    end else begin
      nz = lzc_norm32({S[26:0], 5'b0});                               // S[26:0] != 0 here
      S  = {1'b0, nz[31:5]};
      e  = e - $signed({5'b0, nz[37:32]});
    end
    // S[26:3] = mantissa (with the leading 1), S[2] guard, S[1:0] round+sticky
    if (e <= 11'sd0) return {s, 31'b0};                               // subnormal sum: exact, flushed
    m24 = S[26:3];
    g   = S[2];
    st  = S[1] | S[0];
    inc = g & (st | m24[0]);
    mr  = {1'b0, m24} + 25'(inc);
    if (mr[24]) begin
      mr = mr >> 1;
      e  = e + 11'sd1;
    end
    if (e >= 11'sd255) return {s, 8'hFF, 23'b0};
    return {s, e[7:0], mr[22:0]};
  endfunction

  // ---------------------------------------------------------------- int32 -> fp32
  // i2f in two parts (the GEMM epilogue registers between them): the sign and magnitude, then the
  // normalize and round of a nonzero magnitude
  function automatic logic [31:0] i2f_tail(input logic s, input logic [31:0] u);
    logic        g, st, inc;
    logic [31:0] w;
    logic [24:0] mr;
    logic [23:0] m24;
    logic [37:0] nz;
    logic [7:0]  e;
    nz = lzc_norm32(u);                                               // w = u << (31 - msb)
    w  = nz[31:0];
    e  = 8'(8'd158 - {2'b0, nz[37:32]});                              // 127 + msb
    m24 = w[31:8];
    g   = w[7];
    st  = w[6:0] != 0;
    inc = g & (st | m24[0]);
    mr  = {1'b0, m24} + 25'(inc);
    if (mr[24]) begin
      mr = mr >> 1;
      e  = e + 8'd1;
    end
    return {s, e, mr[22:0]};
  endfunction

  function automatic logic [31:0] i2f(input logic signed [31:0] v);
    if (v == 0) return 32'h0;
    return i2f_tail(v[31], v[31] ? 32'(-v) : 32'(v));
  endfunction

  // ---------------------------------------------------------------- activations (on ftz values)
  // ------------------------------------------------------------------ v0.3 math primitives
  // Fixed sequences of fmul / fadd plus exact bit operations; must match numerics.py step by step.
  localparam logic [31:0] F_ONE = 32'h3F800000, F_TWO = 32'h40000000, F_HALF = 32'h3F000000;
  localparam logic [31:0] LOG2E = 32'h3FB8AA3B, NEG_LN2_HI = 32'hBF317200, NEG_LN2_LO = 32'hB5BFBE8E;
  localparam logic [31:0] EXP_C0 = 32'h3AB60B61, EXP_C1 = 32'h3C088889, EXP_C2 = 32'h3D2AAAAB,
                          EXP_C3 = 32'h3E2AAAAB, EXP_C4 = 32'h3F000000, EXP_C5 = 32'h3F800000,
                          EXP_C6 = 32'h3F800000;
  localparam logic [31:0] RECIP_MAGIC = 32'h7EF311C7, RSQRT_MAGIC = 32'h5F3759DF;
  localparam logic [31:0] GELU_A = 32'h3D372713, GELU_C = 32'h3F4C422A;

  function automatic logic [31:0] fneg(input logic [31:0] x);
    return {~x[31], x[30:0]};
  endfunction

  // y (positive normal) with e added to its biased exponent: >= 255 -> +inf, <= 0 -> +0
  function automatic logic [31:0] set_exp(input logic [31:0] y, input int e);
    int ne;
    ne = int'(y[30:23]) + e;
    if (ne >= 255) return 32'h7F800000;
    if (ne <= 0) return 32'h0;
    return {1'b0, 8'(ne), y[22:0]};
  endfunction

  // rint (ties to even) of a finite fp32 value, clamped to [-200, 200]
  function automatic int rint_clamp(input logic [31:0] t);
    logic [4:0]  k;
    logic [31:0] w;
    logic [8:0]  ip, n;
    logic        g, st;
    if (t[30:23] < 8'd126) return 0;                                  // |t| < 0.5
    if (t[30:23] >= 8'd135) return t[31] ? -200 : 200;                // |t| >= 256
    // |t| = m24 * 2^(e - 150), e in 126 .. 134: shift the 24-bit significand so that the integer
    // part lands in w[31:23] and the guard bit in w[22]; ties to even via guard and sticky
    k  = 5'(t[30:23] - 8'd126);                                       // 0 .. 8
    w  = {8'b0, 1'b1, t[22:0]} << k;                                  // integer part w[31:24]
    ip = {1'b0, w[31:24]};
    g  = w[23];
    st = w[22:0] != 0;
    n  = ip + 9'(g & (st | ip[0]));
    if (n > 9'd200) n = 9'd200;
    return t[31] ? -int'(n) : int'(n);
  endfunction

  function automatic logic [31:0] exp_f(input logic [31:0] x_in);
    logic [31:0] x, nf, r, p;
    int n;
    x = ftz(x_in);
    if (is_nan(x)) return CANON_NAN;
    if (!x[31] && x[30:0] >= 31'h42B20000) return 32'h7F800000;      // x >= 89 (and +inf)
    if (x[31] && x[30:0] >= 31'h42D00000) return 32'h0;               // x <= -104 (and -inf)
    n  = rint_clamp(fmul(x, LOG2E));
    nf = i2f(n);
    r  = fadd(fadd(x, fmul(nf, NEG_LN2_HI)), fmul(nf, NEG_LN2_LO));
    p  = EXP_C0;
    p  = fadd(fmul(p, r), EXP_C1);
    p  = fadd(fmul(p, r), EXP_C2);
    p  = fadd(fmul(p, r), EXP_C3);
    p  = fadd(fmul(p, r), EXP_C4);
    p  = fadd(fmul(p, r), EXP_C5);
    p  = fadd(fmul(p, r), EXP_C6);
    return set_exp(p, n);
  endfunction

  function automatic logic [31:0] recip_f(input logic [31:0] x_in);
    logic [31:0] x, m, y;
    x = ftz(x_in);
    if (is_nan(x)) return CANON_NAN;
    if (x[30:0] == 0) return {x[31], 31'h7F800000};
    if (is_inf(x)) return {x[31], 31'h0};
    m = {1'b0, 8'd127, x[22:0]};
    y = RECIP_MAGIC - m;
    for (int i = 0; i < 3; i++) y = fadd(y, fmul(y, fadd(F_ONE, fneg(fmul(m, y)))));
    y = set_exp(y, -(int'(x[30:23]) - 127));
    return {x[31], y[30:0]};
  endfunction

  function automatic logic [31:0] rsqrt_f(input logic [31:0] x_in);
    logic [31:0] x, m, y;
    int E, k;
    x = ftz(x_in);
    if (is_nan(x)) return CANON_NAN;
    if (x[30:0] == 0) return {x[31], 31'h7F800000};
    if (x[31]) return CANON_NAN;
    if (is_inf(x)) return 32'h0;
    E = int'(x[30:23]) - 127;
    k = E >>> 1;
    m = {1'b0, 8'(127 + (E & 1)), x[22:0]};
    y = RSQRT_MAGIC - (m >> 1);
    for (int i = 0; i < 3; i++)
      y = fadd(y, fmul(y, fadd(F_HALF, fneg(fmul(F_HALF, fmul(fmul(m, y), y))))));
    return set_exp(y, -k);
  endfunction

  function automatic logic [31:0] gelu_f(input logic [31:0] x_in);
    logic [31:0] x, x3, u, t;
    x  = ftz(x_in);
    x3 = fmul(fmul(x, x), x);
    u  = fmul(GELU_C, fadd(x, fmul(GELU_A, x3)));
    t  = fadd(F_ONE, fneg(fmul(F_TWO, recip_f(fadd(exp_f(fmul(F_TWO, u)), F_ONE)))));
    return fmul(fmul(F_HALF, x), fadd(F_ONE, t));
  endfunction

  function automatic logic [31:0] silu_f(input logic [31:0] x_in);
    logic [31:0] x;
    x = ftz(x_in);
    return fmul(x, recip_f(fadd(F_ONE, exp_f(fneg(x)))));
  endfunction

  // v0.4
  localparam logic [31:0] LOG1P_C [9] = '{32'h3D70F0F1, 32'h3D888889, 32'h3D9D89D9, 32'h3DBA2E8C,
                                         32'h3DE38E39, 32'h3E124925, 32'h3E4CCCCD, 32'h3EAAAAAB,
                                         32'h3F800000};               // 1/17, 1/15, ..., 1/3, 1
  localparam logic [31:0] SP_HI = 32'h41A00000, Q127 = 32'h42FE0000, INV127 = 32'h3C010204;

  function automatic logic [31:0] fabs(input logic [31:0] x);
    logic [31:0] f;
    f = ftz(x);
    return {1'b0, f[30:0]};
  endfunction

  function automatic logic [31:0] sigmoid_f(input logic [31:0] x_in);
    return recip_f(fadd(F_ONE, exp_f(fneg(ftz(x_in)))));
  endfunction

  // softplus: x > 20 -> x; else relu(x) + (2 s) * p(s * s), s = t * recip(t + 2), t = exp(-|x|)
  function automatic logic [31:0] softplus_f(input logic [31:0] x_in);
    logic [31:0] x, t, sv, z, p, rl;
    x = ftz(x_in);
    if (is_nan(x)) return CANON_NAN;
    t  = exp_f(fneg(fabs(x)));
    sv = fmul(t, recip_f(fadd(t, F_TWO)));
    z  = fmul(sv, sv);
    p  = LOG1P_C[0];
    for (int i = 1; i < 9; i++) p = fadd(fmul(p, z), LOG1P_C[i]);
    rl = (!x[31] && x[30:0] != 0) ? x : 32'h0;
    if (!x[31] && x[30:0] > SP_HI[30:0]) return x;
    return fadd(rl, fmul(fmul(F_TWO, sv), p));
  endfunction

  // act: 0 identity, 1 relu, 2 relu6, 3 GELU, 4 SiLU, 5 sigmoid, 6 softplus, 7 exp (v0.4)
  function automatic logic [31:0] act_fn(input logic [31:0] x_in, input logic [2:0] act);
    logic [31:0] x, r;
    x = ftz(x_in);
    if (act == 3'd0) return x;
    if (act == 3'd3) return gelu_f(x);
    if (act == 3'd4) return silu_f(x);
    if (act == 3'd5) return sigmoid_f(x);
    if (act == 3'd6) return softplus_f(x);
    if (act == 3'd7) return exp_f(x);
    r = (!x[31] && x[30:0] != 0 && !is_nan(x)) ? x : 32'h0;          // relu(-0) = relu(NaN) = +0
    if (act == 3'd1) return r;
    return (r > 32'h40C00000) ? 32'h40C00000 : r;                     // relu6: positive bits order
  endfunction

  // identity / relu / relu6 only (act 0, 1, 2): the one-stage activation path. act_fn with a
  // runtime act would also build GELU and SiLU (its other branches) in combinational logic.
  function automatic logic [31:0] act_simple(input logic [31:0] x_in, input logic [1:0] act);
    logic [31:0] x, r;
    x = ftz(x_in);
    if (act == 2'd0) return x;
    r = (!x[31] && x[30:0] != 0 && !is_nan(x)) ? x : 32'h0;
    if (act == 2'd1) return r;
    return (r > 32'h40C00000) ? 32'h40C00000 : r;
  endfunction

  function automatic logic [15:0] to_bf16(input logic [31:0] x_in);
    logic [31:0] x;
    logic [32:0] t;
    x = ftz(x_in);
    if (is_nan(x)) return 16'h7FC0;
    t = {1'b0, x} + 33'h7FFF + 33'(x[16]);
    return t[31:16];
  endfunction

  // rint(fmul(x, inv)), ties to even, saturated to [-127, 127], NaN -> 0
  function automatic logic [7:0] q8(input logic [31:0] x, input logic [31:0] inv);
    return q8_tail(fmul(x, inv));
  endfunction

  // the part of q8 after the multiply (the pipelined epilogue registers fmul's result first)
  function automatic logic [7:0] q8_tail(input logic [31:0] y);
    logic [23:0] m24, q;
    logic [23:0] lost;
    logic        g, st, inc;
    logic [8:0]  r;
    int          e, sh;
    if (is_nan(y)) return 8'd0;
    e = int'(y[30:23]);
    if (e < 126) return 8'd0;                                         // |y| < 0.5 (and zeros)
    if (e >= 134) return y[31] ? 8'h81 : 8'h7F;                       // |y| >= 128, inf
    m24  = {1'b1, y[22:0]};
    sh   = 150 - e;                                                   // 17 .. 24
    q    = m24 >> sh;
    g    = m24[sh - 1];
    lost = m24 & ~(24'hFFFFFF << (sh - 1));
    st   = (lost != 0);
    inc  = g & (st | q[0]);
    r    = 9'(q) + 9'(inc);
    if (r > 9'd127) r = 9'd127;
    return y[31] ? 8'(-r) : 8'(r);
  endfunction

  // max for the row ops (contract: NaN if either is NaN; equal values: sign is the AND)
  function automatic logic [31:0] fmax(input logic [31:0] a_in, input logic [31:0] b_in);
    logic [31:0] a, b;
    a = ftz(a_in); b = ftz(b_in);
    if (is_nan(a) || is_nan(b)) return 32'h7FC00000;
    if (a == b || (a[30:0] == 0 && b[30:0] == 0)) return {a[31] & b[31], a[30:0]};
    // ordered compare of two non-NaN, unequal values
    if (a[31] != b[31]) return a[31] ? b : a;
    if (!a[31]) return (a[30:0] > b[30:0]) ? a : b;
    return (a[30:0] < b[30:0]) ? a : b;
  endfunction

endpackage
