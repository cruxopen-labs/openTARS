// TARS-mini pipelined fp32 primitives for a 1 GHz clock: fadd and fmul cut into three stages.
// Each stage is a function from the previous stage's packed state; the composition equals
// tm_fp_pkg::fadd / fmul bit for bit (tests/test_rtl_fp.py drives the pipelined modules too).
// Cut points (ASAP7 RVT TT, pre-layout, budget 650 ps per stage; tools/timing):
//   fadd: 1 specials, |a| >= |b| swap, exponent difference, alignment shift and sticky
//         2 add / subtract, cancellation, leading-zero normalization
//         3 round to nearest even, exponent adjust, overflow, result select
//   fmul: 1 specials, exponent sum, 12 radix-4 partial products compressed (carry-save) to two
//         2 the final carry-propagate add, normalize, guard and sticky
//         3 round, exponent adjust, overflow, the subnormal grid, result select
package tm_fpp_pkg;
  import tm_fp_pkg::*;

  // ---------------------------------------------------------------- fadd
  typedef struct packed {
    logic        done;          // result decided in stage 1 (specials, zeros)
    logic [31:0] res;
    logic        sub, s;
    logic [7:0]  ea;            // exponent of the larger operand
    logic [26:0] A, B;          // aligned significands with guard, round, sticky
  } fadd_s1_t;

  typedef struct packed {
    logic        done;
    logic [31:0] res;
    logic        s;
    logic signed [10:0] e;
    logic [26:0] S;             // normalized: S[26:3] mantissa with the leading 1, S[2:0] g r st
  } fadd_s2_t;

  function automatic fadd_s1_t fadd_st1(input logic [31:0] a_in, input logic [31:0] b_in);
    fadd_s1_t o;
    logic [31:0] a, b, t;
    logic [26:0] mb3;
    logic [7:0]  d;
    o = '0;
    a = ftz(a_in);
    b = ftz(b_in);
    if (is_nan(a) || is_nan(b))           begin o.done = 1'b1; o.res = CANON_NAN; end
    else if (is_inf(a) && is_inf(b))      begin o.done = 1'b1; o.res = (a[31] != b[31]) ? CANON_NAN : a; end
    else if (is_inf(a))                   begin o.done = 1'b1; o.res = a; end
    else if (is_inf(b))                   begin o.done = 1'b1; o.res = b; end
    else if (a[30:0] == 0 && b[30:0] == 0) begin o.done = 1'b1; o.res = {a[31] & b[31], 31'b0}; end
    else if (a[30:0] == 0)                begin o.done = 1'b1; o.res = b; end
    else if (b[30:0] == 0)                begin o.done = 1'b1; o.res = a; end
    if (b[30:0] > a[30:0]) begin t = a; a = b; b = t; end
    o.sub = a[31] ^ b[31];
    o.s   = a[31];
    o.ea  = a[30:23];
    d     = a[30:23] - b[30:23];
    o.A   = {1'b1, a[22:0], 3'b000};
    mb3   = {1'b1, b[22:0], 3'b000};
`ifdef TM_FPGA
    // FPGA: the shift as a multiply (DSP): mb << (26 - d) puts mb3 >> d in [49:23], the lost bits below
    begin
      logic [50:0] t;
      t = 51'({1'b1, b[22:0]}) * 51'(27'd1 << (5'd26 - d[4:0]));
      if (d >= 8'd27) o.B = 27'd1;
      else o.B = {t[49:24], t[23] | (t[22:0] != 0)};
    end
`else
    if (d >= 8'd27) o.B = 27'd1;
    else begin
      o.B    = mb3 >> d[4:0];
      o.B[0] = o.B[0] | ((mb3 & ~(27'h7FFFFFF << d[4:0])) != 0);
    end
`endif
    return o;
  endfunction

  function automatic fadd_s2_t fadd_st2(input fadd_s1_t i);
    fadd_s2_t o;
    logic [27:0] S;
    logic [37:0] nz;
    o.done = i.done; o.res = i.res; o.s = i.s;
    S   = i.sub ? ({1'b0, i.A} - {1'b0, i.B}) : ({1'b0, i.A} + {1'b0, i.B});
    o.e = $signed({3'b0, i.ea});
    if (!i.done && S == 0) begin o.done = 1'b1; o.res = 32'h0; end     // exact cancellation: +0
    if (S[27]) begin
      o.S = {S[27:2], S[1] | S[0]};
      o.e = o.e + 11'sd1;
    end else begin
      nz  = lzc_norm32({S[26:0], 5'b0});
      o.S = nz[31:5];
      o.e = o.e - $signed({5'b0, nz[37:32]});
    end
    return o;
  endfunction

  function automatic logic [31:0] fadd_st3(input fadd_s2_t i);
    logic [24:0] mr;
    logic        inc;
    logic signed [10:0] e;
    if (i.done) return i.res;
    e = i.e;
    if (e <= 11'sd0) return {i.s, 31'b0};                                // subnormal sum: flushed
    inc = i.S[2] & ((i.S[1] | i.S[0]) | i.S[3]);
    mr  = {1'b0, i.S[26:3]} + 25'(inc);
    if (mr[24]) begin mr = mr >> 1; e = e + 11'sd1; end
    if (e >= 11'sd255) return {i.s, 8'hFF, 23'b0};
    return {i.s, e[7:0], mr[22:0]};
  endfunction

  // ---------------------------------------------------------------- fmul
  typedef struct packed {
    logic        done;
    logic [31:0] res;
    logic        s;
    logic signed [10:0] e;      // ea + eb - 127
    logic [47:0] ps, pc;        // ma * mb in carry-save form: ps + pc
  } fmul_s1_t;

  typedef struct packed {
    logic        done;
    logic [31:0] res;
    logic        s;
    logic signed [10:0] e;
    logic [23:0] m24;
    logic        g, st;
  } fmul_s2_t;

  // 3:2 compressor: x + y + z = s + c
  function automatic logic [95:0] csa48(input logic [47:0] x, input logic [47:0] y, input logic [47:0] z);
    return {x ^ y ^ z, ((x & y) | (x & z) | (y & z)) << 1};
  endfunction

  // ma * mb as two vectors whose sum is the product (stage 2 adds them).
`ifdef TM_FPGA
  // FPGA: the product itself and zero; the multiply maps to DSP slices
  function automatic logic [95:0] mul24_cs(input logic [23:0] ma, input logic [23:0] mb);
    return {48'(ma) * 48'(mb), 48'd0};
  endfunction
`else
  // Digits mb[2k+1:2k] select 0, ma, 2 ma or 3 ma (one 26-bit add), then a five-level 3:2 tree
  // reduces the twelve rows to two
  function automatic logic [95:0] mul24_cs(input logic [23:0] ma, input logic [23:0] mb);
    logic [47:0] pp [12];
    logic [47:0] m3;
    logic [95:0] t0, t1, t2, t3, t4, t5, t6, t7, t8, t9;
    m3 = 48'(ma) + (48'(ma) << 1);
    for (int k = 0; k < 12; k++)
      case (mb[2 * k +: 2])
        2'd0: pp[k] = 48'd0;
        2'd1: pp[k] = 48'(ma) << (2 * k);
        2'd2: pp[k] = 48'(ma) << (2 * k + 1);
        default: pp[k] = m3 << (2 * k);
      endcase
    t0 = csa48(pp[0], pp[1], pp[2]);   t1 = csa48(pp[3], pp[4], pp[5]);
    t2 = csa48(pp[6], pp[7], pp[8]);   t3 = csa48(pp[9], pp[10], pp[11]);          // 12 -> 8
    t4 = csa48(t0[95:48], t0[47:0], t1[95:48]); t5 = csa48(t1[47:0], t2[95:48], t2[47:0]);   // -> 6
    t6 = csa48(t4[95:48], t4[47:0], t5[95:48]); t7 = csa48(t5[47:0], t3[95:48], t3[47:0]);   // -> 4
    t8 = csa48(t6[95:48], t6[47:0], t7[95:48]);                                      // -> 3
    t9 = csa48(t8[95:48], t8[47:0], t7[47:0]);                                       // -> 2
    return t9;
  endfunction
`endif

  function automatic fmul_s1_t fmul_st1(input logic [31:0] a_in, input logic [31:0] b_in);
    fmul_s1_t o;
    logic [31:0] a, b;
    logic [23:0] ma, mb;
    o = '0;
    a = ftz(a_in);
    b = ftz(b_in);
    o.s = a[31] ^ b[31];
    if (is_nan(a) || is_nan(b)) begin o.done = 1'b1; o.res = CANON_NAN; end
    else if (is_inf(a) || is_inf(b)) begin
      o.done = 1'b1;
      o.res  = (a[30:0] == 0 || b[30:0] == 0) ? CANON_NAN : {o.s, 8'hFF, 23'b0};
    end else if (a[30:0] == 0 || b[30:0] == 0) begin o.done = 1'b1; o.res = {o.s, 31'b0}; end
    ma    = {1'b1, a[22:0]};
    mb    = {1'b1, b[22:0]};
    o.e   = $signed({3'b0, a[30:23]}) + $signed({3'b0, b[30:23]}) - 11'sd127;
    {o.ps, o.pc} = mul24_cs(ma, mb);
    return o;
  endfunction

  function automatic fmul_s2_t fmul_st2(input fmul_s1_t i);
    fmul_s2_t o;
    logic [47:0] p;
    o.done = i.done; o.res = i.res; o.s = i.s; o.e = i.e;
    p = i.ps + i.pc;                                                     // in [2^46, 2^48)
    if (p[47]) begin
      o.m24 = p[47:24]; o.g = p[23]; o.st = |p[22:0]; o.e = i.e + 11'sd1;
    end else begin
      o.m24 = p[46:23]; o.g = p[22]; o.st = |p[21:0];
    end
    return o;
  endfunction

  function automatic logic [31:0] fmul_st3(input fmul_s2_t i);
    logic        inc, qg, qst;
    logic signed [10:0] e;
    logic [24:0] mr;
    logic [26:0] x, xs, lost;
    logic [23:0] q;
    logic [10:0] sh;
    if (i.done) return i.res;
    e = i.e;
    if (e >= 11'sd1) begin
      inc = i.g & (i.st | i.m24[0]);
      mr  = {1'b0, i.m24} + 25'(inc);
      if (mr[24]) begin mr = mr >> 1; e = e + 11'sd1; end
      if (e >= 11'sd255) return {i.s, 8'hFF, 23'b0};
      return {i.s, e[7:0], mr[22:0]};
    end
    sh = 11'(11'sd1 - e);
    if (sh >= 11'd27) return {i.s, 31'b0};
    x    = {i.m24, i.g, i.st};
    xs   = x >> sh[4:0];
    lost = x & ~(27'h7FFFFFF << sh[4:0]);
    q    = xs[25:2];
    qg   = xs[1];
    qst  = xs[0] | (lost != 0);
    inc  = qg & (qst | q[0]);
    if ((25'(q) + 25'(inc)) == 25'h800000) return {i.s, 8'd1, 23'b0};
    return {i.s, 31'b0};
  endfunction

endpackage
