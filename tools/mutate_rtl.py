"""Mutation check for the verifier: inject known bugs into the RTL one at a time and require the
test suite to catch every one. A mutant that survives means the tests have a blind spot.

    python3 tools/mutate_rtl.py            # all mutants, the default test subset
    python3 tools/mutate_rtl.py --list

Each mutant is (file, original text, replacement, what it breaks). The original text must occur
exactly once, so a stale mutant fails loudly instead of silently testing nothing.
"""
from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = "rtl/tm_core.sv"
FP_PKG = "rtl/tm_fp_pkg.sv"
VPU = "rtl/tm_vpu.sv"
PE = "rtl/tm_pe.sv"
XPU = "rtl/tm_xpu.sv"

# Each mutant runs the subset that targets its logic (a 5th element), else DEFAULT_TESTS; -x stops
# at the first failure, so a caught mutant costs one build and the tests up to the first failure.
FP_T = ["tests/test_rtl_fp.py"]
LOADER_T = ["tests/test_rtl.py", "-k", "loader"]
HOLD_T = ["tests/test_rtl.py", "-k", "hold"]
# (test_hold_filling_tile_waits_for_its_rows matches "hold" too)
ROW_T = ["tests/test_rtl.py", "-k", "compiled_rowops or queue_end_waits or errors_on_rtl or "
         "fuzz_random_queues and (8-8 or 3-5) or hold"]
ERR_T = ["tests/test_rtl.py", "-k", "errors_on_rtl"]
GV_T = ["tests/test_rtl_v04.py", "-k", "gemv"]
V04_T = ["tests/test_rtl_v04.py", "-k", "not qwen and not 4x8 and not 3x5 and not coverage"]
CONV_T = ["tests/test_rtl.py", "-k", "errors_on_rtl or compiled_conv and 8-8 or fuzz_random_queues and (8-8 or 3-5)"]

MUTANTS = [
    (CORE, "(cb >= 4 && cb <= 7) ? 8'h00 :", "1'b0 ? 8'h00 :", "CRC covers its own field"),
    (CORE, "k = int'(cnt) - i;", "k = int'(cnt);", "rows not skewed"),
    (CORE, "f_ldy  = (d_ldy16 != 0) ? 32'(d_ldy16) * 32'd16 : f_yrow;", "f_ldy  = f_yrow;",
     "output stride ignored"),
    (CORE, "f_lda  = (d_lda16 != 0) ? 32'(d_lda16) * 32'd16 : f_arow;", "f_lda  = f_arow;",
     "input stride ignored"),
    (CORE, "assign mw_strb = wr_bstrb;", "assign mw_strb = 16'hFFFF;", "the segment writer ignores its strobes"),
    (CORE, "wr_free = !wr_busy || (wr_acc && wr_b == wr_nb - 8'd1);",
     "wr_free = !wr_busy || (wr_b == wr_nb - 8'd1);", "writer takes the next row before its last beat is accepted"),
    (CORE, "assign a_hold  = x_reuse && !x_minner && n0 != 0;", "assign a_hold  = x_reuse && !x_minner;",
     "held A read on the tile that is still filling the hold buffer", HOLD_T),
    (CORE, "assign w_hold  = x_reuse &&  x_minner && m0 != 0;", "assign w_hold  = x_reuse &&  x_minner && n0 != 0;",
     "held W keyed on the wrong loop index", HOLD_T),
    (CORE, "x_reuse  <= 32'(d_k >> 4) <= 32'(KR);", "x_reuse  <= 1'b1;",
     "hold buffer used when K does not fit", HOLD_T),
    (CORE, "vec_need = !(vec_ok[l_tag] && vec_n0[l_tag] == ln0);", "vec_need = !vec_ok[l_tag];",
     "scale/bias vectors kept across column blocks"),
    (CORE, "vec_ok <= '{default: 1'b0};\n            m0 <= 0;", "m0 <= 0;", "scale/bias vectors kept across descriptors"),
    (CORE, "l_out < (LO + 1)'(OT)", "l_out <= (LO + 1)'(OT)", "one read more than the credit limit", LOADER_T),
    ("rtl/tm_pe.sv", "    if (en) begin\n      a_out", "    if (1'b1) begin\n      a_out",
     "array keeps shifting while the feed stalls", LOADER_T),
    (CORE, "a_f_edge[i] = (k == 0);", "a_f_edge[i] = 1'b0;", "a tile's first product does not restart its bank"),
    # double-buffered accumulators: the epilogue engine runs beside the next tile's feed
    (CORE, "a_t_edge[i] = f_tag;", "a_t_edge[i] = 1'b0;", "every tile accumulates in the same bank"),
    (CORE, "e_busy <= 1'b1; e_bank <= f_tag;", "e_busy <= 1'b1; e_bank <= !f_tag;", "epilogue reads the other bank"),
    (CORE, "e_adv <= e_adv + 32'(arr_en);", "e_adv <= e_adv + 32'd1;",
     "epilogue counts cycles, not array advances"),
    # (dropping !e_busy from the GEMM end is an equivalent mutant: after the last handoff the
    # epilogue launches a row in the next cycle and every later row in the cycle the previous
    # one's last beat is accepted, so the writer stays busy until e_busy has cleared)
    (CORE, "          if (!e_busy) begin\n            e_busy <= 1'b1;", "          if (1'b1) begin\n            e_busy <= 1'b1;",
     "a tile is handed to a busy epilogue engine", ["tests/test_rtl.py", "-k", "epilogue_busy"]),
    # v0.3 math primitives and activations
    (FP_PKG, "p  = fadd(fmul(p, r), EXP_C3);", "p  = fadd(fmul(p, r), EXP_C2);", "exp polynomial coefficient", FP_T),
    (FP_PKG, "for (int i = 0; i < 3; i++) y = fadd(y, fmul(y, fadd(F_ONE",
     "for (int i = 0; i < 2; i++) y = fadd(y, fmul(y, fadd(F_ONE", "recip: two Newton steps, not three", FP_T),
    (FP_PKG, "k = E >>> 1;", "k = E / 2;", "rsqrt halves a negative odd exponent toward zero", FP_T),
    (FP_PKG, "return fmul(x, recip_f(fadd(F_ONE, exp_f(fneg(x)))));",
     "return fmul(x, recip_f(fadd(F_ONE, exp_f(x))));", "SiLU uses e^x instead of e^-x", FP_T),
    # (2 * recip(d) -> recip(d / 2) is an equivalent mutant: recip normalizes the mantissa, so
    # scaling by a power of two commutes with it bit for bit for every d = e^2u + 1 >= 1)
    (FP_PKG, "u  = fmul(GELU_C, fadd(x, fmul(GELU_A, x3)));",
     "u  = fadd(fmul(GELU_C, x), fmul(fmul(GELU_C, GELU_A), x3));", "GELU's c distributed over the sum", FP_T),
    (CORE, "f_act  = {d_epi[5], d_epi[1:0]};", "f_act  = {1'b0, d_epi[1:0]};", "EPI[5] ignored"),
    (CORE, "(f_row && (d_ctrl[15:3] != 0 || d_ctrl[2:0] > 3'd4 || f_act != 3'd0 ||",
     "(f_row && (d_ctrl[15:3] != 0 || d_ctrl[2:0] > 3'd4 ||", "activation accepted on a row op", ERR_T),
    # v0.3 row ops (the fmax zero-sign rule has no mutant: exp(x - mx) is the same for mx = +0 and
    # -0, so no output can show which zero the max returned)
    (CORE, "(x_bias ? 2'd3 : 2'd2) : 2'd1;", "2'd2 : 2'd1;", "uncached output pass skips beta's read", ROW_T),
    (CORE, "            lane <= '{default: ((x_fn == 2'd0) ? 32'hFF800000 : 32'h0)};",
     "            lane <= '{default: 32'h0};", "softmax max starts from 0, not -inf", ROW_T),
    (CORE, "a_lane <= '{default: (d_ctrl[1:0] == 2'd0) ? 32'hFF800000 : 32'h0};", "a_lane <= '{default: 32'h0};",
     "overlapped softmax: the first row's max starts from 0", ROW_T),
    (CORE, "a_lane <= '{default: ((x_fn == 2'd0) ? 32'hFF800000 : 32'h0)};", "a_lane <= '{default: 32'h0};",
     "overlapped softmax: later rows' max starts from 0", ROW_T),
    # row ops: VG-wide hold-buffer passes and overlapped rows
    (CORE, "v_rbase   = (x_ov && b_buf) ? x_nbr : 32'd0;", "v_rbase   = 32'd0;",
     "the later passes read row region 0 whatever the row", ROW_T),
    (CORE, "? 32'(d_n) >> 1 : 32'(d_n) >> 2;", "? 32'(d_n) >> 2 : 32'(d_n) >> 2;",
     "overlapped gamma placed over the second row region", ROW_T),
    # cross-tile prefetch: the loader one tile ahead of the feed
    (CORE, "l_base <= LB'(l_base + LB'(x_k >> 4)); l_st <= 1'b1;", "l_st <= 1'b1;",
     "the next tile's chunks loaded into buffers from chunk 0"),
    (CORE, "f_base <= f_bnx;", "f_base <= 0;", "the feed reads every tile from buffer 0"),
    (CORE, "l_adv   = l_tdone && l_has_next && (l_tidx < f_tidx ||", "l_adv   = l_tdone && l_has_next && (1'b1 ||",
     "the loader runs more than one tile ahead"),
    (CORE, "&& (a_hold || w_hold))));", "&& 1'b0)));",
     "prefetch refills the hold buffer while the current tile reads it", HOLD_T),
    (CORE, "svb[fq_b[fq_h][0]][fq_i[fq_h]] <= mr_rdata;", "svb[0][fq_i[fq_h]] <= mr_rdata;",
     "scale vectors not double-buffered"),
    (CORE, "p[0] ? (!lw_hold &&", "p[0] ? (!w_hold &&", "loader skips W by the feed's hold flag", HOLD_T),
    (CORE, "assign la_hold  = x_reuse && !x_minner && ln0 != 0;", "assign la_hold  = x_reuse && !x_minner;",
     "loader skips A on the tile that fills the hold buffer", HOLD_T),
    (CORE, "lt_cols  = (x_n - ln0 < 32'(C)) ? x_n - ln0 : 32'(C);", "lt_cols  = t_cols;",
     "the loader sizes the vectors by the feed's tile"),
    # v0.3 conv (implicit GEMM)
    # P8 (1 GHz pipelines): the stale P5-P7 mutants rewritten against the pipelined RTL, plus
    # pipeline-alignment bugs (a delay off by one, a latency tap read early)
    (CORE, "assign y1 = x_bias ? y1a : y0d;", "assign y1 = y0d;", "epilogue bias dropped"),
    (CORE, "cv_first = cv & (~cv + 8'd1);", "cv_first = cv & (~cv + 8'd1); if (cv[6]) cv_first = 8'h40;",
     "E_ALIGN checked before the earlier rules"),
    (CORE, "if (!wr_busy && !e_busy && oq_cnt == 0) begin qi <= qi + 1;",
     "if (!wr_busy && !e_busy) begin qi <= qi + 1;", "GEMM descriptor ends with rows still in the epilogue lanes"),
    (CORE, "if (x_minner ? n0 >= x_n : m0 >= x_m) begin\n            if (!wr_busy",
     "if (m0 >= x_m) begin\n            if (!wr_busy", "m-inner loop ends after the first column block", HOLD_T),
    (CORE, "cv[7]  = hi_a > 66'(MEM_BYTES) || hi_y > 66'(MEM_BYTES) ||",
     "cv[7]  = hi_a >= 66'(MEM_BYTES) || hi_y >= 66'(MEM_BYTES) ||", "range check off by one"),
    (CORE, "e_emerge = e_vsr[x_gemm ? e_lat : x_xpu ? xp_lat : v_olat];", "e_emerge = e_vsr[x_gemm ? e_lat - 32'd1 : x_xpu ? xp_lat : v_olat];",
     "output queue filled one cycle before the epilogue lanes are out"),
    (CORE, "e_emerge = e_vsr[x_gemm ? e_lat : x_xpu ? xp_lat : v_olat];", "e_emerge = e_vsr[x_gemm ? e_lat : x_xpu ? xp_lat : v_olat + 32'd1];",
     "output queue filled one cycle after the row-engine lanes are out", ROW_T),
    (FP_PKG, "n  = ip + 9'(g & (st | ip[0]));", "n  = ip + 9'(g);", "exp's rint rounds ties away from zero", FP_T),
    (CORE, "lane_in[((int'(c_g4) + i / 4) % 4) * 4 + i % 4] = 1'b1;\n        lane_c[((int'(c_g4) + i / 4) % 4) * 4 + i % 4]",
     "lane_in[i % 4] = 1'b1;\n        lane_c[i % 4]", "row sum uses 4 lanes, not 16 (hold-buffer passes)", ROW_T),
    (CORE, "if (v_con && !v_last && c_sum && step_lanes[l]) v_cool[l] <= 2'd2;",
     "if (v_con && !v_last && c_sum && step_lanes[l]) v_cool[l] <= 2'd1;",
     "a lane takes the next summed step while its adder is busy", ROW_T),
    (CORE, "{2'd0, 2'd1}: v_fwait <= 8'd51;", "{2'd0, 2'd1}: v_fwait <= 8'd47;",
     "softmax takes 1/sum before it is out of the reciprocal", ROW_T),
    (CORE, "assign cur = lane_p[l][2] ? lane_y[l] : lane[l];", "assign cur = lane[l];",
     "lane accumulator without result forwarding", ROW_T),
    (VPU, "fpu_add u (.clk, .a(t8[2 * i]), .b(t8[2 * i + 1]), .y(t4[i]));",
     "fpu_add u (.clk, .a(t8[i]), .b(t8[i + 4]), .y(t4[i]));", "lane fold pairs the wrong partial sums", ROW_T),
    (VPU, "fpu_add u_a (.clk, .a(m), .b(eps), .y(a));", "fpu_add u_a (.clk, .a(m), .b(32'h0), .y(a));",
     "norms ignore eps", ROW_T),
    (VPU, "2'd1:    begin a1 = F; b1 = kout_d3 ? sc_d3 : F; end",
     "2'd1:    begin a1 = F; b1 = kout_d3 ? sc_d3 : x; end", "LayerNorm variance of (x - mean) x", ROW_T),
    (VPU, "fpu_dly #(32, 6) u_g6 (.clk, .d(g), .q(g6));", "fpu_dly #(32, 5) u_g6 (.clk, .d(g), .q(g6));",
     "LayerNorm gamma one cycle out of step", ROW_T),
    # (the softmax r one cycle out of step, u_sc54 53, is equivalent: the issue register holding a
    # step's r changes only at the next step, and every output step of a row carries the same r,
    # while the next row's first output step comes a whole pass and fold later)
    (CORE, "if (v_con && x_fits && v_pass == 0)", "if (1'b0)", "row not copied into the hold buffer", ROW_T),
    (CORE, "S_V_NEXT: if (!wr_busy && oq_cnt == 0)", "S_V_NEXT: if (1'b1)", "row op ends before its last write", ROW_T),
    (CORE, "S_V_NEXT: if (!wr_busy && oq_cnt == 0)", "S_V_NEXT: if (!wr_busy)",
     "row op ends with output steps still in the lanes", ROW_T),
    (CORE, "oq_l[oq_tl] <= v_n << (32'd2 + 32'(x_esl));", "oq_l[oq_tl] <= 32'd4 << 32'(x_esl);",
     "a wide output step writes only its first group", ROW_T),
    (CORE, "if (a_ready && !b_act) begin", "if (a_ready) begin", "a row handed over while the previous one runs", ROW_T),
    (CORE, "a_fwait <= (x_fn == 2'd0) ? 8'd5 : (x_fn == 2'd1) ? 8'd16 : 8'd75;",
     "a_fwait <= (x_fn == 2'd0) ? 8'd5 : (x_fn == 2'd1) ? 8'd16 : 8'd70;",
     "overlapped RMSNorm takes rs before it is out", ROW_T),
    (CORE, "             (f_row && !f_bias && d_boff != 0) ||", "             1'b0 ||",
     "BIAS_OFF without bias accepted on a row op", ERR_T),
    (CORE, "inv_wait <= 8'd42;", "inv_wait <= 8'd30;", "1/N taken before it is out of its pipeline", ROW_T),
    (CORE, "fw <= fw + 64'(x_k);", "fw <= fw;", "loader addresses every W row of a tile at its first"),
    (CORE, "else if (l_nn0 != ln0) begin l_wrow0 <= l_wrow0 + x_ck;", "else if (1'b0) begin l_wrow0 <= l_wrow0 + x_ck;",
     "loader W rows stay on the first column block"),
    (CORE, "nb_p <= f_p + x_slda; end", "nb_p <= f_p; end",
     "conv loader starts the next row block one pixel early", CONV_T),
    (CORE, "iy >= 19'(x_ch) ||", "iy > 19'(x_ch) ||", "the row below the image read instead of padding", CONV_T),
    (CORE, "f_ox <= f_ox + 16'd1; f_p <= f_p + x_slda;", "f_ox <= f_ox + 16'd1; f_p <= f_p + 64'(x_lda);",
     "conv stride ignored along the width", CONV_T),
    (CORE, "aix[fi] <= 18'($signed({2'b0, f_ox}) * $signed({15'b0, x_cs[2:0]})) - 18'(x_cp);",
     "aix[fi] <= 18'($signed({2'b0, f_ox})) - 18'(x_cp);", "conv padding test ignores the stride", CONV_T),
    (CORE, "32'(d_m) != r_pohw)) ||", "1'b0)) ||", "M not checked against OH x OW", CONV_T),
    (CORE, "cw_cc <= cw_cc + 16'd1; cw_off <= cw_off + 64'd16; end", "cw_cc <= cw_cc + 16'd1; end",
     "channel chunk ignored in the image address", CONV_T),
    # P8 timing closure: the stale mutants rewritten for the registered feed window, the loader's
    # candidate stage, the check sequencer, the pipelined PE and i2f
    (CORE, "cd_h1 <= lh_first && (l_isw == x_minner);", "cd_h1 <= lh_first && (l_isw != x_minner);",
     "the other operand captured into the hold buffer", HOLD_T),
    (CORE, "feed_last <= 32'(d_k) + 32'(SK) - 32'd2;", "feed_last <= 32'(d_k) + 32'(SK) - 32'd3;", "last skewed K step not fed"),
    (CORE, "if (rlive[i] && fa_in[i] && (kc & 15) == 15) a_full", "if (rlive[i] && fa_in[i] && (kc & 15) == 14) a_full",
     "A slot freed one byte before its row is done with it"),
    (CORE, "if (!w_hold && clive[j] && fw_in[j] && !w_full[fw_sl[j]][j]) feed_go = 1'b0;", "if (1'b0) feed_go = 1'b0;",
     "feed does not wait for weights"),
    (CORE, "fa_in[i] <= kc + 1 >= 0 && kc + 1 < int'(x_k);", "fa_in[i] <= kc + 1 >= 0 && kc + 1 <= int'(x_k);",
     "a row fed one K step too many"),
    (CORE, "fw_sl[j] <= LB'(f_base + LB'((kc + 1) >> 4));", "fw_sl[j] <= LB'(f_base + LB'(kc >> 4));",
     "W slot advanced one step late"),
    (CORE, " || a_pend[cd_b][(int'(cd_i) < R) ? int'(cd_i) : 0]", "",
     "slot reissued while its previous read is in flight"),
    (CORE, "e_s <= t_s; e_b <= t_b;", "e_s <= t_s;", "epilogue keeps an older tile's bias"),
    (CORE, "hb_rd(x_gb1 + v_gi_con + 32'(k))", "hb_rd(x_gb0 + v_gi_con + 32'(k))", "beta read from gamma's region", ROW_T),
    (CORE, "a_wad <= a_buf ? 32'd0 : x_nbr;", "a_wad <= a_buf ? x_nbr : 32'd0;",
     "the next row overwrites the region still being read", ROW_T),
    (CORE, "&& (32'(d_n) >> 2) * (f_norm ? (f_bias ? 32'd4 : 32'd3) : 32'd2) <= 32'(SK * KR);\n        r_gb0",
     "&& (32'(d_n) >> 2) * (f_norm ? (f_bias ? 32'd3 : 32'd2) : 32'd1) <= 32'(SK * KR);\n        r_gb0",
     "overlap chosen when the second row does not fit", ROW_T),
    # (a bias or scale delay that is off by a few cycles is equivalent: every row of a tile has the
    # same per-column bias and scale, and the next tile's first launch comes after the handoff, a
    # gap the lanes' alignment never reaches; the lanes must still keep their columns apart)
    (CORE, "fpu_dly #(32, 8) u_b4 (.clk, .d(q_b[j]), .q(b4));", "fpu_dly #(32, 8) u_b4 (.clk, .d(q_b[0]), .q(b4));",
     "every epilogue lane adds column 0's bias"),
    (CORE, "if (!wr_busy && !e_busy && oq_cnt == 0) begin qi <= qi + 1; q_daddr <= q_daddr + 64'd64; state <= S_NEXT; end   // last row written first",
     "if (1'b1) begin qi <= qi + 1; q_daddr <= q_daddr + 64'd64; state <= S_NEXT; end",
     "GEMM descriptor ends before its last row is written"),
    (CORE, "e_adv > e_ei + 32'd1 && oq_cnt", "oq_cnt", "epilogue reads a row before the array has finished it"),
    (CORE, "e_adv > e_ei + 32'd1 && oq_cnt", "e_adv > e_ei && oq_cnt",
     "epilogue forgets the PE's product register (reads a row one step early)"),
    (CORE, "e_go     = e_busy && e_ei < e_nrow && e_adv > e_ei + 32'd1 && oq_cnt < (LQO + 1)'(OQ);",
     "e_go     = e_busy && e_ei < e_nrow && e_adv > e_ei + 32'd1;", "epilogue launches without a free output-queue slot",
     ["tests/test_rtl.py", "-k", "epilogue_busy"]),
    (CORE, "e_nrow <= (x_m - m0 < 32'(R)) ? x_m - m0 : 32'(R);", "e_nrow <= 32'(R);",
     "epilogue writes the rows below the tensor"),
    (CORE, "a_inc[int'(x_fn == 2'd2 ? axx_g4 : ax_g4) * 4 + i] = 1'b1;\n        a_lcc[int'(x_fn == 2'd2 ? axx_g4 : ax_g4) * 4 + i]",
     "a_inc[i] = 1'b1;\n        a_lcc[i]", "row sum uses 4 lanes, not 16 (overlapped pass 0)", ROW_T),
    # (a_drain 7 is still exact: the fold's sample point trails its start by the scalar latency
    # minus one, which covers the RMSNorm x*x contribution three cycles later; 2 does not)
    (CORE, "if (a_con) a_drain <= 8'd10;", "if (a_con) a_drain <= 8'd2;",
     "overlapped pass 0 folds before its last contributions are summed", ROW_T),
    # (+3 is still exact, for the same reason as a_drain below: the scalars are sampled a fold
    # latency after the drain ends, from lanes as they stood ~2 cycles after it)
    (CORE, "v_drain <= 8'(v_clat + 32'd6);", "v_drain <= 8'(v_clat);",
     "pass folds before its last contributions are summed", ROW_T),
    (CORE, "if (lh_first && !x_minner) hbuf[cd_i][cd_c] <= 128'b0;", "",
     "padding zeros not copied into the hold buffer", CONV_T),
    (CORE, "cd_zero <= !l_isv && l_zero;", "cd_zero <= 1'b0;", "padding read from memory instead of zero", CONV_T),
    # (dropping !cd_v from l_tdone is equivalent: a waiting candidate carries its address, slot and
    # destination, and the last read of a tile is never a vector beat, the one kind keyed on l_tag)
    (CORE, "cd_i <= l_isv ? LP'((vg < nvs) ? vg : (vg < nvsb) ? vg - nvs : vg - nvsb) : LP'(lp >> 1);",
     "cd_i <= l_isv ? LP'((vg < nvs) ? vg : (vg < nvsb) ? vg : vg - nvsb) : LP'(lp >> 1);",
     "bias beats stored after the scale beats' slots"),
    (CORE, "cd_i <= l_isv ? LP'((vg < nvs) ? vg : (vg < nvsb) ? vg - nvs : vg - nvsb) : LP'(lp >> 1);",
     "cd_i <= l_isv ? LP'((vg < nvs) ? vg : vg - nvs) : LP'(lp >> 1);",
     "ROWCOL row-scale beats stored in the bias slots", V04_T),
    (CORE, "r_d1  <= 64'(r_lda) -", "r_d1  <= 64'(r_cin) -", "pixel stride (LDA) ignored along the kernel", CONV_T),
    (CORE, "if (ck_i == 5'd1) ck_st <= 3'd4;", "if (ck_i == 5'd2) ck_st <= 3'd4;",
     "the check's dividers stop a bit early", CONV_T),
    # (S_CHKW without its wait survives: the sequencer (dividers 17 cycles, 13 products at 4 bits a
    # cycle of the smaller operand) ends within the 64-cycle CRC on every descriptor the tests
    # build; only a conv with H x W near 2^28 and wide W / OW could need longer. The wait stays.)
    (CORE, "mu_p <= nx; mu_a <= mu_a << 4; mu_b <= mu_b >> 4;", "mu_p <= nx; mu_a <= mu_a << 3; mu_b <= mu_b >> 4;",
     "the check multiplier misaligns its partial products"),
    (CORE, "S_V_NEXT: if (!wr_busy && oq_cnt == 0) begin qi <= qi + 1; q_daddr <= q_daddr + 64'd64;",
     "S_V_NEXT: if (!wr_busy && oq_cnt == 0) begin qi <= qi + 1;",
     "after a row op the next descriptor is fetched from the same address", ROW_T),
    (PE, "if (p_v) begin", "if (a_v_in && w_v_in) begin", "PE accumulates by the operands' flags, not the product's"),
    (CORE, "iu <= q_acc[j][31] ? 32'(-q_acc[j]) : 32'(q_acc[j]);", "iu <= 32'(q_acc[j]);",
     "epilogue i2f loses the magnitude of negative sums"),
    (CORE, "      s0 <= q_s[j]; r0 <= q_r;", "      s0 <= q_s[0]; r0 <= q_r;", "every epilogue lane scales by column 0's scale"),
    # ---- v0.4: the extension unit (ELT, ROPE, FDOT, QUANT) and the new rules
    (XPU, "eop == 2'd3 ? fneg(e_bd[i]) : e_bd[i]", "eop == 2'd1 ? fneg(e_bd[i]) : e_bd[i]", "ELT negates B on ADD, not SUB", V04_T),
    # (C one or two cycles early is equivalent: the issue registers hold a step's C until the next
    # step fires, at least 3 cycles later since a MULADD step has 3 beats; three early is not)
    (XPU, "tm_vdly u_cd (.clk, .rst, .d(ic[i]), .n(act_l + 32'd3)", "tm_vdly u_cd (.clk, .rst, .d(ic[i]), .n(act_l)",
     "MULADD's C not delayed past the multiply", V04_T),
    (XPU, "tm_vdly u_bd (.clk, .rst, .d(ib[i]), .n(act_l)", "tm_vdly u_bd (.clk, .rst, .d(ib[i]), .n(32'd1)",
     "ELT B not delayed by the activation's latency", V04_T),
    # (the scalar word from the per-row pointer is equivalent: for a scalar B that pointer only
    # moves in 16-byte steps, so its word bits stay those of the base)
    (XPU, "b_ws = (bmode == 2'd2) ? rb_row[3:2] : b_base[3:2];", "b_ws = b_base[3:2];",
     "per-row B word taken from the base address", V04_T),
    (XPU, "rb_row <= rb_row + ((bmode == 2'd2) ? 64'd4 : 64'(ldb));", "rb_row <= rb_row + 64'(ldb);",
     "per-row B steps by the B stride", V04_T),
    (XPU, "b_kind = K_A; b_addr = (arow ? a_base : ra_row)", "b_kind = K_A; b_addr = (ra_row)",
     "row-broadcast A read as a matrix", V04_T),
    (XPU, "u_rs (.clk, .a(r_p1[i]), .b(r_neg3 ? fneg(r_p2[i]) : r_p2[i])", "u_rs (.clk, .a(r_p1[i]), .b(r_neg3 ? r_p2[i] : fneg(r_p2[i]))",
     "ROPE rotates the wrong way", V04_T),
    (XPU, "if (rgc <= 1) begin rgc <= g_n;", "if (rgc <= 2) begin rgc <= g_n;", "ROPE table row advanced every G - 1 rows", V04_T),
    (XPU, "rtab2 <= b_base + {31'b0, k_n, 1'b0};", "rtab2 <= b_base + {30'b0, k_n, 2'b0};",
     "ROPE sin half read past the table row", V04_T),
    (XPU, "i_rot <= qro[q_h]; i_neg <= qng[q_h]; i_lg <= qlg[q_h];", "i_rot <= qro[q_h]; i_neg <= qng[q_h]; i_lg <= 2'd0;",
     "FDOT products all into lanes 0-3 (wrong summation order)", V04_T),
    (XPU, "ia[i] <= (qlg[q_h] == 2'd0) ? ca[qws[q_h]] : nn_a;", "ia[i] <= (qlg[q_h] == 2'd0) ? ca[0] : nn_a;",
     "FDOT NN takes word 0 of the A beat for every k", V04_T),
    (XPU, "3'd0: if (w_cnt == 8'd9) begin           // the last product is in its lane",
     "3'd0: if (w_cnt == 8'd2) begin           // the last product is in its lane",
     "FDOT NT folds before the last product is in", V04_T),
    (XPU, "qmax[i] <= i_qfi ? fabs(ia[i]) :", "qmax[i] <= 1'b0 ? fabs(ia[i]) :",
     "QUANT's max carries over from the previous row", V04_T),
    (XPU, "ro_go <= 1'b1; ro_raw <= 1'b1; ro_v[0] <= q_s;", "ro_go <= 1'b1; ro_raw <= 1'b0; ro_v[0] <= q_s;",
     "QUANT's row scale converted to INT8", V04_T),
    (XPU, "3'd1: if (w_cnt == 8'd42) w_ph <= 3'd2;   // recip (38) + fmul (3)", "3'd1: if (w_cnt == 8'd30) w_ph <= 3'd2;   // recip (38) + fmul (3)",
     "QUANT uses 127 / amax before it is out", V04_T),
    (XPU, "assign busy    = r_act || cv || q_rcv != 0 || cnt != 0 || c_wait || i_launch || ro_go;",
     "assign busy    = r_act || cv || q_rcv != 0 || cnt != 0 || c_wait;", "the unit idles before its last launch", V04_T),
    (CORE, ".oq_free(32'(oq_cnt) + 32'(xp_lq_go) < 32'(OQ))", ".oq_free(32'(oq_cnt) < 32'(OQ))",
     "extension unit's queue credit misses the launch in progress", ["tests/test_rtl_v04.py", "-k", "small_output_queue"]),
    (CORE, "(f_rope && (d_k == 0 || d_k[4:0] != 0 || d_k > d_n))", "(f_rope && (d_k == 0 || d_k[4:0] != 0))",
     "ROPE with R > N accepted", V04_T),
    (CORE, "(!f_row && (f_elt && d_hrsv[1] ? d_woff[1:0] != 0 : d_woff[3:0] != 0))", "(!f_row && d_woff[3:0] != 0)",
     "per-row / scalar ELT B must be 16-byte aligned", V04_T),
    (CORE, "r_bx  <= f_rope ? 66'(r_ptab) :", "r_bx  <= f_rope ? 66'(d_k) << 2 :", "ROPE table extent: one row only", V04_T),
    (CORE, "f_ldb  = (d_group != 0) ? 32'(d_group) << 4 : f_brow;", "f_ldb  = f_brow;", "B stride (GROUP_SIZE) ignored", V04_T),
    (CORE, "(f_quant && r_hiq > 66'(MEM_BYTES))", "(1'b0)", "QUANT's row scales not range-checked", V04_T),
    (CORE, "(f_gemm && f_qm == 3'd3 && r_hir > 66'(MEM_BYTES))", "(1'b0)", "ROWCOL row scales not range-checked", V04_T),
    (CORE, "(d_hrsv != 0 && !f_hdr_ok)", "(d_hrsv != 0 && !f_hdr_ok && !f_elt)", "ELT mode bits beyond [2:0] accepted", V04_T),
    (CORE, "(f_fdot && (d_ctrl[15:1] != 0 ||", "(f_fdot && (d_ctrl[15:2] != 0 ||", "FDOT CTRL[1] accepted", V04_T),
    # GEMV (M = 1 on the extension unit)
    (XPU, "gpr[i] <= $signed(gv_w[8 * i +: 8]) * $signed(gv_x[8 * i +: 8]);",
     "gpr[i] <= $signed(gv_w[8 * i +: 8]) * $signed({1'b0, gv_x[8 * i +: 8]});", "GEMV x taken as unsigned", GV_T),
    (XPU, "if (g3v) gacc <= g3f ? 32'(g3) : gacc + 32'(g3);", "if (g3v) gacc <= gacc + 32'(g3);",
     "GEMV accumulator not restarted per channel", GV_T),
    (XPU, ".ra((fire && op == O_GV) ? (qfi[q_h] ? 8'd0 : gidx) : 8'd0)", ".ra((fire && op == O_GV) ? gidx : 8'd0)",
     "GEMV x index not reset at a channel's first beat", GV_T),
    (XPU, "gv_sc <= csb[32 * ((qm == 2'd1) ? 0 : int'(qws[q_h])) +: 32];", "gv_sc <= csb[32 * int'(qws[q_h]) +: 32];",
     "GEMV per-tensor scale read per channel", GV_T),
    (XPU, "        b_ws = rk[1:0];                   // the channel's word", "        b_ws = (qm == 2'd1) ? 2'd0 : rk[1:0];   // the channel's word",
     "GEMV bias word follows the scale's (per-tensor: word 0)", GV_T),
    (XPU, "          if (hk == K_XP) gv_rs <= hd[31:0];", "          if (1'b0) gv_rs <= hd[31:0];", "GEMV row scale ignored", GV_T),
    (XPU, "    gv_lb  = bias && rk[1:0] == 2'd0;", "    gv_lb  = bias && rk == 0;", "GEMV bias beat read once, not every 4 channels", GV_T),
    (XPU, "fpu_dly #(32, 9) u_gs9 (.clk, .d(gsc0), .q(gsc9));", "fpu_dly #(32, 10) u_gs9 (.clk, .d(gsc0), .q(gsc9));",
     "GEMV column scale a cycle late", GV_T),
    (CORE, "f_gv   = f_gemm && d_m == 16'd1 && d_k <= 16'd4096;", "f_gv   = f_gemm && d_m == 16'd1 && d_k <= 16'd4112;",
     "GEMV taken past the x buffer", GV_T),
    # v0.4 in the row engine and the epilogue
    # (keeping one more exp is equivalent: that element enters as -inf, so its exp is +0 already,
    # or NaN only when the row's sum is NaN as well; the kept flag stays as the stated rule)
    (CORE, "sx[i] <= (v_gi_con * 32'd4 + 32'(i) >= v_vlen) ? 32'hFF800000", "sx[i] <= (v_gi_con * 32'd4 + 32'(i) > v_vlen) ? 32'hFF800000",
     "causal softmax keeps one element too many", V04_T),
    (CORE, "x_invn <= x_l2 ? 32'h3F800000 : inv_out;", "x_invn <= inv_out;", "L2NORM divides by N", V04_T),
    (CORE, "kr = 32'(2'(x_rs[3:2] + m0[1:0])) + 32'(i);", "kr = 32'(2'(x_rs[3:2])) + 32'(i);",
     "ROWCOL row scales not offset by the tile's first row", ["tests/test_rtl_v04.py", "-k", "fuzz_v04 and 3x5"]),
    # (the threshold at 20 inclusive is equivalent: ln(1 + e^-20) is far below half an ulp of 20,
    # and so is every x above ~17; a threshold at 10 is not)
    (FP_PKG, "if (!x[31] && x[30:0] > SP_HI[30:0]) return x;", "if (!x[31] && x[30:0] > 31'h41200000) return x;",
     "softplus returns x from 10 on", FP_T),
    (FP_PKG, "return recip_f(fadd(F_ONE, exp_f(fneg(ftz(x_in)))));", "return recip_f(fadd(F_ONE, exp_f(ftz(x_in))));",
     "sigmoid of -x", FP_T),
]

ROW_FAST = ["tests/test_rtl.py", "-k", "compiled_rowops and 8-8-1024-4 or queue_end_waits"]
CONV_FAST = ["tests/test_rtl.py", "-k", "compiled_conv and 8-8 or errors_on_rtl"]
TIMEOUT = 1800       # s: tier 1 normally takes ~6 min on 2 cores (one build + 12 queues)
TIER1 = ["tests/test_rtl.py", "-k", "fuzz_random_queues and 8-8 and not 8-8-4"]

DEFAULT_TESTS = ["tests/test_rtl.py", "tests/test_rtl_fp.py", "-k",
                 "fuzz_random_queues and (8-8 or 3-5) or errors_on_rtl or compiled and 5-3 "
                 "or coherence or boundary or loader or hold or test_exp or test_recip or test_rsqrt "
                 "or test_act"]


def run_tests(tree: Path, tests: list[str], timeout: float | None = None) -> bool:
    """True if the tests pass. A run past `timeout` counts as a failure: a mutant that hangs the
    core (a simulation that never reaches its end) is caught, and is not worth waiting out."""
    env = dict(os.environ, TM_MAXCYC="5000000")     # longest legitimate test run: < 1 M cycles
    p = subprocess.Popen([sys.executable, "-m", "pytest", "-q", "-x", *tests], cwd=tree, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        return p.wait(timeout=timeout) == 0
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", help="run only the mutants whose description contains one of these "
                                   "'|'-separated texts")
    ap.add_argument("--no-baseline", action="store_true", help="skip the unmutated run (already known to pass)")
    args = ap.parse_args()
    if args.list:
        for f, _, _, what, *_ in MUTANTS:
            print(f"{f}: {what}")
        return 0
    survivors = []
    with tempfile.TemporaryDirectory() as td:
        tree = Path(td) / "tree"
        shutil.copytree(ROOT, tree, ignore=shutil.ignore_patterns("build", ".git", "__pycache__",
                                                                  ".pytest_cache"))
        if not args.no_baseline and not run_tests(tree, DEFAULT_TESTS):
            print("the unmutated tree fails the tests; fix that first")
            return 2
        for f, old, new, what, *tests in MUTANTS:
            if args.only and not any(o in what for o in args.only.split("|")):
                continue
            path = tree / f
            src = path.read_text()
            if src.count(old) != 1:
                print(f"STALE   {what}: original text found {src.count(old)} times")
                survivors.append(what)
                continue
            path.write_text(src.replace(old, new))
            # tier 1: the 8x8 fuzz (one build) catches most core mutants; only the rest pay for
            # their full subset (each parameter set is a Verilator build)
            # (v0.4 mutants go straight to their own subset: the v0.3 fuzz never reaches them)
            caught = f != FP_PKG and not (tests and tests[0][0] == "tests/test_rtl_v04.py") \
                and not run_tests(tree, TIER1, timeout=TIMEOUT)
            # a second fast tier for row-op and conv mutants: one array size of the directed tests
            fast = {id(ROW_T): ROW_FAST, id(CONV_T): CONV_FAST}.get(id(tests[0]) if tests else 0)
            caught = caught or (fast is not None and not run_tests(tree, fast, timeout=TIMEOUT))
            caught = caught or not run_tests(tree, tests[0] if tests else DEFAULT_TESTS, timeout=4 * TIMEOUT)
            path.write_text(src)
            shutil.rmtree(tree / "build", ignore_errors=True)     # each mutant's builds are dead weight
            print(f"{'caught ' if caught else 'SURVIVED'} {what}")
            if not caught:
                survivors.append(what)
    ran = sum(1 for _, _, _, what, *_ in MUTANTS
              if not args.only or any(o in what for o in args.only.split("|")))
    print(f"{ran - len(survivors)}/{ran} mutants caught")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
