// Runs the fp package over a vector file: "op a b" per line (hex), writes "result" per line.
// op: 0 fmul, 1 fadd, 2 i2f, 3 act(act=b[2:0]), 4 bf16, 5 q8(a, inv=b), 6 exp, 7 recip, 8 rsqrt,
//     9 fadd and 10 fmul through their three pipeline-stage functions (tm_fpp_pkg)
module tb_fp;
  import tm_fp_pkg::*;
  import tm_fpp_pkg::*;
  initial begin
    int fi, fo, n, op;
    logic [31:0] a, b, r;
    string inp, outp;
    if (!$value$plusargs("in=%s", inp)) $fatal(1, "+in= missing");
    if (!$value$plusargs("out=%s", outp)) $fatal(1, "+out= missing");
    fi = $fopen(inp, "r");
    fo = $fopen(outp, "w");
    while (!$feof(fi)) begin
      n = $fscanf(fi, "%d %h %h\n", op, a, b);
      if (n != 3) break;
      case (op)
        0: r = fmul(a, b);
        1: r = fadd(a, b);
        2: r = i2f(a);
        3: r = act_fn(a, b[2:0]);
        4: r = {16'b0, to_bf16(a)};
        5: r = {24'b0, q8(a, b)};
        6: r = exp_f(a);
        7: r = recip_f(a);
        8: r = rsqrt_f(a);
        9: r = fadd_st3(fadd_st2(fadd_st1(a, b)));
        10: r = fmul_st3(fmul_st2(fmul_st1(a, b)));
        default: r = 32'hDEADBEEF;
      endcase
      $fwrite(fo, "%08h\n", r);
    end
    $fclose(fi);
    $fclose(fo);
    $finish;
  end
endmodule
