// Streams a vector file through the pipelined units (rtl/tm_fpu.sv), one operation per cycle,
// and writes each result when it leaves the pipeline. +op: 0 fmul, 1 fadd, 2 exp, 3 recip,
// 4 rsqrt, 5 act (act = b[2:0], per line: each result is read at its own act's latency; the
// ASIC fpu_act needs one act for the whole file, the FPGA one takes any mix); lines "a b" in hex.
module tb_fpu;
  logic clk = 1'b0;
  always #5 clk = ~clk;
  logic [31:0] a, b, ym, ya, ye, yr, ys, y1, ylong, ysp;
  logic [2:0]  act;
  fpu_mul   u_m (.clk, .a, .b, .y(ym));
  fpu_add   u_a (.clk, .a, .b, .y(ya));
  fpu_exp   u_e (.clk, .x(a), .y(ye));
  fpu_recip u_r (.clk, .x(a), .y(yr));
  fpu_rsqrt u_s (.clk, .x(a), .y(ys));
  fpu_act   u_t (.clk, .x(a), .act, .y1, .ylong, .ysp
`ifdef TM_FPGA
                 , .esel(1'b0), .xe(32'h0), .ye()     // softmax's direct exp: the core's structural test
`endif
                 );
  assign act = b[2:0];
  int op, lat, n;
  logic [31:0] va [$], vb [$];
  logic [31:0] hm [$], ha [$], he [$], hr [$], hs [$], h1 [$], hl [$], hp [$];
  initial begin
    int fi, fo, k;
    string inp, outp;
    logic [31:0] x, y;
    if (!$value$plusargs("in=%s", inp)) $fatal(1, "+in= missing");
    if (!$value$plusargs("out=%s", outp)) $fatal(1, "+out= missing");
    if (!$value$plusargs("op=%d", op)) $fatal(1, "+op= missing");
    fi = $fopen(inp, "r");
    while (!$feof(fi)) begin
      k = $fscanf(fi, "%h %h\n", x, y);
      if (k != 2) break;
      va.push_back(x); vb.push_back(y);
    end
    $fclose(fi);
    n = va.size();
    for (int c = 0; c < n + tm_fp_pkg::SP_LAT + 8; c++) begin
      a = (c < n) ? va[c] : 32'h0;
      b = (c < n) ? vb[c] : vb[n-1];   // act held after the last vector (the ASIC fpu_act selects at its output)
      @(negedge clk);       // inputs settle between edges
      @(posedge clk);
      #1;
      // the outputs now hold the results of the vectors fed at cycle c - latency + 1
      hm.push_back(ym); ha.push_back(ya); he.push_back(ye); hr.push_back(yr); hs.push_back(ys);
      h1.push_back(y1); hl.push_back(ylong); hp.push_back(ysp);
    end
    fo = $fopen(outp, "w");
    for (int i = 0; i < n; i++) begin
      case (op)
        0: lat = 3; 1: lat = 3; 2: lat = 51; 3: lat = 38; 4: lat = 56;
        default: lat = (vb[i][2:0] == 3'd6) ? tm_fp_pkg::SP_LAT : (vb[i][2:0] >= 3'd3) ? 122 : 1;
      endcase
      k = i + lat - 1;
      $fwrite(fo, "%08h\n", op == 0 ? hm[k] : op == 1 ? ha[k] : op == 2 ? he[k] : op == 3 ? hr[k]
                            : op == 4 ? hs[k] : lat == 1 ? h1[k] : lat == 122 ? hl[k] : hp[k]);
    end
    $fclose(fo);
    $finish;
  end
endmodule
