// R x C output-stationary systolic array. Row i's activation enters at the left edge and reaches
// column j j cycles later; column j's weight enters at the top and reaches row i i cycles later.
// The feeder skews the edges (row i by i cycles, column j by j cycles) so that PE (i, j) sees
// a[i][k] and w[j][k] in the same cycle. Accumulation is integer, so results do not depend on
// R, C or the order the chunks arrive in. `en` advances the whole array by one step; holding it
// low freezes every PE, so a feed stall delays all edges alike and the skew alignment survives.
module tm_array #(
  parameter int R = 8,
  parameter int C = 8
) (
  input  logic                    clk,
  input  logic                    en,
  input  logic signed [7:0]       a_edge [R],
  input  logic                    a_v_edge [R],
  input  logic                    a_t_edge [R],   // the tile's accumulator bank
  input  logic                    a_f_edge [R],   // K step 0
  input  logic signed [7:0]       w_edge [C],
  input  logic                    w_v_edge [C],
  output logic signed [31:0]      acc [2][R][C]   // both banks
);
  logic signed [7:0] a_h [R][C+1];
  logic              av_h [R][C+1];
  logic              at_h [R][C+1];
  logic              af_h [R][C+1];
  logic signed [7:0] w_v [R+1][C];
  logic              wv_v [R+1][C];

  for (genvar i = 0; i < R; i++) begin : g_row_edge
    assign a_h[i][0]  = a_edge[i];
    assign av_h[i][0] = a_v_edge[i];
    assign at_h[i][0] = a_t_edge[i];
    assign af_h[i][0] = a_f_edge[i];
  end
  for (genvar j = 0; j < C; j++) begin : g_col_edge
    assign w_v[0][j]  = w_edge[j];
    assign wv_v[0][j] = w_v_edge[j];
  end

  for (genvar i = 0; i < R; i++) begin : g_r
    for (genvar j = 0; j < C; j++) begin : g_c
      tm_pe u_pe (
        .clk, .en,
        .a_in(a_h[i][j]), .a_v_in(av_h[i][j]), .a_t_in(at_h[i][j]), .a_f_in(af_h[i][j]),
        .w_in(w_v[i][j]), .w_v_in(wv_v[i][j]),
        .a_out(a_h[i][j+1]), .a_v_out(av_h[i][j+1]), .a_t_out(at_h[i][j+1]), .a_f_out(af_h[i][j+1]),
        .w_out(w_v[i+1][j]), .w_v_out(wv_v[i+1][j]),
        .acc0(acc[0][i][j]), .acc1(acc[1][i][j]));
    end
  end
endmodule
