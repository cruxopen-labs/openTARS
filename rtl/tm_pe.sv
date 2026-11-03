// One processing element of the output-stationary systolic array: the activation moves right,
// the weight moves down, and the PE keeps the exact INT32 sum of a*w while both are valid.
// Two accumulator banks: the activation carries its tile's bank (a_t) and a flag on K step 0
// (a_f). The first product of a tile replaces the bank's old value, so the next tile can start
// accumulating in one bank while the epilogue still reads the other; no clear is needed.
// Pipelined: the product is registered with its flags and added one array step later, so a sum
// is final one step after its last operands passed the PE.
// TM_FPGA: each bank is one DSP48E2 (multiplier, M register, P accumulator; the clock enable is
// the bank's, opmode Z = 0 on K step 0), the operands and flags move on in fabric registers.
module tm_pe (
  input  logic              clk,
  input  logic              en,           // array-wide advance; 0 freezes the PE (feed stall)
  input  logic signed [7:0] a_in,
  input  logic              a_v_in,
  input  logic              a_t_in,       // bank of the tile this activation belongs to
  input  logic              a_f_in,       // K step 0: start a new sum
  input  logic signed [7:0] w_in,
  input  logic              w_v_in,
  output logic signed [7:0] a_out,
  output logic              a_v_out,
  output logic              a_t_out,
  output logic              a_f_out,
  output logic signed [7:0] w_out,
  output logic              w_v_out,
  output logic signed [31:0] acc0,
  output logic signed [31:0] acc1
);
  logic p_v, p_t, p_f;
  always_ff @(posedge clk) begin
    if (en) begin
      a_out   <= a_in;
      a_v_out <= a_v_in;
      a_t_out <= a_t_in;
      a_f_out <= a_f_in;
      w_out   <= w_in;
      w_v_out <= w_v_in;
      p_v     <= a_v_in && w_v_in;
      p_t     <= a_t_in;
      p_f     <= a_f_in;
    end
  end
`ifdef TM_FPGA
  for (genvar b = 0; b < 2; b++) begin : g_bank
    (* use_dsp = "yes" *) logic signed [15:0] m_r;
    (* use_dsp = "yes" *) logic signed [31:0] p_r;
    always_ff @(posedge clk) begin
      if (en) m_r <= a_in * w_in;          // exact int8 x int8 (M register)
      if (en && p_v && p_t == 1'(b))       // exact: |acc| < 2^31 for K <= 65535 (contract)
        p_r <= (p_f ? 32'sd0 : p_r) + 32'(m_r);
    end
    if (b == 0) assign acc0 = p_r; else assign acc1 = p_r;
  end
`else
  logic signed [15:0] prod, prod_r;
  logic signed [31:0] p32;
  assign prod = a_in * w_in;                // 16-bit signed context: exact int8 x int8
  assign p32  = {{16{prod_r[15]}}, prod_r};
  always_ff @(posedge clk) begin
    if (en) prod_r <= prod;
    if (en && p_v) begin
      if (p_t) acc1 <= (p_f ? 32'sd0 : acc1) + p32;
      else     acc0 <= (p_f ? 32'sd0 : acc0) + p32;
    end
  end
`endif
endmodule
