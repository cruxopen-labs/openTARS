// One AXI4 traffic generator for the DDR bandwidth test: bursts of LEN+1 beats (16 B each) to
// BASE + (offset & MASK), the offset stepping one burst at a time, at most OT bursts in flight,
// NBURST bursts in all; MODE 0 reads (R beats counted, data dropped), 1 writes (a fixed pattern).
// Measures the cycles from start to the last completion, the beats, and per-burst latency (AR to
// the first R beat; AW to B), sum / min / max. One ID, so completions return in issue order.
module tg_master #(parameter AW = 40, parameter IDW = 6) (
  input  wire          clk,
  input  wire          rstn,
  input  wire          start,
  input  wire [AW-1:0] base,
  input  wire [31:0]   mask,
  input  wire [7:0]    len,
  input  wire [6:0]    ot,          // 1..64
  input  wire [31:0]   nburst,
  input  wire          mode,
  output reg           busy,
  output reg  [31:0]   cyc,
  output reg  [31:0]   beats,
  output reg  [47:0]   lat_sum,
  output reg  [31:0]   lat_min,
  output reg  [31:0]   lat_max,
  output reg  [31:0]   errs,
  output wire [AW-1:0] m_axi_araddr,
  output wire [7:0]    m_axi_arlen,
  output wire [2:0]    m_axi_arsize,
  output wire [1:0]    m_axi_arburst,
  output wire [3:0]    m_axi_arcache,
  output wire [2:0]    m_axi_arprot,
  output wire [IDW-1:0] m_axi_arid,
  output wire          m_axi_arlock,
  output wire [3:0]    m_axi_arqos,
  output reg           m_axi_arvalid,
  input  wire          m_axi_arready,
  input  wire [127:0]  m_axi_rdata,
  input  wire [1:0]    m_axi_rresp,
  input  wire          m_axi_rlast,
  input  wire [IDW-1:0] m_axi_rid,
  input  wire          m_axi_rvalid,
  output wire          m_axi_rready,
  output wire [AW-1:0] m_axi_awaddr,
  output wire [7:0]    m_axi_awlen,
  output wire [2:0]    m_axi_awsize,
  output wire [1:0]    m_axi_awburst,
  output wire [3:0]    m_axi_awcache,
  output wire [2:0]    m_axi_awprot,
  output wire [IDW-1:0] m_axi_awid,
  output wire          m_axi_awlock,
  output wire [3:0]    m_axi_awqos,
  output reg           m_axi_awvalid,
  input  wire          m_axi_awready,
  output wire [127:0]  m_axi_wdata,
  output wire [15:0]   m_axi_wstrb,
  output wire          m_axi_wlast,
  output wire          m_axi_wvalid,
  input  wire          m_axi_wready,
  input  wire [1:0]    m_axi_bresp,
  input  wire [IDW-1:0] m_axi_bid,
  input  wire          m_axi_bvalid,
  output wire          m_axi_bready
);
  reg          md;                        // mode latched at start
  reg [31:0]   off, issued;
  reg [6:0]    inflt;                     // bursts issued and not complete
  reg [AW-1:0] addr;
  reg [31:0]   now;
  reg [31:0]   ts [0:63];                 // issue times, in order
  reg [5:0]    ts_w, ts_r;
  reg          r_first;
  reg [31:0]   wpend;                     // write bursts whose data is not all sent
  reg [7:0]    wcnt;

  wire a_fire  = md ? (m_axi_awvalid & m_axi_awready) : (m_axi_arvalid & m_axi_arready);
  wire c_fire  = md ? m_axi_bvalid : (m_axi_rvalid & m_axi_rlast);     // a burst completes
  wire l_fire  = md ? m_axi_bvalid : (m_axi_rvalid & r_first);         // its latency is known
  wire w_fire  = m_axi_wvalid & m_axi_wready;
  wire a_v     = md ? m_axi_awvalid : m_axi_arvalid;

  wire [31:0] issued_n = issued + (a_fire ? 32'd1 : 32'd0);
  wire [6:0]  inflt_n  = inflt + (a_fire ? 7'd1 : 7'd0) - (c_fire ? 7'd1 : 7'd0);
  wire        want     = busy && issued_n < nburst && inflt_n < ot;
  wire [31:0] lat      = now - ts[ts_r];

  assign m_axi_araddr  = addr;   assign m_axi_awaddr  = addr;
  assign m_axi_arlen   = len;    assign m_axi_awlen   = len;
  assign m_axi_arsize  = 3'd4;   assign m_axi_awsize  = 3'd4;
  assign m_axi_arburst = 2'b01;  assign m_axi_awburst = 2'b01;
  assign m_axi_arcache = 4'b0011; assign m_axi_awcache = 4'b0011;
  assign m_axi_arprot  = 3'b010; assign m_axi_awprot  = 3'b010;
  assign m_axi_arid    = {IDW{1'b0}}; assign m_axi_awid = {IDW{1'b0}};
  assign m_axi_arlock  = 1'b0;   assign m_axi_awlock  = 1'b0;
  assign m_axi_arqos   = 4'd0;   assign m_axi_awqos   = 4'd0;
  assign m_axi_rready  = 1'b1;
  assign m_axi_bready  = 1'b1;
  assign m_axi_wdata   = {4{now}};
  assign m_axi_wstrb   = 16'hFFFF;
  assign m_axi_wvalid  = wpend != 0;
  assign m_axi_wlast   = wcnt == len;

  always @(posedge clk) begin
    if (!rstn) begin
      busy <= 1'b0; md <= 1'b0; m_axi_arvalid <= 1'b0; m_axi_awvalid <= 1'b0;
      now <= 32'd0; wpend <= 32'd0; wcnt <= 8'd0; inflt <= 7'd0;
    end else begin
      now <= now + 32'd1;
      if (start && !busy) begin
        busy <= 1'b1; md <= mode; off <= 32'd0; issued <= 32'd0; inflt <= 7'd0;
        addr <= base; ts_w <= 6'd0; ts_r <= 6'd0; r_first <= 1'b1; wpend <= 32'd0; wcnt <= 8'd0;
        cyc <= 32'd0; beats <= 32'd0; lat_sum <= 48'd0; lat_min <= 32'hFFFF_FFFF; lat_max <= 32'd0;
        errs <= 32'd0;
      end else if (busy) begin
        cyc <= cyc + 32'd1;
        issued <= issued_n;
        inflt <= inflt_n;
        if (a_fire) begin
          ts[ts_w] <= now;
          ts_w <= ts_w + 6'd1;
        end
        // a valid stays up until it is taken; the next one goes out when allowed
        if (!a_v || a_fire) begin
          if (md) m_axi_awvalid <= want; else m_axi_arvalid <= want;
          if (a_fire) begin
            off  <= off + {16'd0, len, 4'd0} + 32'd16;
            addr <= base + ((off + {16'd0, len, 4'd0} + 32'd16) & mask);
          end
        end
        if (l_fire) begin
          ts_r <= ts_r + 6'd1;
          lat_sum <= lat_sum + {16'd0, lat};
          if (lat < lat_min) lat_min <= lat;
          if (lat > lat_max) lat_max <= lat;
        end
        if (!md && m_axi_rvalid) begin
          beats <= beats + 32'd1;
          r_first <= m_axi_rlast;
          if (m_axi_rresp != 2'b00) errs <= errs + 32'd1;
        end
        if (md && m_axi_bvalid && m_axi_bresp != 2'b00) errs <= errs + 32'd1;
        if (md) begin
          if (w_fire) begin
            beats <= beats + 32'd1;
            wcnt <= m_axi_wlast ? 8'd0 : wcnt + 8'd1;
          end
          wpend <= wpend + (a_fire ? 32'd1 : 32'd0) - ((w_fire && m_axi_wlast) ? 32'd1 : 32'd0);
        end
        if (issued_n == nburst && inflt_n == 0 && !a_v) busy <= 1'b0;
      end
    end
  end
endmodule
