"""Writes tg_top.v: AXI-Lite registers + NM traffic generators, one AXI4 master port each (m<i>_axi).
Registers (byte offsets): 0x000 ID "TGEN"; 0x004 CTRL write: bit0 start, bits[7:4] master mask;
0x008 STATUS: busy bits; 0x00C/0x010 free-running cycle counter lo/hi; 0x014 NM.
Master i at 0x100*(i+1): +00 BASE_LO +04 BASE_HI +08 MASK +0C LEN +10 OT +14 NBURST +18 MODE
(read/write); +20 CYC +24 BEATS +28 LAT_SUM_LO +2C LAT_SUM_HI +30 LAT_MIN +34 LAT_MAX +38 ERRS (read only)."""
NM = 4
P = []
P.append("""  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
  (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axil:%s, ASSOCIATED_RESET rstn" *)
  input  wire clk,
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rstn RST" *)
  (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
  input  wire rstn,
  input  wire [13:0] s_axil_awaddr, input wire [2:0] s_axil_awprot, input wire s_axil_awvalid, output wire s_axil_awready,
  input  wire [31:0] s_axil_wdata, input wire [3:0] s_axil_wstrb, input wire s_axil_wvalid, output wire s_axil_wready,
  output wire [1:0] s_axil_bresp, output reg s_axil_bvalid, input wire s_axil_bready,
  input  wire [13:0] s_axil_araddr, input wire [2:0] s_axil_arprot, input wire s_axil_arvalid, output wire s_axil_arready,
  output reg [31:0] s_axil_rdata, output wire [1:0] s_axil_rresp, output reg s_axil_rvalid, input wire s_axil_rready""" % ":".join(f"m{i}_axi" for i in range(NM)))
sig = [("araddr","o",40),("arlen","o",8),("arsize","o",3),("arburst","o",2),("arcache","o",4),("arprot","o",3),
       ("arid","o",6),("arlock","o",1),("arqos","o",4),("arvalid","o",1),("arready","i",1),
       ("rdata","i",128),("rresp","i",2),("rlast","i",1),("rid","i",6),("rvalid","i",1),("rready","o",1),
       ("awaddr","o",40),("awlen","o",8),("awsize","o",3),("awburst","o",2),("awcache","o",4),("awprot","o",3),
       ("awid","o",6),("awlock","o",1),("awqos","o",4),("awvalid","o",1),("awready","i",1),
       ("wdata","o",128),("wstrb","o",16),("wlast","o",1),("wvalid","o",1),("wready","i",1),
       ("bresp","i",2),("bid","i",6),("bvalid","i",1),("bready","o",1)]
for i in range(NM):
    for n, d, w in sig:
        P.append(f"  {'output wire' if d=='o' else 'input  wire'} {'['+str(w-1)+':0] ' if w>1 else ''}m{i}_axi_{n}")
body = []
body.append(f"""  localparam NM = {NM};
  reg [31:0] r_blo [0:NM-1], r_bhi [0:NM-1], r_mask [0:NM-1], r_nb [0:NM-1];
  reg [7:0]  r_len [0:NM-1];
  reg [6:0]  r_ot  [0:NM-1];
  reg        r_mode[0:NM-1];
  reg [NM-1:0] go;
  reg [63:0] fr;
  wire [NM-1:0] busy;
  wire [31:0] cyc [0:NM-1], beats [0:NM-1], lmin [0:NM-1], lmax [0:NM-1], errs [0:NM-1];
  wire [47:0] lsum [0:NM-1];

  // AXI-Lite: a write needs address and data together; one read at a time
  wire wr = s_axil_awvalid && s_axil_wvalid && !s_axil_bvalid;
  assign s_axil_awready = wr;
  assign s_axil_wready  = wr;
  assign s_axil_bresp   = 2'b00;
  assign s_axil_arready = !s_axil_rvalid;
  assign s_axil_rresp   = 2'b00;
  wire [3:0] wm = s_axil_awaddr[11:8] - 4'd1, rm = s_axil_araddr[11:8] - 4'd1;
  integer k;
  always @(posedge clk) begin
    if (!rstn) begin
      s_axil_bvalid <= 1'b0; s_axil_rvalid <= 1'b0; go <= 0; fr <= 64'd0;
    end else begin
      fr <= fr + 64'd1;
      go <= 0;
      if (s_axil_bvalid && s_axil_bready) s_axil_bvalid <= 1'b0;
      if (wr) begin
        s_axil_bvalid <= 1'b1;
        if (s_axil_awaddr[13:8] == 6'd0) begin
          if (s_axil_awaddr[7:0] == 8'h04 && s_axil_wdata[0]) go <= s_axil_wdata[4 +: NM];
        end else if (wm < NM) begin
          case (s_axil_awaddr[7:0])
            8'h00: r_blo[wm]  <= s_axil_wdata;
            8'h04: r_bhi[wm]  <= s_axil_wdata;
            8'h08: r_mask[wm] <= s_axil_wdata;
            8'h0C: r_len[wm]  <= s_axil_wdata[7:0];
            8'h10: r_ot[wm]   <= s_axil_wdata[6:0];
            8'h14: r_nb[wm]   <= s_axil_wdata;
            8'h18: r_mode[wm] <= s_axil_wdata[0];
            default: ;
          endcase
        end
      end
      if (s_axil_rvalid && s_axil_rready) s_axil_rvalid <= 1'b0;
      if (s_axil_arvalid && !s_axil_rvalid) begin
        s_axil_rvalid <= 1'b1;
        s_axil_rdata  <= 32'd0;
        if (s_axil_araddr[13:8] == 6'd0) begin
          case (s_axil_araddr[7:0])
            8'h00: s_axil_rdata <= 32'h4E454754;   // "TGEN"
            8'h08: s_axil_rdata <= {{{{(32-NM){{1'b0}}}}, busy}};
            8'h0C: s_axil_rdata <= fr[31:0];
            8'h10: s_axil_rdata <= fr[63:32];
            8'h14: s_axil_rdata <= NM;
            default: ;
          endcase
        end else if (rm < NM) begin
          case (s_axil_araddr[7:0])
            8'h00: s_axil_rdata <= r_blo[rm];
            8'h04: s_axil_rdata <= r_bhi[rm];
            8'h08: s_axil_rdata <= r_mask[rm];
            8'h0C: s_axil_rdata <= {{24'd0, r_len[rm]}};
            8'h10: s_axil_rdata <= {{25'd0, r_ot[rm]}};
            8'h14: s_axil_rdata <= r_nb[rm];
            8'h18: s_axil_rdata <= {{31'd0, r_mode[rm]}};
            8'h20: s_axil_rdata <= cyc[rm];
            8'h24: s_axil_rdata <= beats[rm];
            8'h28: s_axil_rdata <= lsum[rm][31:0];
            8'h2C: s_axil_rdata <= {{16'd0, lsum[rm][47:32]}};
            8'h30: s_axil_rdata <= lmin[rm];
            8'h34: s_axil_rdata <= lmax[rm];
            8'h38: s_axil_rdata <= errs[rm];
            default: ;
          endcase
        end
      end
    end
  end
""")
for i in range(NM):
    conns = ",\n      ".join(f".m_axi_{n}(m{i}_axi_{n})" for n, _, _ in sig)
    body.append(f"""  tg_master u_m{i} (.clk(clk), .rstn(rstn), .start(go[{i}]), .base({{r_bhi[{i}][7:0], r_blo[{i}]}}),
      .mask(r_mask[{i}]), .len(r_len[{i}]), .ot(r_ot[{i}]), .nburst(r_nb[{i}]), .mode(r_mode[{i}]),
      .busy(busy[{i}]), .cyc(cyc[{i}]), .beats(beats[{i}]), .lat_sum(lsum[{i}]), .lat_min(lmin[{i}]),
      .lat_max(lmax[{i}]), .errs(errs[{i}]),
      {conns});
""")
src = ("// Generated by gen_top.py: the DDR bandwidth test's top (registers + traffic generators).\n"
       + __doc__.replace("\n", "\n// ").join(["// ", "\n"])
       + "module tg_top (\n" + ",\n".join(P) + "\n);\n" + "".join(body) + "endmodule\n")
open("tg_top.v", "w").write(src)
