// Timing-only DRAM model for tb_core: an AXI-like front end over DRAMsim3 (LPDDR4X-4266 x32).
//
// Data never passes through here: tb_core keeps the bytes (a read returns memory as it is when
// the read returns, a write lands when it is accepted), so results cannot depend on this model.
// What this decides is when the core's port says ready and when each read returns.
//
// Front end (one per core port, the NPU clock is 1 GHz, DRAM tCK from the config):
//  - reads: a 16-byte beat maps to its DRAM line (one BL16 burst, 64 B). A beat whose line is in
//    flight waits for that burst (merge); a beat whose line was read recently (rbuf lines, a
//    controller read buffer) or sits in the write buffer is served from there; otherwise the line
//    is queued for DRAMsim3. Returns are in request order (one AXI ID), at most one per cycle,
//    each at least `fabric` cycles after its data left the DRAM (NoC + controller + PHY).
//  - writes: posted. Beats merge into 64-byte lines in a write buffer (wbuf lines); a full line,
//    an evicted line or one idle for `wflush` cycles goes to DRAMsim3 as one write burst.
//  - ready: reads while fewer than `rcap` are outstanding; writes while the buffer has room.
#include <cstdint>
#include <cstdio>
#include <deque>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "memory_system.h"

namespace {

struct Rd {
  uint64_t line;
  uint64_t ready_at;   // NPU cycle the beat may return; UINT64_MAX while its burst is pending
  uint64_t issued_at;
};

struct WLine {
  uint64_t line;
  uint64_t mask;       // bytes written (64-bit mask over the 64-byte line)
  uint64_t touched;
};

struct Model {
  std::unique_ptr<dramsim3::MemorySystem> ms;
  double tck = 1.0, acc = 0.0;
  uint64_t now = 0;
  unsigned line_bytes = 64;
  // parameters
  unsigned fabric = 40, rcap = 64, rbuf_lines = 4, wbuf_lines = 8, wflush = 64;
  // state
  std::deque<Rd> rq;                                       // reads in request order
  std::unordered_map<uint64_t, std::vector<size_t>> pend;  // line -> seq numbers waiting
  std::deque<uint64_t> issue_q;                            // lines to send to DRAMsim3
  std::deque<uint64_t> rbuf;                               // recently read lines
  std::list<WLine> wbuf;
  std::deque<uint64_t> wissue_q;                           // write lines to send
  uint64_t seq_head = 0;                                   // sequence number of rq.front()
  // statistics
  uint64_t n_rd = 0, n_rd_merge = 0, n_rd_rbuf = 0, n_rd_wbuf = 0, n_dram_rd = 0, n_dram_wr = 0;
  uint64_t n_wr = 0, lat_sum = 0, lat_max = 0;

  uint64_t line_of(uint64_t a) const { return a / line_bytes; }

  bool in_rbuf(uint64_t l) const {
    for (auto x : rbuf) if (x == l) return true;
    return false;
  }
  WLine *in_wbuf(uint64_t l) {
    for (auto &w : wbuf) if (w.line == l) return &w;
    return nullptr;
  }
  void rbuf_add(uint64_t l) {
    if (rbuf_lines == 0) return;
    for (auto it = rbuf.begin(); it != rbuf.end(); ++it)
      if (*it == l) { rbuf.erase(it); break; }
    rbuf.push_back(l);
    while (rbuf.size() > rbuf_lines) rbuf.pop_front();
  }

  void on_read_done(uint64_t addr) {
    uint64_t l = line_of(addr);
    auto it = pend.find(l);
    if (it == pend.end()) return;
    for (size_t s : it->second) {
      size_t i = s - seq_head;
      if (i < rq.size()) rq[i].ready_at = now + fabric;
    }
    pend.erase(it);
    rbuf_add(l);
  }

  bool rd_can() const { return rq.size() < rcap; }

  void rd(uint64_t addr) {
    uint64_t l = line_of(addr);
    Rd r{l, UINT64_MAX, now};
    uint64_t s = seq_head + rq.size();
    n_rd++;
    if (in_wbuf(l)) { r.ready_at = now + fabric; n_rd_wbuf++; }
    else if (pend.count(l)) { pend[l].push_back(s); n_rd_merge++; }
    else if (in_rbuf(l)) { r.ready_at = now + fabric; n_rd_rbuf++; }
    else { pend[l].push_back(s); issue_q.push_back(l); }
    rq.push_back(r);
  }

  bool wr_can() const { return wbuf.size() < wbuf_lines; }

  void wr(uint64_t addr, uint16_t strb) {
    uint64_t l = line_of(addr);
    unsigned off = static_cast<unsigned>(addr % line_bytes) & ~15u;
    uint64_t m = static_cast<uint64_t>(strb) << off;
    n_wr++;
    WLine *w = in_wbuf(l);
    if (!w) { wbuf.push_back({l, 0, now}); w = &wbuf.back(); }
    w->mask |= m;
    w->touched = now;
    // a write makes a recently read copy stale for timing purposes too: drop it
    for (auto it = rbuf.begin(); it != rbuf.end(); ++it)
      if (*it == l) { rbuf.erase(it); break; }
  }

  // full, idle or (when the buffer is full) oldest lines go to DRAM
  void flush_writes() {
    for (auto it = wbuf.begin(); it != wbuf.end();) {
      bool full = it->mask == ~0ull;
      bool idle = now - it->touched >= wflush;
      if (full || idle) { wissue_q.push_back(it->line); it = wbuf.erase(it); }
      else ++it;
    }
    if (wbuf.size() >= wbuf_lines && !wbuf.empty()) {
      wissue_q.push_back(wbuf.front().line);
      wbuf.pop_front();
    }
  }

  void tick() {
    now++;
    flush_writes();
    acc += 1.0;                                   // one 1 ns NPU cycle
    while (acc >= tck) {
      acc -= tck;
      // reads first (the core waits on them), then posted writes
      if (!issue_q.empty()) {
        uint64_t a = issue_q.front() * line_bytes;
        if (ms->WillAcceptTransaction(a, false)) { ms->AddTransaction(a, false); issue_q.pop_front(); n_dram_rd++; }
      }
      if (!wissue_q.empty()) {
        uint64_t a = wissue_q.front() * line_bytes;
        if (ms->WillAcceptTransaction(a, true)) { ms->AddTransaction(a, true); wissue_q.pop_front(); n_dram_wr++; }
      }
      ms->ClockTick();
    }
  }

  // the head read returns this cycle?
  bool pop() {
    if (rq.empty() || rq.front().ready_at > now) return false;
    uint64_t l = now - rq.front().issued_at;
    lat_sum += l;
    if (l > lat_max) lat_max = l;
    rq.pop_front();
    seq_head++;
    return true;
  }
};

Model *g = nullptr;

}  // namespace

extern "C" {

void dram_init(const char *cfg, int fabric, int rcap, int rbuf, int wbuf, int wflush) {
  g = new Model();
  g->fabric = fabric; g->rcap = rcap; g->rbuf_lines = rbuf; g->wbuf_lines = wbuf; g->wflush = wflush;
  g->ms.reset(new dramsim3::MemorySystem(
      cfg, "/tmp", [](uint64_t a) { g->on_read_done(a); }, [](uint64_t) {}));
  g->tck = g->ms->GetTCK();
  g->line_bytes = g->ms->GetBusBits() / 8 * g->ms->GetBurstLength();
}

int dram_rd_can() { return g->rd_can(); }
int dram_wr_can() { return g->wr_can(); }
void dram_rd(long long addr) { g->rd(static_cast<uint64_t>(addr)); }
void dram_wr(long long addr, int strb) { g->wr(static_cast<uint64_t>(addr), static_cast<uint16_t>(strb)); }
int dram_pop() { return g->pop(); }
void dram_tick() { g->tick(); }

void dram_report() {
  std::printf("PERF dram_line_bytes %u\n", g->line_bytes);
  std::printf("PERF dram_rd_beats %llu\n", (unsigned long long)g->n_rd);
  std::printf("PERF dram_rd_merged %llu\n", (unsigned long long)g->n_rd_merge);
  std::printf("PERF dram_rd_rbuf %llu\n", (unsigned long long)g->n_rd_rbuf);
  std::printf("PERF dram_rd_wbuf %llu\n", (unsigned long long)g->n_rd_wbuf);
  std::printf("PERF dram_rd_bursts %llu\n", (unsigned long long)g->n_dram_rd);
  std::printf("PERF dram_wr_beats %llu\n", (unsigned long long)g->n_wr);
  std::printf("PERF dram_wr_bursts %llu\n", (unsigned long long)g->n_dram_wr);
  std::printf("PERF dram_rd_lat_avg %llu\n", (unsigned long long)(g->n_rd ? g->lat_sum / g->n_rd : 0));
  std::printf("PERF dram_rd_lat_max %llu\n", (unsigned long long)g->lat_max);
}

}  // extern "C"
