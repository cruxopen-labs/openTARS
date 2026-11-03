// libtmdrv: user-space access to TARS-mini on the KV260 (Linux, run as root).
//   registers   /dev/mem at 0xA000_0000 (16 KB, the PL's M_AXI_HPM0_FPD window), uncached
//   memory      a buffer of 2 MB hugepages (MAP_HUGETLB), each page's physical address from
//               /proc/self/pagemap: the core's page table maps its flat address space onto them
//   coherency   the core reaches DDR through S_AXI_HP0_FPD, which does not snoop the CPU caches:
//               tm_clean writes CPU data back before the core reads it, tm_inval drops the CPU's
//               copies before reading what the core wrote (DC CVAC / DC CIVAC, allowed at EL0)
// Build: aarch64-linux-gnu-gcc -O2 -shared -fPIC -o libtmdrv.so tmdrv.c
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif

#define CSR_BASE 0xA0000000UL
#define CSR_SIZE 0x4000UL
#define HPAGE    (2UL << 20)

static volatile uint32_t *csr;

int tm_open(void) {
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) return -errno;
  void *p = mmap(NULL, CSR_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, CSR_BASE);
  int e = errno;
  close(fd);
  if (p == MAP_FAILED) return -e;
  csr = (volatile uint32_t *)p;
  return 0;
}

uint32_t tm_rd(uint32_t off) { return csr[off >> 2]; }

void tm_wr(uint32_t off, uint32_t v) {
  csr[off >> 2] = v;
  __asm__ volatile("dsb sy" ::: "memory");
}

// bytes (a multiple of 2 MB) of hugepages; ppn[i] = physical address of page i >> 21.
// Returns the buffer or NULL (err: -errno, or -1000 - i if page i is not where it should be).
void *tm_alloc(uint64_t bytes, uint64_t *ppn, int *err) {
  *err = 0;
  void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
  if (p == MAP_FAILED) { *err = -errno; return NULL; }
  if (mlock(p, bytes) != 0) { *err = -errno; munmap(p, bytes); return NULL; }
  int fd = open("/proc/self/pagemap", O_RDONLY);
  if (fd < 0) { *err = -errno; munmap(p, bytes); return NULL; }
  uint64_t n = bytes / HPAGE;
  for (uint64_t i = 0; i < n; i++) {
    volatile uint8_t *q = (volatile uint8_t *)p + i * HPAGE;
    q[0] = q[0];                                 // resident
    uint64_t va = (uint64_t)(uintptr_t)q, ent = 0;
    if (pread(fd, &ent, 8, (off_t)((va / 4096) * 8)) != 8) { *err = -errno; break; }
    // not present, or no PFN for a non-root process (PFN 0 is real DDR on the ZynqMP: its DDR starts at 0)
    if (!(ent >> 63) || ((ent & ((1ULL << 55) - 1)) == 0 && geteuid() != 0)) { *err = -1000 - (int)i; break; }
    uint64_t pa = (ent & ((1ULL << 55) - 1)) * 4096;
    if (pa % HPAGE) { *err = -2000 - (int)i; break; }
    ppn[i] = pa / HPAGE;
  }
  close(fd);
  if (*err) { munmap(p, bytes); return NULL; }
  return p;
}

static inline uint64_t line_size(void) {
  uint64_t ctr;
  __asm__ volatile("mrs %0, ctr_el0" : "=r"(ctr));
  return 4ULL << ((ctr >> 16) & 0xF);            // DminLine: words -> bytes
}

void tm_clean(void *p, uint64_t n) {
  uint64_t ls = line_size(), a = (uint64_t)(uintptr_t)p & ~(ls - 1), e = (uint64_t)(uintptr_t)p + n;
  __asm__ volatile("dsb sy" ::: "memory");
  for (; a < e; a += ls) __asm__ volatile("dc cvac, %0" ::"r"(a) : "memory");
  __asm__ volatile("dsb sy" ::: "memory");
}

void tm_inval(void *p, uint64_t n) {
  uint64_t ls = line_size(), a = (uint64_t)(uintptr_t)p & ~(ls - 1), e = (uint64_t)(uintptr_t)p + n;
  __asm__ volatile("dsb sy" ::: "memory");
  for (; a < e; a += ls) __asm__ volatile("dc civac, %0" ::"r"(a) : "memory");
  __asm__ volatile("dsb sy" ::: "memory");
}
