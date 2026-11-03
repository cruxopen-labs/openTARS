"""TARS-mini: the first end-to-end slice of TARS (contract v0.1).

contract  - descriptor layout, codes, encode/decode, CRC
numerics  - the contract's fp32 steps (RNE, FTZ, no FMA)
golden    - the bit-exact golden model (DDR image + status)
compiler  - minimal compiler: DDR layout, INT8 quantization, descriptor emission
disasm    - queue listing
"""
