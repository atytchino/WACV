import sys
path = sys.argv[1]
# 1) read raw bytes, show the first 40 and guess encoding
with open(path, "rb") as f:
    raw = f.read()
print(f"file size: {len(raw)} bytes")
print(f"first 40 bytes (hex): {raw[:40].hex()}")
print(f"first 40 bytes (repr): {raw[:40]!r}")
nul = raw[:8192].count(0)
print(f"NUL bytes in first 8KB: {nul}")
# 2) try several encodings, count how many lines contain REALVAL or VALprobe
for enc in ("utf-8", "utf-16", "utf-16-le", "utf-8-sig", "latin-1", "cp1252"):
    try:
        txt = raw.decode(enc, errors="ignore")
        n_rv = txt.count("REALVAL")
        n_vp = txt.count("VALprobe")
        n_gap = txt.count("GAPg=")
        n_g = txt.count("g(raw/wm/gap)")
        print(f"  {enc:12s}: REALVAL={n_rv:4d}  VALprobe={n_vp:4d}  GAPg={n_gap:4d}  g(raw/wm/gap)={n_g:4d}")
    except Exception as e:
        print(f"  {enc:12s}: decode error {e}")
