# sanitize_dump.py — оставить только пары, где orig И wm ровно 256×256
from PIL import Image
from pathlib import Path

DUMP = Path(r"E:\ATTACK_DATA\TLD_aug_final")
SIZE = 256

orig_root = DUMP / "orig"
wm_root   = DUMP / "wm"

kept = 0
removed_size = 0
removed_unpaired = 0

for cls_dir in sorted(orig_root.iterdir()):
    if not cls_dir.is_dir():
        continue
    cls = cls_dir.name
    for of in sorted(cls_dir.iterdir()):
        if of.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        wf = wm_root / cls / of.name
        # 1. непарные — удалить orig
        if not wf.exists():
            of.unlink()
            removed_unpaired += 1
            continue
        # 2. проверить размеры обоих
        try:
            oi = Image.open(of); ow, oh = oi.size; oi.close()
            wi = Image.open(wf); ww, wh = wi.size; wi.close()
        except Exception:
            of.unlink(); wf.unlink(); removed_size += 1
            continue
        # 3. оба должны быть ровно 256×256
        if (ow, oh) == (SIZE, SIZE) and (ww, wh) == (SIZE, SIZE):
            kept += 1
        else:
            # битый размер — удалить пару
            of.unlink()
            if wf.exists(): wf.unlink()
            removed_size += 1

# 4. подчистить wm без orig (осиротевшие wm)
for cls_dir in sorted(wm_root.iterdir()):
    if not cls_dir.is_dir():
        continue
    cls = cls_dir.name
    for wf in sorted(cls_dir.iterdir()):
        of = orig_root / cls / wf.name
        if not of.exists():
            wf.unlink()
            removed_unpaired += 1

print(f"KEPT (256x256 pairs): {kept}")
print(f"removed (bad size):   {removed_size}")
print(f"removed (unpaired):   {removed_unpaired}")