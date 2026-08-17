# resize_orig.py — обработать orig ИДЕНТИЧНО тренеру (PadToSquareNoUpscale)
from PIL import Image
from pathlib import Path

SIZE = 256                # TLD image_size (НЕ дефолт 160!)
INTERP = Image.BILINEAR   # тренер: interpolation=Image.BILINEAR
CENTER = True             # тренер: center=True
PAD_RGB = (0, 0, 0)       # тренер: pad_value=0.0 -> pv=round(0*255)=0 -> чёрный

def pad_to_square(img):
    img = img.convert("RGB")
    w, h = img.size
    scale = SIZE / max(w, h)
    if scale < 1.0:                          # только downscale (как тренер)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        if (nw, nh) != (w, h):
            img = img.resize((nw, nh), resample=INTERP)
    else:
        nw, nh = w, h
    canvas = Image.new("RGB", (SIZE, SIZE), PAD_RGB)
    if CENTER:
        left = (SIZE - nw) // 2
        top  = (SIZE - nh) // 2
    else:
        left, top = 0, 0
    canvas.paste(img, (left, top))
    return canvas

src = Path(r"E:\ATTACK_DATA\TLD_aug_final\orig")
dst = Path(r"E:\ATTACK_DATA\TLD_aug_final\orig_256")
total = 0
for cls_dir in sorted(src.iterdir()):
    if not cls_dir.is_dir():
        continue
    out_cls = dst / cls_dir.name
    out_cls.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(cls_dir.iterdir()):
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        out = pad_to_square(Image.open(f))
        out.save(out_cls / f.name)           # то же имя (для матчинга с wm)
        n += 1
    total += n
    print(f"{cls_dir.name}: {n}")
print(f"DONE — {total} images processed")