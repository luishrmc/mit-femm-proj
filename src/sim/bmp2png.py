from pathlib import Path
from PIL import Image

root = Path("img")

for bmp_path in root.rglob("*.bmp"):
    png_path = bmp_path.with_suffix(".png")
    with Image.open(bmp_path) as im:
        im.save(png_path)
    print(f"{bmp_path} -> {png_path}")