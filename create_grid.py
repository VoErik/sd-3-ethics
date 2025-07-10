import os
import random
from PIL import Image
import math

def collect_image_paths(root_dir, valid_exts={'.png', '.jpg', '.jpeg', '.bmp'}):
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_exts:
                image_paths.append(os.path.join(root, file))
    return image_paths

def create_wallpaper_grid(image_paths, wallpaper_size, thumbnail_size, output_path):
    wallpaper_width, wallpaper_height = wallpaper_size
    thumb_width, thumb_height = thumbnail_size

    cols = wallpaper_width // thumb_width
    rows = wallpaper_height // thumb_height
    max_images = cols * rows

    if len(image_paths) < max_images:
        print(f"Warning: Not enough images ({len(image_paths)}) to fill the grid ({max_images}).")
        sampled_images = image_paths  # use all
    else:
        sampled_images = random.sample(image_paths, max_images)

    grid_img = Image.new('RGB', (cols * thumb_width, rows * thumb_height), color=(0, 0, 0))

    for idx, img_path in enumerate(sampled_images):
        img = Image.open(img_path).convert("RGB").resize((thumb_width, thumb_height), Image.LANCZOS)
        x = (idx % cols) * thumb_width
        y = (idx // cols) * thumb_height
        grid_img.paste(img, (x, y))

    grid_img.save(output_path, quality=95, optimize=True)
    print(f"Saved wallpaper grid to {output_path} with size {grid_img.size}")

# === CONFIGURATION ===
root_directory = r'C:\Users\erik\Downloads\images'  # Change this
wallpaper_size = (3840, 2160)          # 4K resolution
thumbnail_size = (120, 120)            # Small thumbnails, adjust as needed
output_file = 'assets/wallpaper_grid.jpg'

# === EXECUTION ===
all_images = collect_image_paths(root_directory)
create_wallpaper_grid(all_images, wallpaper_size, thumbnail_size, output_file)

