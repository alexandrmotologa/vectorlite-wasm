import os
import glob
from PIL import Image

def create_demo_gif():
    frames_dir = "docs/images/frames"
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frame_files:
        print("No frames found!")
        return

    print(f"Found {len(frame_files)} frames: {frame_files}")
    images = []
    
    # Target width for crisp, compact GitHub README rendering
    target_width = 1100
    
    for f in frame_files:
        im = Image.open(f).convert("RGBA")
        # High quality downsampling
        aspect = im.height / im.width
        new_height = int(target_width * aspect)
        resized = im.resize((target_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to RGB with dark background blend for GIF palette
        bg = Image.new("RGB", resized.size, (5, 9, 20))
        bg.paste(resized, mask=resized.split()[3])
        # Quantize to adaptive 256 colors
        quantized = bg.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        images.append(quantized)

    # Frame durations in milliseconds
    # Frame 1 (Hero studio): 2200ms
    # Frame 2 (Search results & RAG): 2400ms
    # Frame 3 (3D Orbit initial): 1200ms
    # Frame 4 (3D Orbit rotated A): 1000ms
    # Frame 5 (3D Orbit rotated B): 1200ms
    # Frame 6 (Performance Benchmarks): 2500ms
    durations = [2200, 2400, 1200, 1000, 1200, 2500]
    if len(durations) < len(images):
        durations += [1500] * (len(images) - len(durations))
    durations = durations[:len(images)]

    gif_path = "docs/images/demo.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True
    )
    print(f"Successfully generated {gif_path} (Size: {os.path.getsize(gif_path) / 1024:.1f} KB)")

    # Also save animated WebP
    webp_path = "docs/images/demo.webp"
    # For webp, use original RGBA resized for maximum visual fidelity
    rgba_frames = []
    for f in frame_files:
        im = Image.open(f).convert("RGBA")
        aspect = im.height / im.width
        new_height = int(target_width * aspect)
        rgba_frames.append(im.resize((target_width, new_height), Image.Resampling.LANCZOS))

    rgba_frames[0].save(
        webp_path,
        save_all=True,
        append_images=rgba_frames[1:],
        duration=durations,
        loop=0,
        quality=90
    )
    print(f"Successfully generated {webp_path} (Size: {os.path.getsize(webp_path) / 1024:.1f} KB)")

if __name__ == "__main__":
    create_demo_gif()
