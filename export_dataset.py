import numpy as np
from PIL import Image
import os

npy_file_path = "ffhq-256.npy"

output_dir = "data/ffhq-256"
os.makedirs(output_dir, exist_ok=True)

data = np.load(npy_file_path)

for i, image_data in enumerate(data):
    if image_data.max() > 255 or image_data.min() < 0:
        image_data = (
            255
            * (image_data - image_data.min())
            / (image_data.max() - image_data.min())
        )

    image_data = image_data.astype(np.uint8)

    image = Image.fromarray(image_data)
    output_path = os.path.join(output_dir, f"{i:04d}.png") 
    image.save(output_path)

    print(f"Saved {output_path}")
