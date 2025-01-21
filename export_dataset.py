import numpy as np
from PIL import Image
import os

# .npy ファイルのパス
npy_file_path = "ffhq-256.npy"

# 出力先ディレクトリ
output_dir = "data/images"
os.makedirs(output_dir, exist_ok=True)

# .npy ファイルを読み込む
data = np.load(npy_file_path)

# データが複数の画像で構成されている場合の処理
for i, image_data in enumerate(data):
    # 必要に応じてデータを正規化 (0-255 の範囲に変換)
    if image_data.max() > 255 or image_data.min() < 0:
        image_data = (
            255
            * (image_data - image_data.min())
            / (image_data.max() - image_data.min())
        )

    # データを uint8 型に変換 (画像保存に必要)
    image_data = image_data.astype(np.uint8)

    # PIL を使って画像として保存
    image = Image.fromarray(image_data)
    output_path = os.path.join(output_dir, f"image_{i:04d}.png")  # 連番付きで保存
    image.save(output_path)

    print(f"Saved {output_path}")
