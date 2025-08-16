import os
import cv2
import numpy as np

# 📂 Carpeta con las máscaras originales exportadas desde LabelMe
input_folder = "data/masks"
# 📂 Carpeta donde se guardarán las máscaras normalizadas
output_folder = "data/masks_bin"

os.makedirs(output_folder, exist_ok=True)

# Recorremos todas las imágenes
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".png", ".jpg", ".tif")):
        continue

    path = os.path.join(input_folder, filename)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Ver valores únicos
    valores = np.unique(mask)
    print(f"{filename} → valores únicos: {valores}")

    # Convertir a binario estricto (0 y 255)
    mask_bin = (mask > 0).astype(np.uint8) * 255

    # Guardar
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, mask_bin)

print("✅ Conversión completa. Máscaras guardadas en:", output_folder)
