#Entiendo que quieres verificar la Solución Fundamental 1 que mencioné anteriormente, que se refiere a verificar y corregir las máscaras de ground truth en data/masks/ para asegurar que los elementos que quieres segmentar (líneas, números, texto) estén marcados correctamente.
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Carpetas
carpeta_imagenes = "data/imgs1"
carpeta_masks = "data/masks"
carpeta_salida = "data/masks_bin"

os.makedirs(carpeta_salida, exist_ok=True)

# Función para verificar y convertir a binario
def procesar_mascara(path_mascara, path_salida):
    mask = cv2.imread(path_mascara, cv2.IMREAD_UNCHANGED)

    if mask is None:
        print(f"❌ No se pudo leer: {path_mascara}")
        return None

    # Si la máscara tiene más de un canal, tomar solo el primero
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    # Valores únicos
    valores_unicos = np.unique(mask)
    print(f"Valores únicos en {os.path.basename(path_mascara)}: {valores_unicos}")

    # Normalizar a 0 y 255
    mask_binaria = np.where(mask > 0, 255, 0).astype(np.uint8)

    # Guardar binaria
    cv2.imwrite(path_salida, mask_binaria)

    return mask, mask_binaria

# Recorrer imágenes y máscaras
for nombre_archivo in os.listdir(carpeta_masks):
    if not nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        continue

    path_mascara = os.path.join(carpeta_masks, nombre_archivo)
    path_imagen = os.path.join(carpeta_imagenes, nombre_archivo)
    path_salida = os.path.join(carpeta_salida, nombre_archivo)

    mask_original, mask_binaria = procesar_mascara(path_mascara, path_salida)

    if mask_original is None:
        continue

    # Cargar imagen original (si existe)
    img_original = cv2.imread(path_imagen)
    if img_original is not None:
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

    # Mostrar comparación
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    if img_original is not None:
        axs[0].imshow(img_original)
        axs[0].set_title("Imagen original")
    else:
        axs[0].imshow(np.zeros_like(mask_original), cmap="gray")
        axs[0].set_title("Sin imagen")

    axs[1].imshow(mask_original, cmap="gray")
    axs[1].set_title("Máscara original")

    axs[2].imshow(mask_binaria, cmap="gray")
    axs[2].set_title("Máscara binaria corregida")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
