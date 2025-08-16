import numpy as np  
from PIL import Image  
import os  
from os.path import join  
  
def standardize_image_and_mask_formats(data_root_dir):  
    """  
    Script para estandarizar formatos:  
    - Imágenes: 3 canales RGB  
    - Máscaras: 1 canal (escala de grises) con valores binarios 0 y 255  
    """  
    imgs_dir = join(data_root_dir, 'imgs')  
    masks_dir = join(data_root_dir, 'masks')  
      
    # Crear directorios de backup  
    imgs_backup_dir = join(data_root_dir, 'imgs_backup_format')  
    masks_backup_dir = join(data_root_dir, 'masks_backup_format')  
      
    os.makedirs(imgs_backup_dir, exist_ok=True)  
    os.makedirs(masks_backup_dir, exist_ok=True)  
      
    print("=== Procesando Imágenes para 3 Canales RGB ===")  
      
    # Procesar imágenes para asegurar 3 canales RGB  
    for filename in os.listdir(imgs_dir):  
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):  
            img_path = join(imgs_dir, filename)  
            backup_path = join(imgs_backup_dir, filename)  
              
            # Hacer backup  
            img = Image.open(img_path)  
            img.save(backup_path)  
              
            # Cargar y analizar imagen  
            img_array = np.array(img)  
            print(f"\nImagen {filename}:")  
            print(f"  Formato original: {img_array.shape}, modo: {img.mode}")  
              
            # Convertir a RGB de 3 canales  
            if img.mode != 'RGB':  
                img_rgb = img.convert('RGB')  
                print(f"  Convertido de {img.mode} a RGB")  
            else:  
                img_rgb = img  
              
            # Verificar que tenga exactamente 3 canales  
            img_rgb_array = np.array(img_rgb)  
            if len(img_rgb_array.shape) == 3 and img_rgb_array.shape[2] == 3:  
                print(f"  ✅ Formato final: {img_rgb_array.shape} (3 canales RGB)")  
            else:  
                print(f"  ⚠️ Formato inesperado: {img_rgb_array.shape}")  
              
            # Guardar imagen procesada  
            img_rgb.save(img_path, 'PNG')  
      
    print("\n=== Procesando Máscaras para 1 Canal (Escala de Grises) ===")  
      
    # Procesar máscaras para asegurar 1 canal con valores binarios  
    for filename in os.listdir(masks_dir):  
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):  
            mask_path = join(masks_dir, filename)  
            backup_path = join(masks_backup_dir, filename)  
              
            # Hacer backup  
            mask = Image.open(mask_path)  
            mask.save(backup_path)  
              
            # Cargar y analizar máscara  
            mask_array = np.array(mask)  
            print(f"\nMáscara {filename}:")  
            print(f"  Formato original: {mask_array.shape}, modo: {mask.mode}")  
            print(f"  Valores únicos: {np.unique(mask_array)}")  
              
            # Procesar según el formato actual  
            if len(mask_array.shape) > 2:  
                # Múltiples canales - tomar el primer canal  
                mask_single = mask_array[:, :, 0]  
                print(f"  Reducido de {mask_array.shape} a {mask_single.shape}")  
            else:  
                # Ya es 2D  
                mask_single = mask_array  
              
            # Convertir a binario: mantener solo 0 y 255  
            # Cualquier valor > 0 se convierte en 255  
            mask_binary = np.where(mask_single > 0, 255, 0).astype(np.uint8)  
              
            print(f"  Valores finales: {np.unique(mask_binary)}")  
            print(f"  ✅ Formato final: {mask_binary.shape} (1 canal, valores 0-255)")  
              
            # Guardar máscara procesada como escala de grises  
            Image.fromarray(mask_binary, mode='L').save(mask_path, 'PNG')  
      
    print(f"\n=== Procesamiento Completado ===")  
    print(f"Backups guardados en:")  
    print(f"  - Imágenes: {imgs_backup_dir}")  
    print(f"  - Máscaras: {masks_backup_dir}")  
      
    # Verificación final  
    print(f"\n=== Verificación Final ===")  
    verify_final_formats(data_root_dir)  
  
def verify_final_formats(data_root_dir):  
    """  
    Verificar que los formatos finales sean correctos  
    """  
    imgs_dir = join(data_root_dir, 'imgs')  
    masks_dir = join(data_root_dir, 'masks')  
      
    print("Verificando imágenes...")  
    for filename in os.listdir(imgs_dir)[:3]:  # Solo primeras 3  
        if filename.endswith('.png'):  
            img_path = join(imgs_dir, filename)  
            img = Image.open(img_path)  
            img_array = np.array(img)  
            print(f"  {filename}: {img_array.shape}, modo: {img.mode}")  
      
    print("Verificando máscaras...")  
    for filename in os.listdir(masks_dir)[:3]:  # Solo primeras 3  
        if filename.endswith('.png'):  
            mask_path = join(masks_dir, filename)  
            mask = Image.open(mask_path)  
            mask_array = np.array(mask)  
            print(f"  {filename}: {mask_array.shape}, modo: {mask.mode}, valores: {np.unique(mask_array)}")  
  
if __name__ == "__main__":  
    # Ejecutar el script  
    data_root_dir = "data"  # Ajusta esta ruta según tu configuración  
      
    print("Iniciando estandarización de formatos...")  
    print("- Imágenes: se mantendrán en 3 canales RGB")  
    print("- Máscaras: se convertirán a 1 canal con valores 0 y 255")  
      
    standardize_image_and_mask_formats(data_root_dir)  
      
    print("\n¡Listo! Ahora puedes ejecutar el entrenamiento de SegCaps.")  
    print("Recuerda limpiar los archivos NPZ:")  
    print("rm -rf data/np_files*")