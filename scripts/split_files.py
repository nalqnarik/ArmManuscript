import os
import shutil

# Пути
base_dir = 'dataset/'
img_source = 'dataset/images/' # Где сейчас лежат все фото

def move_files(manifest_path, target_folder):
    os.makedirs(os.path.join(base_dir, target_folder), exist_ok=True)
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Извлекаем имя файла из строки манифеста (до табуляции)
            img_name = line.split('\t')[0]
            src = os.path.join(img_source, img_name)
            dst = os.path.join(base_dir, target_folder, img_name)
            
            if os.path.exists(src):
                shutil.copy(src, dst) # Копируем файл в новую папку

# Запускаем распределение
move_files('dataset/train_line_list.txt', 'train')
move_files('dataset/val_line_list.txt', 'val')

print("Файлы успешно распределены по папкам train и val!")