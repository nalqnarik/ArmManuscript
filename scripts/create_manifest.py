import os
import random

# Настройки
img_dir = 'dataset/images/'
label_dir = 'dataset/labels/'
train_file = 'dataset/train_line_list.txt'
val_file = 'dataset/val_line_list.txt'

# Получаем список всех файлов
all_labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
random.shuffle(all_labels) # Перемешиваем для честности

split_idx = int(len(all_labels) * 0.85) # 85% на обучение
train_labels = all_labels[:split_idx]
val_labels = all_labels[split_idx:]

def write_manifest(file_list, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for label_name in file_list:
            img_name = label_name.replace('.txt', '.png') # проверь расширение своих фото (.jpg или .png)
            
            with open(os.path.join(label_dir, label_name), 'r', encoding='utf-8') as lbl:
                text = lbl.read().strip()
            
            # Записываем: путь_к_фото ТЕКСТ
            f.write(f"{img_name}\t{text}\n")

write_manifest(train_labels, train_file)
write_manifest(val_labels, val_file)

print(f"Готово! Создано {len(train_labels)} строк для обучения и {len(val_labels)} для проверки.")