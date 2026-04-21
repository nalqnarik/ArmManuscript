import os
import cv2

# Пути к твоим папкам
img_dir = 'dataset/images/'
label_dir = 'dataset/labels/'

# Создаем папку для текстов, если ее нет
if not os.path.exists(label_dir):
    os.makedirs(label_dir)

# Проходим по всем картинкам в папке
images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

for img_name in images:
    label_name = os.path.splitext(img_name)[0] + '.txt'
    
    # Пропускаем, если уже разметили
    if os.path.exists(os.path.join(label_dir, label_name)):
        continue
        
    # Читаем и показываем картинку
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    cv2.imshow('What is written here?', img)
    cv2.waitKey(1)
    
    print(f"Обработка: {img_name}")
    text = input("Введите армянский текст и нажмите Enter (или 'q' для выхода): ")
    
    if text.lower() == 'q':
        break
        
    # Сохраняем текст в файл
    with open(os.path.join(label_dir, label_name), 'w', encoding='utf-8') as f:
        f.write(text)
        
    cv2.destroyAllWindows()

print("Готово!")