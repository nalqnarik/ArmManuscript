import os

label_dir = 'dataset/labels/'
alphabet = set()

# Собираем все уникальные символы из файлов разметки
for label_name in os.listdir(label_dir):
    if label_name.endswith('.txt'):
        with open(os.path.join(label_dir, label_name), 'r', encoding='utf-8') as f:
            text = f.read().strip()
            for char in text:
                alphabet.add(char)

# Сортируем и сохраняем
sorted_alphabet = sorted(list(alphabet))
with open('dataset/alphabet.txt', 'w', encoding='utf-8') as f:
    f.write("".join(sorted_alphabet))

print(f"Алфавит создан! Найдено символов: {len(sorted_alphabet)}")
print(f"Символы: {''.join(sorted_alphabet)}")