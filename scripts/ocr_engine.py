import torch
import numpy as np
from PIL import Image
import os

class OCREngine:
    def __init__(self, model_class, model_path, alphabet_path, target_h=64):
        self.target_h = target_h
        self.device = torch.device('cpu')
        
        # Загрузка алфавита
        with open(alphabet_path, 'r', encoding='utf-8') as f:
            self.alphabet = f.read()
            
        # Инициализация модели
        self.model = model_class(len(self.alphabet)).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint.get('model', checkpoint))
        self.model.eval()

    def preprocess(self, image):
        """Подготовка изображения к формату [1, 1, 64, W]"""
        img = image.convert('L')
        w, h = img.size
        new_w = max(1, int(w * (self.target_h / h)))
        img = img.resize((new_w, self.target_h), Image.Resampling.LANCZOS)
        
        img_array = (np.array(img, dtype=np.float32) / 127.5) - 1.0
        return torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)

    def decode(self, preds):
        """Жадное CTC декодирование"""
        indices = torch.argmax(preds, dim=2).squeeze(1).cpu().numpy()
        res = []
        last = 0
        for idx in indices:
            if idx != 0 and idx != last:
                res.append(self.alphabet[idx - 1])
            last = idx
        return "".join(res)

    def recognize(self, image_path):
        """Полный цикл: от пути к файлу до строки текста"""
        img = Image.open(image_path)
        img_t = self.preprocess(img)
        with torch.no_grad():
            preds = self.model(img_t)
        return self.decode(preds)