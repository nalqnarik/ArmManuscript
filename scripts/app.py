import sys
import os  # Добавили импорт os
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QTextEdit)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt
from model import ArmenianCRNN
from ocr_engine import OCREngine 

def resource_path(relative_path):
    """ Получает абсолютный путь к ресурсам, работает для dev и для PyInstaller """
    try:
        # PyInstaller создает временную папку _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Инициализируем пути ОДИН РАЗ здесь
MODEL_PATH = resource_path("models/model_best_03.pth")
ALPHABET_PATH = resource_path("dataset/alphabet.txt")

class OCRGui(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine 
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Armenian OCR - OOP Version')
        self.setGeometry(100, 100, 600, 750)
        
        layout = QVBoxLayout()

        self.img_label = QLabel('Загрузите изображение')
        self.img_label.setFixedSize(580, 250)
        self.img_label.setStyleSheet("border: 2px dashed #bdc3c7; background: #ecf0f1;")
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_load = QPushButton('📁 Выбрать файл')
        self.btn_predict = QPushButton('🔍 Распознать')
        self.btn_predict.setEnabled(False)
        
        self.result_area = QTextEdit()
        self.result_area.setFont(QFont('Sylfaen', 20))

        layout.addWidget(self.img_label)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_predict)
        layout.addWidget(QLabel("Результат:"))
        layout.addWidget(self.result_area)
        
        self.setLayout(layout)

        self.btn_load.clicked.connect(self.open_file)
        self.btn_predict.clicked.connect(self.process_image)

        self.current_path = None

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.current_path = path
            pixmap = QPixmap(path)
            self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
            self.btn_predict.setEnabled(True)

    def process_image(self):
        try:
            text = self.engine.recognize(self.current_path)
            self.result_area.setText(text)
        except Exception as e:
            self.result_area.setText(f"Ошибка: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Используем переменные, которые мы определили в начале файла через resource_path
    engine = OCREngine(ArmenianCRNN, MODEL_PATH, ALPHABET_PATH)
    gui = OCRGui(engine)
    
    gui.show()
    sys.exit(app.exec())