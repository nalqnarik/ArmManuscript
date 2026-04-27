# Armenian Manuscript OCR (CRNN)

An end-to-end Optical Character Recognition (OCR) system designed to transcribe handwritten Armenian texts using a **CRNN** (Convolutional Recurrent Neural Network) architecture. This project covers the full pipeline: from deep learning model training to a functional desktop GUI.

## Features
- **Architecture:** CNN (feature extraction) + RNN (sequence processing) + CTC Loss.
- **GUI:** Modern desktop interface built with **PyQt6**, featuring full Unicode support for the Armenian alphabet (Sylfaen font).

## Performance Metrics
The model currently achieves the following results on the validation set:
- **Sequence Accuracy:** ~45%
- **CER (Character Error Rate):** 20%

## Tech Stack
- **Python 3.14**
- **PyTorch** (Deep Learning framework)
- **PyQt6** (Graphical User Interface)
- **Pillow** (Image processing)
- **NumPy**
