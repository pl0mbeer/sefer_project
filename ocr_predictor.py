import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

# Константы

CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
VOCAB = ['-'] + list(CHARACTERS)
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
IMG_HEIGHT = 32
NUM_CLASSES = len(VOCAB)

# Предобработка

val_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, 256), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Архитектура модели

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.AdaptiveMaxPool2d((1, None))
        )
        self.rnn = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        out = self.fc(rnn_out)
        out = out.permute(1, 0, 2)  # (seq_len, batch, classes)
        return torch.log_softmax(out, dim=2)

# Декодирование CTC

def decode_ctc(preds):
    # preds: (seq_len, batch, num_classes) после log_softmax
    probs = torch.exp(preds)
    pred_indices = preds.argmax(dim=2)
    
    results = []
    batch_size = pred_indices.shape[1]
    for b in range(batch_size):
        text = []
        step_confs = []
        prev_idx = -1
        for t in range(pred_indices.shape[0]):
            idx = pred_indices[t, b].item()
            if idx != 0 and idx != prev_idx:
                text.append(IDX_TO_CHAR[idx])
                step_confs.append(probs[t, b, idx].item())
            prev_idx = idx
        conf = sum(step_confs) / len(step_confs) if step_confs else 0.0
        results.append((''.join(text), conf))
    return results

# Класс-обёртка для удобного вызова

class OCRPredictor:
    def __init__(self, weights_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CRNN(NUM_CLASSES).to(self.device)
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def predict(self, image_input):
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # Безопасная конвертация в uint8 (если массив float)
            if image_input.dtype != np.uint8:
                image_input = np.clip(image_input * 255, 0, 255).astype(np.uint8)
                
            if image_input.ndim == 2:
                img = Image.fromarray(image_input).convert('RGB')
            elif image_input.ndim == 3:
                if image_input.shape[2] == 1:
                    img = Image.fromarray(image_input.squeeze()).convert('RGB')
                elif image_input.shape[2] == 3:
                    # OpenCV по умолчанию хранит BGR → меняем на RGB
                    img = Image.fromarray(image_input[:, :, ::-1])
                elif image_input.shape[2] == 4:
                    img = Image.fromarray(image_input).convert('RGB')
                else:
                    raise ValueError(f"Неподдерживаемое кол-во каналов: {image_input.shape[2]}")
            else:
                raise ValueError(f"Неподдерживаемая размерность массива: {image_input.ndim}")
        else:
            raise TypeError("Ожидается str, PIL.Image или numpy.ndarray")

        tensor = val_transforms(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        return decode_ctc(output)[0]  # Возвращает (text, confidence)