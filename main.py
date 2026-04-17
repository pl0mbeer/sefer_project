import os
import cv2
import numpy as np
import torch
import re
import warnings
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import pandas as pd
import sys

warnings.filterwarnings('ignore')

# НАСТРОЙКИ

YOLO_MODEL_PATH = "./yolo11_best.pt" # путь модели детекции
OCR_MODEL_PATH = "./best_ocr_crnn.pth" # путь модели ocr

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ВВОД ДАННЫХ

def get_user_inputs():
    img_dir = input("📁 Введите путь к папке с изображениями: ").strip().strip('"\'')
    prefix = input("🏷 Введите префикс для имён файлов (или нажмите Enter): ").strip()
    if not img_dir:
        raise ValueError("❌ Путь к папке не указан.")
    if not Path(img_dir).is_dir():
        raise FileNotFoundError(f"❌ Папка не найдена: {img_dir}")
    return Path(img_dir), prefix

# ЗАГРУЗКА МОДЕЛЕЙ

def load_models():
    print(f"⏳ Загрузка YOLO11 на устройстве: {DEVICE}...")
    yolo = YOLO(YOLO_MODEL_PATH)
    
    print("⏳ Загрузка CRNN...")
    try:
        from ocr_predictor import OCRPredictor
        ocr_model = OCRPredictor(OCR_MODEL_PATH)
    except ModuleNotFoundError:
        raise ImportError("❌ Файл ocr_predictor.py не найден! Положите его в одну папку с main.py")
    except Exception as e:
        raise RuntimeError(f"❌ Ошибка загрузки CRNN: {e}")
        
    print("✅ Модели успешно загружены.")
    return yolo, ocr_model

# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

# Определяем категорию уверенности
def get_confidence_category(conf):
    if conf >= 0.85:
        return "sure"
    elif conf >= 0.60:
        return "maybe"
    else:
        return "not sure"

# Удаляем недопустимые символы для имён файлов
def clean_filename(text, prefix):
    text = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', text).strip()
    if not text:
        text = "unknown"
    return f"{prefix}_{text}" if prefix else text

# Генерируем уникальное имя файла с суффиксами _1, _2 и т.д.
def get_unique_path(directory, stem, ext):
    candidate = directory / f"{stem}{ext}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{counter}{ext}"
        counter += 1
    return candidate

# ОСНОВНОЙ ПАЙПЛАЙН

def process_images(img_dir, prefix, yolo, ocr_model):
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    img_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in extensions]
    if not img_paths:
        print("⚠️ Изображений в указанной папке не найдено.")
        return []

    results = []
    print(f"📊 Найдено изображений: {len(img_paths)}. Запуск обработки...\n")

    for img_path in tqdm(img_paths, desc="Обработка"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        yolo_res = yolo(img, verbose=False, device=DEVICE)
        boxes = yolo_res[0].boxes

        if len(boxes) == 0:
            results.append({
                "original_file": img_path.name, "status": "no_detection",
                "ocr_text": "", "ocr_confidence": 0.0, "category": "no detection"
            })
            continue

        best_idx = int(boxes.conf.argmax())
        xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        yolo_conf = boxes.conf[best_idx].item()
        x1, y1, x2, y2 = xyxy
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        text, ocr_conf = ocr_model.predict(crop)
        category = get_confidence_category(ocr_conf)

        ext = img_path.suffix
        new_stem = clean_filename(text, prefix)
        new_path = get_unique_path(img_dir, new_stem, ext)

        try:
            img_path.rename(new_path)
            final_name = new_path.name
        except Exception as e:
            final_name = f"ERROR_{img_path.name}"
            print(f"⚠️ Ошибка переименования {img_path.name}: {e}")

        results.append({
            "original_file": img_path.name,
            "new_file": final_name,
            "yolo_conf": round(yolo_conf, 4),
            "ocr_text": text,
            "ocr_confidence": round(ocr_conf, 4),
            "category": category
        })

    return results

# ВЫВОД И СОХРАНЕНИЕ

def print_results(results, img_dir):
    if not results:
        print("❌ Нет результатов для вывода.")
        return

    df_results = pd.DataFrame(results)
    print("\n📊 Результаты обработки:")
    # to_string() корректно выводит таблицу в консоль
    print(df_results.to_string(index=False))

    print("\n📈 Статистика по уверенности:")
    print(df_results['category'].value_counts().to_markdown())

    report_path = img_dir / "report.csv"
    df_results.to_csv(report_path, index=False, sep=';')
    print(f"\n💾 Отчёт сохранён в: {report_path}")

    success_count = len([r for r in results if 'ERROR' not in r.get('new_file', '')])
    print(f"\n✅ Обработка завершена! Успешно переименовано: {success_count} файлов.")

# ТОЧКА ВХОДА

if __name__ == "__main__":
    try:
        IMG_DIR, PREFIX = get_user_inputs()
        yolo_model, ocr_predictor = load_models()
        results_data = process_images(IMG_DIR, PREFIX, yolo_model, ocr_predictor)
        print_results(results_data, IMG_DIR)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        sys.exit(1)