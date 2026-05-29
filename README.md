```markdown
# Morse Decoder

Распознавание азбуки Морзе с помощью нейросети (BiLSTM/Transformer/Conformer + CTC).

## Установка

```bash
pip install torch torchaudio pandas numpy tqdm
```

Данные

Структура директорий:

```
morse_data/
├── combined_train_clean.csv    # колонки: id, message
├── morse_cache/
│   └── train/
│       └── *.pt                # мел-спектрограммы [80, T]
└── synthetic_cache/
    └── train/
        └── *.pt
```

CSV-файл содержит пути к аудио и текстовые метки.

Обучение

```bash
python train.py
```

При запуске можно выбрать чекпоинт для продолжения обучения или начать с нуля. Чекпоинты сохраняются в morse_checkpoints_v3/epoch_checkpoints/. Лучшая модель сохраняется в папку запуска как best_model.pt.

Рабочие модели:

·MorseCRNN v2 (morse2)

CNN + BiLSTM + CTC Loss. Более лёгкая и быстрая архитектура.

· Три двойных свёрточных блока (3×3) с MaxPool
· BiLSTM: 3 слоя, hidden_size=256
· CosineAnnealingLR, градиентный клиппинг
· Результат: расстояние Левенштейна 0.24 на валидации три слоя 

· Conformer (morse3)
Conformer Encoder + CTC. 12 блоков с self-attention (относительное позиционирование) и depthwise свёрткой (kernel 31), d_model=144. Текущий Dist 0.29, обучение продолжается, ожидается 0.18–0.25

Метрики

· Loss: CTC Loss
· Dist: среднее расстояние Левенштейна на валидации

Сравнение моделей

MorseTransformer 3.4M ~0.50 Transformer, SE‑блоки.

MorseCRNN v2 1.8M 0.24 BiLSTM, проще и точнее.

Conformer 2.7M ~0.29, Ещё учится.
```
https://github.com/Maxs-Cloud/Morse/blob/217a88b4b824be1449e60c9af27ddb63ebeda55e/Image1435.jpg

