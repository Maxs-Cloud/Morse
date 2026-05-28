```markdown
# Morse Decoder

Распознавание азбуки Морзе с помощью нейросети (CNN + Transformer + CTC).

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

Архитектура

· CNN: три слоя ResidualBlock с SE-вниманием и MaxPool
· Transformer: 3 энкодера, 4 головы, d_model=192
· CTC Loss: выравнивание без разметки границ символов
· SpecAugment: аугментация спектрограмм

Метрики

· Loss: CTC Loss
· Dist: среднее расстояние Левенштейна на валидации

Результаты

Лучшая модель достигает расстояния Левенштейна ~0.5 на валидационной выборке (точность символов >95%).

```
