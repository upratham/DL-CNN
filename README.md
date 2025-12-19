# Flower Image Classification (CNN & VGG16)

This repository contains a deep learning project for **multi-class flower image classification (10 classes)** using:
- a **custom Convolutional Neural Network (CNN)**, and
- **VGG16 transfer learning** (pretrained on ImageNet).

The code is implemented in **TensorFlow / Keras**.

---

## Project Structure

```text
.
├── src/                   # All source code is here
│   ├── data_preprocess.py
│   ├── models.py
│   ├── train.py
│   └── eval.ipynb
└── data/                  # Dataset folder
    └── flowers/           # Put your .jpg images here
```

---

## Notes about folders

- **`src/`** contains all Python and notebook files.
- **`data/`** contains the dataset used for training and evaluation.

---

## Dataset

- Images are loaded from: `data/flowers` (dataset folder)
- Images are resized to: **200 × 200**
- Number of classes: **10**
- Class names (in code):  
  `bougainvillea, daisies, garden_roses, gardenias, hibiscus, hydrangeas, lilies, orchids, peonies, tulip`

**Important naming note:**  
`src/data_preprocess.py` extracts the label name from each filename by splitting on the first digit.  
So filenames should start with the class name (e.g., `daisies1.jpg`, `tulip23.jpg`, etc.).

---

## Models

### 1) Custom CNN
Defined in `src/models.py` as `build_cnn(dropout_size, n_classes)`.

Main components:
- Convolution + MaxPooling blocks
- Dropout for regularization
- GlobalAveragePooling2D (lower memory than Flatten)
- Dense + Softmax output

### 2) VGG16 (Transfer Learning)
Defined in `src/models.py` as `build_VGG16(dropout_size, n_classes, trasnfer_flag=False, input_shape=(200,200,3))`.

- Uses `VGG16(weights="imagenet")`
- If `trasnfer_flag=False`: convolution base is frozen (feature extractor)
- Adds GlobalAveragePooling2D + Dense classifier head

---

## Installation

### 1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Run training using:

```bash
python src/train.py
```

Default training settings in `src/train.py`:
- Train/test split: 80/20 (stratified)
- Optimizer: Adam
- Loss: sparse_categorical_crossentropy
- Epochs: 50
- Batch size: 64

> Note: callbacks (EarlyStopping / ReduceLROnPlateau) are defined in `src/train.py` but currently commented out in `model.fit()`.

---

## Evaluation

Open and run the notebook:

```bash
jupyter notebook src/eval.ipynb
```

The notebook can be used to:
- load the trained model,
- evaluate accuracy,
- generate plots and a confusion matrix (if implemented in your notebook cells).

---

## Tips / Troubleshooting

- If you have a GPU, TensorFlow may use it automatically.  
  `src/models.py` enables GPU memory growth (helps avoid out-of-memory errors).
- If training accuracy is high but test accuracy is low, increase regularization (Dropout), add validation, and enable EarlyStopping.

---

