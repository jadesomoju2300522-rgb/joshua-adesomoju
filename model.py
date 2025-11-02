"""
model.py — training script (optional fine‑tune)

This script shows how you *could* fine‑tune a tiny CNN for facial expression recognition
on a folder-structured dataset and export it to ./models/SMILEYSAGE_v1.keras.

Notes:
- The live app uses the pre‑trained FER model from the `fer` package for inference.
- If your instructor requires a "trained model file" to be shipped, run this after
  preparing a dataset in ./data/train/<class>/*.jpg .
"""
import os, glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, utils, optimizers, callbacks

DATA_DIR = "./data/train"  # Expect subfolders named with emotion labels
MODEL_OUT = "./models/SMILEYSAGE_v1.keras"
IMG_SIZE = 48
CLASSES = ["angry","disgust","fear","happy","neutral","sad","surprise"]

def load_folder_dataset(root):
    X, y = [], []
    cls_to_idx = {c:i for i,c in enumerate(CLASSES)}
    for cls in CLASSES:
        p = os.path.join(root, cls)
        if not os.path.isdir(p):
            print(f"[warn] missing class folder: {p}")
            continue
        for fp in glob.glob(os.path.join(p, "*")):
            try:
                img = Image.open(fp).convert("L").resize((IMG_SIZE, IMG_SIZE))
                X.append(np.array(img, dtype=np.float32) / 255.0)
                y.append(cls_to_idx[cls])
            except Exception:
                pass
    X = np.array(X)[..., None]
    y = utils.to_categorical(np.array(y), num_classes=len(CLASSES))
    return X, y

def build_model(num_classes=7):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    if not os.path.isdir(DATA_DIR):
        print(f"[!] Expected dataset at {DATA_DIR}. Create folders like:")
        print("    ./data/train/happy, ./data/train/sad, ... with images inside.")
        raise SystemExit(0)
    X, Y = load_folder_dataset(DATA_DIR)
    if len(X) < 50:
        print("[!] Not enough images to train a meaningful model. Add more data.")
        raise SystemExit(0)
    xtr, xva, ytr, yva = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y.argmax(axis=1))
    model = build_model(num_classes=len(CLASSES))
    ckpt = callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True)
    model.fit(xtr, ytr, validation_data=(xva, yva), epochs=10, batch_size=64, callbacks=[ckpt])
    print(f"[ok] Saved model to {MODEL_OUT}")