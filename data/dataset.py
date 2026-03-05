"""
╔══════════════════════════════════════════════════════════════════╗
║  Gestion des données — Article 2024  (VERSION CORRIGÉE)         ║
║  44 classes comme indiqué dans la Fig. 2 de l'article           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
from PIL import Image


# ═════════════════════════════════════════════════════════════════
#  VOCABULAIRE ARABE  —  44 caractères (Fig. 2 : Dense(44))
# ═════════════════════════════════════════════════════════════════

ARABIC_CHARS = (
    "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"   # 28 lettres de base
    "أإآةى"                            #  5 formes spéciales
    "٠١٢٣٤٥٦٧٨٩"                      # 10 chiffres arabes
    " "                                #  1 espace
)  # 28 + 5 + 10 + 1 = 44 caractères (Dense(44) dans la Fig. 2)
# 28 + 5 + 1 + 10 = 44 caractères

BLANK_IDX   = 0
CHAR2IDX    = {c: i + 1 for i, c in enumerate(ARABIC_CHARS)}
IDX2CHAR    = {i + 1: c for i, c in enumerate(ARABIC_CHARS)}
IDX2CHAR[BLANK_IDX] = "<blank>"
NUM_CLASSES = len(ARABIC_CHARS) + 1   # 45 (44 chars + 1 blank)


def encode_label(text: str) -> list:
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]


def decode_indices(indices: list) -> str:
    return "".join(IDX2CHAR.get(i, "") for i in indices if i != BLANK_IDX)


# ═════════════════════════════════════════════════════════════════
#  DÉCODAGE CTC GREEDY
# ═════════════════════════════════════════════════════════════════

def ctc_greedy_decode(y_pred: np.ndarray) -> list:
    """
    Args:
        y_pred : (batch, T, num_classes)
    Returns:
        liste de strings décodées
    """
    results = []
    for seq in y_pred:
        indices = np.argmax(seq, axis=-1)
        decoded, prev = [], BLANK_IDX
        for idx in indices:
            if idx != BLANK_IDX and idx != prev:
                decoded.append(int(idx))
            prev = idx
        results.append(decode_indices(decoded))
    return results


# ═════════════════════════════════════════════════════════════════
#  DATASET FACTICE
# ═════════════════════════════════════════════════════════════════

class FakeArabicDataset:
    def __init__(self, size=500, img_height=64, img_width=256,
                 min_len=3, max_len=10):
        self.size       = size
        self.img_height = img_height
        self.img_width  = img_width
        self.min_len    = min_len
        self.max_len    = max_len
        self.chars      = list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")

    def __len__(self):
        return self.size

    def get_batch(self, batch_size=16):
        images, labels, lengths = [], [], []
        for _ in range(batch_size):
            img = np.random.rand(self.img_height, self.img_width, 1).astype("float32")
            images.append(img)
            n = np.random.randint(self.min_len, self.max_len + 1)
            chars = np.random.choice(self.chars, n, replace=True)
            label = [CHAR2IDX[c] for c in chars]
            labels.append(label)
            lengths.append(n)
        return np.array(images, dtype="float32"), labels, lengths

    def generator(self, batch_size=16):
        while True:
            yield self.get_batch(batch_size)


# ═════════════════════════════════════════════════════════════════
#  DATASET RÉEL  (IFN/ENIT ou KALIMA)
# ═════════════════════════════════════════════════════════════════

class ArabicWordDataset:
    """
    Structure attendue du dossier :
        data_dir/
            images/
                word_001.png
            train_labels.txt    ← "word_001.png\\tكتاب"
            val_labels.txt
            test_labels.txt
    """
    def __init__(self, data_dir, split="train", img_height=64, img_width=256):
        self.data_dir   = data_dir
        self.img_height = img_height
        self.img_width  = img_width
        self.samples    = []
        self._load(split)

    def _load(self, split):
        f = os.path.join(self.data_dir, f"{split}_labels.txt")
        if not os.path.exists(f):
            f = os.path.join(self.data_dir, "labels.txt")
        if not os.path.exists(f):
            print(f"⚠️  Fichier introuvable : {f}")
            return
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if "\t" not in line:
                    continue
                name, label_str = line.split("\t", 1)
                path = os.path.join(self.data_dir, "images", name)
                if not os.path.exists(path):
                    continue
                enc = encode_label(label_str)
                if enc:
                    self.samples.append((path, label_str, enc))
        print(f"   [{split}] {len(self.samples)} samples chargés")

    def __len__(self):
        return len(self.samples)

    def _load_img(self, path):
        img = Image.open(path).convert("L")
        img = img.resize((self.img_width, self.img_height), Image.LANCZOS)
        arr = np.array(img, dtype="float32") / 255.0
        return arr[..., np.newaxis]

    def get_batch(self, indices):
        images, labels, lengths, strs = [], [], [], []
        for i in indices:
            p, s, enc = self.samples[i]
            images.append(self._load_img(p))
            labels.append(enc)
            lengths.append(len(enc))
            strs.append(s)
        return np.array(images, dtype="float32"), labels, lengths, strs

    def generator(self, batch_size=16, shuffle=True):
        idx = list(range(len(self.samples)))
        while True:
            if shuffle:
                np.random.shuffle(idx)
            for s in range(0, len(idx) - batch_size + 1, batch_size):
                yield self.get_batch(idx[s:s + batch_size])


if __name__ == "__main__":
    print(f"Vocab : {len(ARABIC_CHARS)} chars  |  NUM_CLASSES : {NUM_CLASSES}")
    ds = FakeArabicDataset(size=50)
    imgs, labels, lengths = ds.get_batch(4)
    print(f"Batch : {imgs.shape}  |  label ex : {decode_indices(labels[0])}")
    print("✅ dataset.py OK")