"""
╔══════════════════════════════════════════════════════════════════╗
║  ENTRAÎNEMENT CNN-BLSTM — Article 2024  (VERSION CORRIGÉE)      ║
║  Framework : TensorFlow / Keras                                  ║
║                                                                  ║
║  Hyperparamètres EXACTS de l'article :                          ║
║    Optimizer  : Nadam                                            ║
║    LR initial : 0.001                                            ║
║    Décroissance: 2% / epoch à partir de l'epoch 50              ║
║    Batch size : 16                                               ║
║    Max epochs : 200                                              ║
║    Loss       : CTC Loss                                         ║
║    Classes    : 44  (Dense final de la Fig. 2)                   ║
╚══════════════════════════════════════════════════════════════════╝

UTILISATION :
    Kaggle / Colab  →  tout est dans un seul notebook
    Local           →  python train/train_cnn_blstm_2024.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# ── Chemin projet ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cnn_blstm_2024 import build_cnn_blstm
from data.dataset import (FakeArabicDataset, ArabicWordDataset,
                           NUM_CLASSES, IDX2CHAR, ctc_greedy_decode)
from utils.metrics import evaluate_batch


# ═════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════
CONFIG = {
    # Données
    "USE_FAKE"    : True,       # False → mettre DATA_DIR
    "DATA_DIR"    : None,       # ex: "/kaggle/input/ifn-enit"
    "IMG_HEIGHT"  : 64,
    "IMG_WIDTH"   : 256,

    # Entraînement (valeurs exactes de l'article)
    "BATCH_SIZE"  : 16,
    "MAX_EPOCHS"  : 200,        # Mettre 5 pour un test rapide
    "LR_INIT"     : 0.001,
    "LR_DECAY"    : 0.02,       # 2% par epoch
    "LR_START"    : 50,         # Epoch de début de décroissance

    # Modèle (corrigé selon Fig. 2)
    "NUM_CLASSES" : NUM_CLASSES,  # 45 (44 chars + 1 blank)

    # Sauvegarde
    "SAVE_DIR"    : "checkpoints",
}


# ═════════════════════════════════════════════════════════════════
#  SCHEDULER LR  (2% décroissance à partir epoch 50)
# ═════════════════════════════════════════════════════════════════
class ArticleLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, decay_start, decay_rate, steps_per_epoch):
        super().__init__()
        self.lr              = float(lr)
        self.decay_start     = float(decay_start)
        self.decay_rate      = float(decay_rate)
        self.steps_per_epoch = float(steps_per_epoch)

    def __call__(self, step):
        epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
        return tf.cond(
            epoch < self.decay_start,
            true_fn  = lambda: self.lr,
            false_fn = lambda: self.lr * (1.0 - self.decay_rate) ** (epoch - self.decay_start)
        )

    def get_config(self):
        return {
            "lr": self.lr, "decay_start": self.decay_start,
            "decay_rate": self.decay_rate,
            "steps_per_epoch": self.steps_per_epoch
        }


# ═════════════════════════════════════════════════════════════════
#  CTC LOSS
# ═════════════════════════════════════════════════════════════════
def ctc_loss_fn(y_pred, labels_list, label_lengths, input_lengths):
    """
    Calcule la CTC Loss via tf.nn.ctc_loss
    y_pred : (batch, T, num_classes)
    """
    batch_size = len(labels_list)

    # Construire le SparseTensor (format requis par tf.nn.ctc_loss)
    indices, values = [], []
    for i, lab in enumerate(labels_list):
        for j, v in enumerate(lab):
            indices.append([i, j])
            values.append(v)

    sparse = tf.SparseTensor(
        indices     = indices,
        values      = tf.cast(values, tf.int32),
        dense_shape = [batch_size, max(label_lengths)]
    )
    sparse = tf.sparse.reorder(sparse)

    # Log pour stabilité numérique
    log_probs = tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1.0))
    log_probs_t = tf.transpose(log_probs, [1, 0, 2])   # (T, B, C)

    loss = tf.nn.ctc_loss(
        labels       = sparse,
        logits       = log_probs_t,
        label_length = tf.cast(label_lengths, tf.int32),
        logit_length = tf.cast(input_lengths, tf.int32),
        logits_time_major = True,
        blank_index  = 0
    )
    return tf.reduce_mean(loss)


# ═════════════════════════════════════════════════════════════════
#  ÉTAPE D'ENTRAÎNEMENT
# ═════════════════════════════════════════════════════════════════
@tf.function(reduce_retracing=True)
def train_step(model, optimizer, images, log_probs_fn,
               labels_sparse, input_lens, label_lens):
    with tf.GradientTape() as tape:
        y_pred = model(images, training=True)
        log_p  = tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1.0))
        log_pt = tf.transpose(log_p, [1, 0, 2])
        loss   = tf.reduce_mean(tf.nn.ctc_loss(
            labels       = labels_sparse,
            logits       = log_pt,
            label_length = label_lens,
            logit_length = input_lens,
            logits_time_major = True,
            blank_index  = 0
        ))
    grads, _ = tf.clip_by_global_norm(
        tape.gradient(loss, model.trainable_variables), 5.0
    )
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def run_epoch(model, optimizer, dataset, batch_size, n_batches, T_seq):
    total = 0.0
    for _ in range(n_batches):
        imgs, labels, label_lens = dataset.get_batch(batch_size)
        input_lens = np.full(batch_size, T_seq, dtype=np.int32)

        indices, values = [], []
        for i, lab in enumerate(labels):
            for j, v in enumerate(lab):
                indices.append([i, j])
                values.append(v)
        sparse = tf.SparseTensor(
            indices     = indices,
            values      = tf.cast(values, tf.int32),
            dense_shape = [batch_size, max(label_lens)]
        )
        sparse = tf.sparse.reorder(sparse)

        loss = train_step(
            model, optimizer,
            tf.constant(imgs, dtype=tf.float32),
            None, sparse,
            tf.constant(input_lens, dtype=tf.int32),
            tf.constant(label_lens, dtype=tf.int32)
        )
        total += float(loss)
    return total / n_batches


# ═════════════════════════════════════════════════════════════════
#  ÉVALUATION
# ═════════════════════════════════════════════════════════════════
def run_eval(model, dataset, batch_size, n_batches, T_seq):
    all_preds, all_gts, total_loss = [], [], 0.0
    for _ in range(n_batches):
        imgs, labels, label_lens = dataset.get_batch(batch_size)
        input_lens = np.full(batch_size, T_seq, dtype=np.int32)

        y_pred = model(tf.constant(imgs, dtype=tf.float32), training=False)

        # Loss
        indices, values = [], []
        for i, lab in enumerate(labels):
            for j, v in enumerate(lab):
                indices.append([i, j])
                values.append(v)
        sparse = tf.sparse.reorder(tf.SparseTensor(
            indices     = indices,
            values      = tf.cast(values, tf.int32),
            dense_shape = [batch_size, max(label_lens)]
        ))
        log_pt = tf.transpose(
            tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1.0)), [1, 0, 2]
        )
        loss = tf.reduce_mean(tf.nn.ctc_loss(
            labels=sparse, logits=log_pt,
            label_length=tf.cast(label_lens, tf.int32),
            logit_length=tf.cast(input_lens, tf.int32),
            logits_time_major=True, blank_index=0
        ))
        total_loss += float(loss)

        # Décodage
        preds = ctc_greedy_decode(y_pred.numpy())
        gts   = ["".join(IDX2CHAR.get(i,"") for i in lab if i!=0) for lab in labels]
        all_preds.extend(preds)
        all_gts.extend(gts)

    m = evaluate_batch(all_preds, all_gts)
    return total_loss / n_batches, m["cer"], m["wer"]


# ═════════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT PRINCIPAL
# ═════════════════════════════════════════════════════════════════
def train():
    print("\n" + "═" * 62)
    print("  CNN-BLSTM CORRIGÉ — Article 2024 (Bouchal & Belaid)")
    print(f"  TensorFlow {tf.__version__}  |  GPU : "
          f"{'Oui ✅' if tf.config.list_physical_devices('GPU') else 'Non (CPU)'}")
    print("═" * 62)

    BS    = CONFIG["BATCH_SIZE"]
    IMG_H = CONFIG["IMG_HEIGHT"]
    IMG_W = CONFIG["IMG_WIDTH"]
    # T = longueur séquence en sortie du modèle
    # Après 5 MaxPool (W/32) puis 2 ConvTranspose (*4) = W/8
    T_SEQ = IMG_W // 8   # 256 // 8 = 32

    # ── Données ──
    print("\n📦 Données...")
    if CONFIG["USE_FAKE"]:
        print("   Mode : DONNÉES FACTICES (pas besoin de dataset réel)")
        train_ds = FakeArabicDataset(400, IMG_H, IMG_W)
        val_ds   = FakeArabicDataset(80,  IMG_H, IMG_W)
        test_ds  = FakeArabicDataset(80,  IMG_H, IMG_W)
    else:
        print(f"   Chargement : {CONFIG['DATA_DIR']}")
        train_ds = ArabicWordDataset(CONFIG["DATA_DIR"], "train", IMG_H, IMG_W)
        val_ds   = ArabicWordDataset(CONFIG["DATA_DIR"], "val",   IMG_H, IMG_W)
        test_ds  = ArabicWordDataset(CONFIG["DATA_DIR"], "test",  IMG_H, IMG_W)

    n_train = len(train_ds) // BS
    n_val   = len(val_ds)   // BS
    n_test  = len(test_ds)  // BS
    print(f"   Train : {n_train * BS} samples  |  Val : {n_val * BS}")

    # ── Modèle ──
    print("\n🏗️  Modèle (architecture corrigée Fig. 2)...")
    model = build_cnn_blstm(
        num_classes = CONFIG["NUM_CLASSES"],
        img_height  = IMG_H,
        img_width   = IMG_W
    )
    print(f"   Paramètres : {model.count_params():,}")
    print(f"   Séquence T : {T_SEQ}  |  Classes : {CONFIG['NUM_CLASSES']}")

    # ── Optimizer + LR Scheduler ──
    lr_sched  = ArticleLRSchedule(
        lr              = CONFIG["LR_INIT"],
        decay_start     = CONFIG["LR_START"],
        decay_rate      = CONFIG["LR_DECAY"],
        steps_per_epoch = n_train
    )
    optimizer = keras.optimizers.Nadam(learning_rate=lr_sched)

    # ── Répertoire de sauvegarde ──
    os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
    best_cer, best_epoch = float("inf"), 0

    # ── Boucle d'entraînement ──
    print(f"\n🚀 Entraînement — {CONFIG['MAX_EPOCHS']} epochs, batch={BS}")
    print("─" * 62)
    print(f"{'Ep':>4} │ {'Train Loss':>10} │ {'Val Loss':>8} │ {'CER%':>6} │ {'WER%':>6}")
    print("─" * 62)

    history = {"train_loss":[], "val_loss":[], "val_cer":[], "val_wer":[]}

    for epoch in range(1, CONFIG["MAX_EPOCHS"] + 1):

        tr_loss = run_epoch(model, optimizer, train_ds, BS, n_train, T_SEQ)

        if epoch % 5 == 0 or epoch == 1:
            va_loss, va_cer, va_wer = run_eval(model, val_ds, BS, n_val, T_SEQ)
            print(f"{epoch:>4} │ {tr_loss:>10.4f} │ {va_loss:>8.4f} │"
                  f" {va_cer:>6.2f} │ {va_wer:>6.2f}")
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["val_cer"].append(va_cer)
            history["val_wer"].append(va_wer)

            if va_cer < best_cer:
                best_cer, best_epoch = va_cer, epoch
                model.save_weights(
                    os.path.join(CONFIG["SAVE_DIR"], "best.weights.h5")
                )
                print(f"       ✅ Sauvegardé — CER={va_cer:.2f}%")

    print("─" * 62)
    print(f"\n✅ Terminé ! Meilleur CER = {best_cer:.2f}% (epoch {best_epoch})")

    # ── Test final ──
    print("\n📊 Test final...")
    model.load_weights(os.path.join(CONFIG["SAVE_DIR"], "best.weights.h5"))
    te_loss, te_cer, te_wer = run_eval(model, test_ds, BS, n_test, T_SEQ)
    print(f"\n{'═'*42}")
    print(f"  CER test : {te_cer:.2f}%   (article : 11.64%)")
    print(f"  WER test : {te_wer:.2f}%   (article : 32.51%)")
    print(f"{'═'*42}")
    return history


if __name__ == "__main__":
    CONFIG["MAX_EPOCHS"] = 5    # ← Changer à 200 pour l'entraînement complet
    train()