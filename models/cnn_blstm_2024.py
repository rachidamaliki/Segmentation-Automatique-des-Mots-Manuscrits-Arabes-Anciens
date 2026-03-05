"""
╔══════════════════════════════════════════════════════════════════╗
║  CNN-BLSTM  —  Article 2024 (Bouchal & Belaid)                  ║
║  VERSION CORRIGÉE selon Fig. 2  ✅                              ║
║  Framework : TensorFlow / Keras                                  ║
║                                                                  ║
║  Encodeur  : [16 → 32 → 128 → 128 → 128]  (5 MaxPool)          ║
║  Décodeur  : 2× ConvTranspose + Skip                             ║
║  Séquence  : Dense(256)→BN→Drop(0.2)                            ║
║            → BLSTM(128) → BLSTM(64)                             ║
║            → Dense(128)→Drop→Dense(128)→Dense(44)→Softmax       ║
║  Loss      : CTC (44 classes)                                    ║
╚══════════════════════════════════════════════════════════════════╝


"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


# ═════════════════════════════════════════════════════════════════
#  BLOCS DE BASE
# ═════════════════════════════════════════════════════════════════

def conv_block(x, filters, dropout=0.2, name=""):
    """
    Conv(3×3) → BN → ReLU → MaxPool(2,2) → Dropout
    ↓ Flèche orange dans la Fig. 2
    """
    x = layers.Conv2D(filters, 3, padding="same",
                      use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    x = layers.MaxPooling2D((2, 2), name=f"{name}_pool")(x)
    x = layers.Dropout(dropout,     name=f"{name}_drop")(x)
    return x


def decoder_block(x, skip, filters, name=""):
    """
    ↑ ConvTranspose (flèche verte) + → Concat skip (flèche bleue)
    → Conv → BN → ReLU → Dropout (flèche rouge)
    """
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same",
                                use_bias=False, name=f"{name}_up")(x)
    x = layers.BatchNormalization(name=f"{name}_up_bn")(x)
    x = layers.ReLU(name=f"{name}_up_relu")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = layers.Conv2D(filters, 3, padding="same",
                      use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    x = layers.Dropout(0.2, name=f"{name}_drop")(x)
    return x


# ═════════════════════════════════════════════════════════════════
#  MODÈLE PRINCIPAL
# ═════════════════════════════════════════════════════════════════

def build_cnn_blstm(num_classes: int = 44,
                    img_height:  int = 64,
                    img_width:   int = 256) -> Model:
    """
    Architecture CNN-BLSTM conforme à la Fig. 2 (article 2024).

    Flux des dimensions (H × W) avec img 64×256 :
      Entrée  : 64 × 256
      enc1    : 32 × 128  (16 filtres)
      enc2    : 16 ×  64  (32 filtres)
      enc3    :  8 ×  32  (128 filtres) ← skip2
      enc4    :  4 ×  16  (128 filtres) ← skip1
      enc5    :  2 ×   8  (128 filtres) ← bottom (4×2×128 dans figure)
      dec1    :  4 ×  16  (128 filtres)  ↑ upsample + skip e4
      dec2    :  8 ×  32  (128 filtres)  ↑ upsample + skip e3
      reshape :  (32, 8×128) = (32, 1024) séquence temporelle T=32
    """
    inputs = keras.Input(shape=(img_height, img_width, 1),
                         name="image_input")

    # ── ENCODEUR ─────────────────────────────────────────────────
    e1 = conv_block(inputs, 16,  name="enc1")   # (32,128, 16)
    e2 = conv_block(e1,     32,  name="enc2")   # (16, 64, 32)
    e3 = conv_block(e2,     128, name="enc3")   # ( 8, 32,128) ← skip2
    e4 = conv_block(e3,     128, name="enc4")   # ( 4, 16,128) ← skip1
    e5 = conv_block(e4,     128, name="enc5")   # ( 2,  8,128) ← bottom

    # ── DÉCODEUR ─────────────────────────────────────────────────
    d1 = decoder_block(e5, e4, 128, name="dec1")  # ( 4, 16,128)
    d2 = decoder_block(d1, e3, 128, name="dec2")  # ( 8, 32,128)

    # ── RESHAPE : (B, H', W', C) → (B, W', H'×C) ────────────────
    # d2 shape : (batch, 8, 32, 128)
    #
    # Traçage des dimensions :
    #   Entrée       : H=64,  W=256
    #   Après 5 pool : H=2,   W=8      (divisé par 2^5 = 32)
    #   Après 2 dec  : H=8,   W=32     (multiplié par 2^2 = 4)
    #   → d2 shape   : (batch, H=8, W=32, C=128)
    #
    # On permute pour avoir W en axe 1 (axe temporel des LSTM)
    # puis on aplatit H et C en un seul vecteur de features
    #
    #   T    = W après décodeur = img_width  // 2**3 = 32
    #   feat = H après décodeur = img_height // 2**3 = 8  → × 128 = 1024
    #
    # ⚠️  ERREUR CORRIGÉE :
    #   Avant : H_dec = img_height // 2**5 = 2  (H après pool seulement, sans upsample)
    #   Après : H_dec = img_height // 2**3 = 8  (H après pool ET 2 ConvTranspose)

    T_seq = img_width  // (2 ** 3)   # 256 // 8  = 32  (longueur séquence)
    H_dec = img_height // (2 ** 3)   # 64  // 8  = 8   (hauteur après décodeur)
    feat  = H_dec * 128              # 8   × 128 = 1024

    # Permute : (batch, H=8, W=32, C=128) → (batch, W=32, H=8, C=128)
    x = layers.Permute((2, 1, 3), name="permute")(d2)
    # Reshape : (batch, 32, 8, 128) → (batch, 32, 1024)
    x = layers.Reshape((T_seq, feat), name="reshape")(x)
    # → (batch, T=32, features=1024)

    # ── Dense(256) → BN → Dropout(0.2)  [avant les BLSTM] ───────
    x = layers.Dense(256, use_bias=False, name="pre_dense")(x)
    x = layers.BatchNormalization(name="pre_bn")(x)
    x = layers.ReLU(name="pre_relu")(x)
    x = layers.Dropout(0.2, name="pre_drop")(x)
    # (batch, 32, 256)

    # ── BLSTM 1 : 128 unités bidirectionnel ──────────────────────
    # Sortie : 128 × 2 = 256
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True),
        name="blstm1"
    )(x)
    # (batch, 32, 256)

    # ── BLSTM 2 : 64 unités bidirectionnel ───────────────────────
    # Sortie : 64 × 2 = 128
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True),
        name="blstm2"
    )(x)
    # (batch, 32, 128)

    # ── Dense(128) → Dropout(0.2) ─────────────────────────────────
    x = layers.Dense(128, use_bias=False, name="head_dense1")(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.ReLU(name="head_relu1")(x)
    x = layers.Dropout(0.2, name="head_drop")(x)

    # ── Dense(128) ────────────────────────────────────────────────
    x = layers.Dense(128, use_bias=False, name="head_dense2")(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.ReLU(name="head_relu2")(x)

    # ── Dense(44) → Softmax → CTC ─────────────────────────────────
    outputs = layers.Dense(num_classes, activation="softmax",
                            name="output")(x)
    # (batch, 32, 44)

    return Model(inputs=inputs, outputs=outputs,
                 name="CNN_BLSTM_2024")


# ═════════════════════════════════════════════════════════════════
#  TEST
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import numpy as np

    print("=" * 65)
    print("  CNN-BLSTM CORRIGÉ — Fig. 2  (Bouchal & Belaid 2024)")
    print("=" * 65)

    model = build_cnn_blstm(num_classes=44, img_height=64, img_width=256)
    fake  = np.random.randn(4, 64, 256, 1).astype("float32")
    out   = model(fake, training=False)

    print(f"\n  Entrée   : {fake.shape}")
    print(f"  Sortie   : {out.shape}  → (batch, T=32, classes=44)")
    print(f"  Params   : {model.count_params():,}")

    print(f"\n✅ Vérification des corrections :")
    print(f"   BLSTM 1 = 128 unités  (avant : 256 ❌)")
    print(f"   BLSTM 2 =  64 unités  (avant : 128 ❌)")
    print(f"   Canaux  = 16→32→128→128→128  (avant : 64→128→256→512 ❌)")
    print(f"   Classes = 44  (avant : 80 ❌)")
    print(f"   Dropout = 0.2 (avant : 0.3 ❌)")
    print("=" * 65)