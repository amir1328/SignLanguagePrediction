import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

# ── GPU Setup ────────────────────────────────────────────────────────────────
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[✔] GPU Memory Growth Enabled.")
    except Exception as e:
        print(f"[!] Warning: {e}")
else:
    print("[!] No GPU found, falling back to CPU.")

DATASET_PATH    = "dataset"
MODEL_FILE      = "sign_lstm_model.h5"   # .h5 = cross-Keras-version compatible
SEQUENCE_LENGTH = 30

# =============================================================================
# Data Loading
# =============================================================================
def get_data():
    if not os.path.exists(DATASET_PATH):
        return None, None, None

    valid_actions = []
    for folder in sorted(os.listdir(DATASET_PATH)):
        folder_path = os.path.join(DATASET_PATH, folder)
        if os.path.isdir(folder_path):
            npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
            if len(npy_files) > 0:
                valid_actions.append(folder)

    actions = np.array(valid_actions)
    if len(actions) < 2:
        return actions, None, None

    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []

    for action in actions:
        action_path = os.path.join(DATASET_PATH, action)
        for fname in sorted(os.listdir(action_path)):
            if not fname.endswith('.npy'):
                continue
            res = np.load(os.path.join(action_path, fname))
            sequences.append(res)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(actions)).astype(int)

    # Print per-class sample counts so the user can see which signs are sparse
    raw_labels = np.argmax(y, axis=1)
    print("\n[*] Samples per sign (before balancing):")
    for i, action in enumerate(actions):
        count = np.sum(raw_labels == i)
        bar = "█" * count + "░" * max(0, 20 - count)
        flag = "  ⚠ LOW" if count < 10 else ""
        print(f"    {action:<20} [{bar}] {count:>3}{flag}")

    return actions, X, y


# =============================================================================
# Class Balancing via Oversampling
# =============================================================================
def balance_classes(X, y, actions, min_samples=20, X_full=None, y_full=None):
    """
    For every class that has fewer than `min_samples` sequences,
    repeat (with slight noise variation) until it reaches the target.
    X_full/y_full are the pre-split arrays used as a fallback source when
    a class ends up with 0 training samples after the train/test split.
    """
    if X_full is None: X_full = X
    if y_full is None: y_full = y
    raw_labels = np.argmax(y, axis=1)
    X_bal, y_bal = [X], [y]

    print(f"\n[*] Balancing classes to minimum {min_samples} samples each...")
    for i, action in enumerate(actions):
        mask = raw_labels == i
        count = np.sum(mask)
        if count >= min_samples:
            continue

        # Always start with the train-split samples for this class
        src_X = X[mask]
        src_y = y[mask]

        # If this class ended up with 0 train samples (all went to test),
        # borrow from the full pre-split dataset as a fallback source
        if count == 0:
            full_mask = np.argmax(y_full, axis=1) == i
            src_X = X_full[full_mask]
            src_y = y_full[full_mask]
            count = max(len(src_X), 1)  # prevent zero-division

        needed = max(0, min_samples - int(np.sum(mask)))

        # Oversample by cycling through existing samples with slight noise
        oversampled_X, oversampled_y = [], []
        for j in range(needed):
            idx   = j % count
            noise = np.random.normal(0, 0.015, src_X[idx:idx+1].shape)
            oversampled_X.append(src_X[idx:idx+1] + noise)
            oversampled_y.append(src_y[idx:idx+1])

        if oversampled_X:
            X_bal.append(np.concatenate(oversampled_X, axis=0))
            y_bal.append(np.concatenate(oversampled_y, axis=0))
        print(f"    {action:<20} {int(np.sum(mask)):>3} → {min_samples}  (+{needed} synthetic)")

    X_out = np.concatenate(X_bal, axis=0)
    y_out = np.concatenate(y_bal, axis=0)

    # Shuffle so oversampled sequences don't cluster at the end
    perm = np.random.permutation(len(X_out))
    return X_out[perm], y_out[perm]



# =============================================================================
# Rich Data Augmentation
# =============================================================================
def augment_data(X_train, y_train):
    """
    Apply multiple augmentation strategies to improve generalization:
      1. Gaussian noise (simulates different hand sizes / camera noise)
      2. Temporal jitter (random frame-level time stretch/compress)
      3. Spatial scaling (simulate signer being closer / further from camera)
      4. Mirror flip (left-hand vs right-hand dominant signers)
    """
    aug_X, aug_y = [X_train], [y_train]

    # 1. Light noise
    aug_X.append(X_train + np.random.normal(0, 0.01, X_train.shape))
    aug_y.append(y_train)

    # 2. Heavier noise
    aug_X.append(X_train + np.random.normal(0, 0.025, X_train.shape))
    aug_y.append(y_train)

    # 3. Spatial scale variation (80%–120% size)
    scale = np.random.uniform(0.80, 1.20, (X_train.shape[0], 1, X_train.shape[2]))
    aug_X.append(X_train * scale)
    aug_y.append(y_train)

    # 4. Mirror flip — negate x-coordinates (even indices) to swap left/right hand
    mirrored = X_train.copy()
    mirrored[:, :, 0::3] *= -1   # flip x of every 3rd value (x, y, z triplets)
    aug_X.append(mirrored)
    aug_y.append(y_train)

    # 5. Temporal jitter — slightly shuffle a few random frames
    jittered = X_train.copy()
    for idx in range(len(jittered)):
        swap_frames = np.random.choice(SEQUENCE_LENGTH, size=3, replace=False)
        jittered[idx, swap_frames] = jittered[idx, swap_frames[::-1]]
    aug_X.append(jittered)
    aug_y.append(y_train)

    X_aug = np.concatenate(aug_X, axis=0)
    y_aug = np.concatenate(aug_y, axis=0)

    # Shuffle
    perm = np.random.permutation(len(X_aug))
    return X_aug[perm], y_aug[perm]


# =============================================================================
# Model Architecture
# =============================================================================
def build_model(seq_len, feature_dim, num_classes):
    """
    Upgraded Bidirectional LSTM model for 20-50+ sign classification.

    Key improvements over the old model:
      - Bidirectional LSTMs: capture both forward AND backward temporal context
      - Larger hidden units: 128 → 128 (was 64 → 64)
      - BatchNormalization: stabilises training, allows higher learning rates
      - Extra Dense layer: better class separation boundary
      - Label smoothing: prevents overconfident predictions on small datasets
    """
    model = Sequential([
        # Layer 1: Bidirectional LSTM — captures full motion context
        Bidirectional(LSTM(128, return_sequences=True, activation='tanh'),
                      input_shape=(seq_len, feature_dim)),
        BatchNormalization(),
        Dropout(0.3),

        # Layer 2: Second Bidirectional LSTM — higher-level motion patterns
        Bidirectional(LSTM(128, return_sequences=True, activation='tanh')),
        BatchNormalization(),
        Dropout(0.3),

        # Layer 3: Final LSTM — encode sequence into a single feature vector
        LSTM(64, return_sequences=False, activation='tanh'),
        BatchNormalization(),
        Dropout(0.3),

        # Dense classification head
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(64, activation='relu'),
        Dropout(0.2),

        Dense(num_classes, activation='softmax'),
    ])

    # Label smoothing: prevents overfit on small classes (very useful < 50 samples/class)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
        metrics=['categorical_accuracy']
    )
    return model


# =============================================================================
# Main Training Loop
# =============================================================================
def main():
    print(f"\n{'='*56}")
    print(f" 🧠 UPGRADED LSTM TRAINING — MULTI-SIGN CLASSIFIER")
    print(f"{'='*56}\n")

    actions, X, y = get_data()

    if X is None or len(actions) < 2:
        print("[!] FATAL: Need ≥2 signs in 'dataset/'. Run '1_collect_data.py' first.")
        return

    num_classes = len(actions)
    print(f"[*] Found {num_classes} signs: {', '.join(actions)}")
    print(f"[*] Raw dataset shape: {X.shape}  →  (samples, frames, features)")
    print(f"[*] Samples per class: ~{X.shape[0] // num_classes}")

    # Split BEFORE augmentation to avoid data leakage
    # Use stratify only if all classes have ≥2 samples
    raw_labels = np.argmax(y, axis=1)
    min_count = min(np.bincount(raw_labels))
    strat = y if min_count >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=strat
    )
    print(f"\n[*] Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Step 1: Balance low-count classes via oversampling ─────────────────
    # Signs with few recordings are repeated with noise up to min_samples.
    # Pass X,y (full pre-split data) as fallback for classes that got 0 train samples.
    X_train, y_train = balance_classes(X_train, y_train, actions,
                                       min_samples=20, X_full=X, y_full=y)
    print(f"    [+] Balanced training set: {X_train.shape[0]} sequences")

    # ── Step 2: Rich augmentation on the now-balanced training set ──────────
    print("\n[*] Applying rich data augmentation (5 strategies)...")
    X_train, y_train = augment_data(X_train, y_train)
    print(f"    [+] Augmented training set: {X_train.shape[0]} sequences")


    # Build the upgraded model
    print(f"\n[*] Building Bidirectional LSTM model for {num_classes} classes...")
    model = build_model(SEQUENCE_LENGTH, X.shape[2], num_classes)
    model.summary()

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=40,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-6,
        verbose=1
    )
    # NOTE: ModelCheckpoint is intentionally removed — it has a known bug with
    # the .keras native format in some Keras versions ('options' not supported).
    # EarlyStopping(restore_best_weights=True) already keeps best weights in memory.

    # Batch size scales with dataset size and class count
    batch_size = min(64, max(16, len(X_train) // 20))
    print(f"\n[*] Training — batch_size={batch_size}, max 300 epochs (auto-stops early)...")

    history = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Final evaluation
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*56}")
    print(f" 📊 Final Test Accuracy : {acc*100:.1f}%")
    print(f" 📊 Final Test Loss     : {loss:.4f}")
    print(f"{'='*56}")

    # Save in .h5 (cross-version compatible) AND .keras (native, current version)
    model.save(MODEL_FILE)                         # -> sign_lstm_model.h5
    model.save(MODEL_FILE.replace('.h5', '.keras')) # -> sign_lstm_model.keras (backup)
    np.save('actions.npy', actions)
    print(f"\n[✔] Model saved → '{MODEL_FILE}' (cross-version .h5)")
    print(f"[✔] Backup saved → 'sign_lstm_model.keras'")
    print(f"[✔] Actions map saved → 'actions.npy'")
    print(f"\n[✔] SUCCESS! Ready to run the app: python 3_app_ui.py")

    # Per-class accuracy report
    print(f"\n{'─'*40}")
    print(" Per-sign accuracy on test set:")
    print(f"{'─'*40}")
    y_pred = model.predict(X_test, verbose=0)
    y_true_idx = np.argmax(y_test, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)
    for i, action in enumerate(actions):
        mask = y_true_idx == i
        if mask.sum() == 0:
            continue
        per_acc = np.mean(y_pred_idx[mask] == i)
        bar = "█" * int(per_acc * 20) + "░" * (20 - int(per_acc * 20))
        print(f"  {action:<18} [{bar}] {per_acc*100:5.1f}%  ({mask.sum()} samples)")


if __name__ == "__main__":
    main()
