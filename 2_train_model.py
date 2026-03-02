import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Enable GPU Memory Growth (fixes crashes on standard consumer GPUs)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[!] GPU Memory Growth Enabled.")
    except Exception as e:
        print(f"[!] Warning limiting GPU growth: {e}")
else:
    print("[!] No GPU found, falling back to CPU. Ensure CUDA/cuDNN is installed if you want GPU acceleration.")

DATASET_PATH = "dataset"
MODEL_FILE = "sign_lstm_model.keras"
SEQUENCE_LENGTH = 30

def get_data():
    if not os.path.exists(DATASET_PATH):
        return None, None, None

    actions = np.array(os.listdir(DATASET_PATH))
    if len(actions) < 2:
        return actions, None, None

    label_map = {label:num for num, label in enumerate(actions)}
    
    sequences, labels = [], []
    for action in actions:
        action_path = os.path.join(DATASET_PATH, action)
        for sequence in np.array(os.listdir(action_path)):
            if not sequence.endswith('.npy'):
                continue
            res = np.load(os.path.join(action_path, sequence))
            sequences.append(res)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    return actions, X, y

def main():
    print(f"\n{'='*50}")
    print(f" 🧠 LSTM NEURAL NETWORK TRAINING  ")
    print(f"{'='*50}\n")

    actions, X, y = get_data()
    
    if X is None or len(actions) < 2:
        print("[!] FATAL: You need at least 2 different signs recorded in 'dataset/'.")
        print("    Please run '1_collect_data.py' first to build sequence arrays.")
        return
        
    print(f"[*] Found {len(actions)} signs: {', '.join(actions)}")
    print(f"[*] Dataset Input Shape: {X.shape}") # Should be (num_sequences, 30, 126)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Data Augmentation: subtle noise to avoid overfitting without masking sign differences ---
    print("[*] Augmenting training data with noise...")
    noise = np.random.normal(0, 0.005, X_train.shape)  # 0.5% noise — subtle enough to preserve sign differences
    X_train = np.concatenate([X_train, X_train + noise])
    y_train = np.concatenate([y_train, y_train])
    print(f"    [+] Augmented training set size: {X_train.shape[0]} sequences")

    print("\n[*] Compiling Spatial-Temporal LSTM Model...")
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    print("\n[*] Training Model (200 Epochs — full run for best discrimination)...")
    model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

    print(f"\n[*] Saving best model to '{MODEL_FILE}'...")
    model.save(MODEL_FILE)
    
    # Save the action labels so the Inference UI app knows what to map the index output to
    np.save('actions.npy', actions)
    print("    [+] Actions map saved to 'actions.npy'")
    
    print("\n[✔] SUCCESS! Deep Learning LSTM Model is completely ready.")

if __name__ == "__main__":
    main()
