"""
Phase 3: Model Training
=======================
Trains the RandomForest classifier on the dataset and prints a clean report.
"""

import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Configuration
DATASET_FILE = "dataset.csv"
MODEL_FILE   = "sign_model.pkl"

def main():
    print(f"\n{'='*50}")
    print(f" 🧠 SIGN LANGUAGE MODEL TRAINING  ")
    print(f"{'='*50}\n")

    # 1. Load Data
    if not os.path.exists(DATASET_FILE):
        print(f"[!] FATAL: Dataset '{DATASET_FILE}' not found.")
        print("    Please run '1_collect_data.py' to record some signs first.")
        return
        
    print(f"[*] Loading dataset from '{DATASET_FILE}'...")
    df = pd.read_csv(DATASET_FILE)
    if df.empty:
        print("[!] FATAL: Dataset is empty.")
        return

    # Info summary
    total_frames = len(df)
    labels       = sorted(df['label'].unique())
    print(f"[*] Total frames recorded: {total_frames}")
    print(f"[*] Unique signs class : {', '.join(labels)} ({len(labels)} total)")

    if len(labels) < 2:
        print("\n[!] WARNING: You only have one sign recorded.")
        print("    The model won't be able to distinguish between different signs.")
        print("    Please collect at least 2 different signs before training.")
        return

    # 2. Split Data
    print("\n[*] Splitting dataset (80% Train, 20% Test)...")
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"\n[!] FATAL SPLITTING ERROR: {e}")
        print("    This usually means you have very few frames for one of your signs.")
        return

    # 3. Train
    print("[*] Training RandomForest Classifier (n_trees=200)...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    
    # Simple terminal spinner imitation
    print("    Working", end="", flush=True)
    clf.fit(X_train, y_train)
    print(" ... Done!")

    # 4. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'-'*50}")
    print(f" MODEL PERFORMANCE REPORT ")
    print(f" Overall Accuracy: {acc*100:.2f}%")
    print(f"{'-'*50}")
    
    # Hide the warnings and long text, summarize cleanly
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    # 5. Save
    print(f"[*] Saving trained model to '{MODEL_FILE}'...")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)

    print("\n[✔] SUCCESS! Model is ready for live recognition.")
    print("    You can now run '3_live_app.py'.\n")

if __name__ == "__main__":
    main()
