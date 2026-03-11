"""
0_reset_everything.py
=====================
This script safely DELETES all generated data so you can start fresh.
It will remove:
  - The dataset/ folder and all .npy files
  - The sign_gifs/ folder
  - The audio_signs/ folder
  - sign_lstm_model.keras
  - actions.npy
"""

import os
import shutil

def clear_directory(dir_path):
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"[*] DELETED directory: {dir_path}/")
        except Exception as e:
            print(f"[!] Errror deleting {dir_path}/: {e}")

def delete_file(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"[*] DELETED file:      {file_path}")
        except Exception as e:
            print(f"[!] Error deleting {file_path}: {e}")

print("WARNING: This will delete ALL your recorded signs, dataset files,")
print("GIFs, and your trained model.")
print()
confirmation = input("Are you sure you want to completely start fresh? (y/n): ")

if confirmation.lower() == 'y':
    print("\n--- Resetting ---")
    
    # 1. Delete directories
    clear_directory("dataset")
    clear_directory("sign_gifs")
    clear_directory("audio_signs")
    
    # 2. Delete files
    delete_file("sign_lstm_model.keras")
    delete_file("sign_lstm_model.h5")
    delete_file("actions.npy")
    delete_file("sign_zones.json")
    delete_file("sign_translation.mp4")
    
    print("\n[OK] Reset complete! Your project is now a clean slate.")
    print("Next steps:")
    print("  1. python 7_msasl_downloader.py   (download signs from MS-ASL)")
    print("     - or -")
    print("  1. python 1_collect_data.py       (record your own signs)")
    print("  2. python 2_train_model.py        (train the model)")
    print("  3. python 3_app_ui.py             (run the app)")
else:
    print("\nReset Cancelled. Your files are safe.")
