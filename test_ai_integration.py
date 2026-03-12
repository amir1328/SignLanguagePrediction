import os
import sys

# Add current dir to path to import 4_video_translate
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # We use importlib because the file starts with a number
    import importlib
    vt = importlib.import_module("4_video_translate")
    
    test_word = "artificial intelligence"
    print(f"Testing AI Pose generation for: {test_word}")
    
    path = vt.render_sign_mt_pose(test_word)
    
    if path and os.path.exists(path):
        print(f"SUCCESS! AI Pose rendered to: {path}")
        print(f"File size: {os.path.getsize(path)} bytes")
    else:
        print("FAILED! AI Pose was not rendered correctly.")
        
except Exception as e:
    print(f"Error during integration test: {e}")
