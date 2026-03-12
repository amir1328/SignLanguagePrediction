import requests
import os
import cv2
import numpy as np
import tempfile
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer

def test_sign_mt_api(text="hello"):
    print(f"Testing Sign.mt API for: {text}")
    url = f"https://us-central1-sign-mt.cloudfunctions.net/spoken_text_to_signed_pose?text={text}&spoken=en&signed=ase"
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"API Error: {response.status_code}")
        return
    
    temp_pose = "temp.pose"
    with open(temp_pose, "wb") as f:
        f.write(response.content)
    
    print("Pose data saved. Parsing...")
    try:
        with open(temp_pose, "rb") as f:
            p = Pose.read(f.read())
        
        # p.body.data is a numpy array (frames, persons, points, dims)
        print(f"Pose shape: {p.body.data.shape}")
        
        # Simple visualizer check (if pose_format has it)
        # Or manual OpenCV render
        v = PoseVisualizer(p)
        # v.save_video("test_pose.mp4", v.draw())
        # Let's try manual render to be safe and compatible with our app
        
        frames, persons, points, dims = p.body.data.shape
        width, height = 640, 480
        out = cv2.VideoWriter("test_rendered.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
        
        for f_idx in range(frames):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            # data for first person
            pts = p.body.data[f_idx][0] # (points, 3) where last is confidence
            
            for pt_idx in range(points):
                x, y, conf = pts[pt_idx][:3]
                if conf > 0.1:
                    px = int(x * width)
                    py = int(y * height)
                    cv2.circle(img, (px, py), 2, (0, 255, 255), -1)
            
            out.write(img)
        
        out.release()
        print("Success! test_rendered.mp4 created.")
        
    except Exception as e:
        print(f"Parsing/Rendering error: {e}")
    finally:
        if os.path.exists(temp_pose):
            os.remove(temp_pose)

if __name__ == "__main__":
    test_sign_mt_api("testing AI signs")
