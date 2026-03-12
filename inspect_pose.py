import requests
import os
from pose_format import Pose
import numpy as np

def inspect_pose(text="hello"):
    url = f"https://us-central1-sign-mt.cloudfunctions.net/spoken_text_to_signed_pose?text={text}&spoken=en&signed=ase"
    resp = requests.get(url)
    if resp.status_code != 200:
        print("API Error")
        return
        
    p = Pose.read(resp.content)
    print(f"Header dimensions: {p.header.dimensions.width}x{p.header.dimensions.height}")
    data = p.body.data
    # print type and sample
    print(f"Data type: {type(data)}")
    
    # Try to convert to numpy if it has it, otherwise try indexing
    try:
        arr = np.array(data)
        print(f"Numpy conversion Success. Shape: {arr.shape}")
    except:
        arr = data
        print("Using data object directly")

    # Check first frame, first person
    sample = arr[0, 0]
    print(f"Sample point (first frame, first person): {sample[0]}")
    
    # Check max/min
    x_vals = arr[..., 0]
    y_vals = arr[..., 1]
    conf_vals = arr[..., 2]
    
    print(f"X range: {x_vals.min()} to {x_vals.max()}")
    print(f"Y range: {y_vals.min()} to {y_vals.max()}")
    print(f"Conf range: {conf_vals.min()} to {conf_vals.max()}")

if __name__ == "__main__":
    inspect_pose("test")
