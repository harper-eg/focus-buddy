import cv2

def test_camera_access():
    """Test camera access with different indices"""
    print("Testing camera access...")
    
    for i in range(3):  # Test camera indices 0, 1, 2
        print(f"Testing camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            print(f"✓ Camera {i} opened successfully")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera {i} can read frames (shape: {frame.shape})")
                print(f"✓ Camera {i} working correctly")
            else:
                print(f"✗ Camera {i} opened but cannot read frames")
            
            cap.release()
        else:
            print(f"✗ Camera {i} could not be opened")
    
    print("\nCamera test complete.")

if __name__ == "__main__":
    test_camera_access()