import cv2

# Try different indices to find the iVCam
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"iVCam is at index {i}")
        cap.release()
        break
    cap.release()
