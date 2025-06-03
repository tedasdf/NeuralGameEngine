import cv2
import numpy as np
import requests
import time

url = "http://192.168.0.57/capture"

while True:
    try:
        response = requests.get(url, stream=True, timeout=5)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imshow("ESP32 CAM Capture", img)
            else:
                print("Failed to decode image")
        else:
            print(f"Failed to get image, status code: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.5)  # adjust delay to control capture rate

cv2.destroyAllWindows()
