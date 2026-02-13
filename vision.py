import time
import face_recognition
import cv2
import os
import warnings
import sys

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# GLOBAL SETTINGS
# =========================
# Path to your reference photo
REFERENCE_IMAGE = "roshan.png"
CAMERA_INDEX = 0

# =========================
# INITIALIZE ENCODING
# =========================
ROSHAN_ENCODING = None

print(f"\n[A.I.S.A. Vision]: Initializing...")

if os.path.exists(REFERENCE_IMAGE):
    try:
        print(f"[A.I.S.A. Vision]: Encoding {REFERENCE_IMAGE}...")
        roshan_image = face_recognition.load_image_file(REFERENCE_IMAGE)
        encodings = face_recognition.face_encodings(roshan_image)
        
        if encodings:
            ROSHAN_ENCODING = encodings[0]
            print("Success: Roshan's identity profile loaded.")
        else:
            print("Error: No face detected in the reference photo. Try a clearer picture.")
            sys.exit(1)
    except Exception as e:
        print(f"Error during encoding: {e}")
        sys.exit(1)
else:
    print(f"Error: '{REFERENCE_IMAGE}' not found. Place your photo in the project folder.")
    sys.exit(1)

# =========================
# IDENTIFICATION FUNCTION
# =========================
def identify_person():
    """Captures a frame and compares faces against the reference encoding."""
    video_capture = cv2.VideoCapture(CAMERA_INDEX)
    if not video_capture.isOpened():
        return "Camera_Error"
    time.sleep(0.6) 
    ret, frame = video_capture.read()
    video_capture.release()

    if not ret or frame is None:
        return "Capture_Error"
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    if not face_encodings:
        return "None"
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([ROSHAN_ENCODING], face_encoding, tolerance=0.5)
        if True in matches:
            return "Roshan"
    return "Stranger"

# =========================
# MONITORING LOOP
# =========================
def start_surveillance():
    print(f"\n{'='*40}")
    print("A.I.S.A. VISION ACTIVE")
    print("Press Ctrl+C to exit")
    print(f"{'='*40}\n")
    
    try:
        while True:
            # We check every 5 seconds for this standalone test
            person = identify_person()
            timestamp = time.strftime('%H:%M:%S')
            
            if person == "Roshan":
                print(f"[{timestamp}] Result:Roshan identified.")
            elif person == "Stranger":
                print(f"[{timestamp}] Result:UNKNOWN PERSON DETECTED.")
            elif person == "None":
                print(f"[{timestamp}] Result: ... No one in view.")
            else:
                print(f"[{timestamp}] Result:{person}")

            time.sleep(4) # Delay between scans
            
    except KeyboardInterrupt:
        print("\n\n[A.I.S.A.]: Vision systems powering down. Goodbye.")

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    start_surveillance()