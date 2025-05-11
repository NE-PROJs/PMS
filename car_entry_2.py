import cv2
from ultralytics import YOLO
import pytesseract
import os
import time
import serial
import serial.tools.list_ports
import csv
import random
from collections import Counter
import threading
from process_payment import PaymentProcessor

# ======== CONFIGURATION ========

# Configure Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLO model (relative path for portability)
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"[ERROR] Failed to load YOLO model: {e}")
    exit(1)

# Setup save directory and CSV log
save_dir = 'plates'
os.makedirs(save_dir, exist_ok=True)

csv_file = 'plates_log.csv'
if not os.path.exists(csv_file):
    try:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Plate Number', 'Payment Status', 'Timestamp', 'Amount'])
    except Exception as e:
        print(f"[ERROR] Failed to create CSV file: {e}")
        exit(1)

# ======== ARDUINO DETECTION ========
def detect_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if "Arduino" in port.description or "COM" in port.device or "USB-SERIAL" in port.description:
            return port.device
    return None

arduino = None
try:
    arduino_port = detect_arduino_port()
    if arduino_port:
        print(f"[CONNECTED] Arduino on {arduino_port}")
        arduino = serial.Serial(arduino_port, 9600, timeout=1)
        time.sleep(2)
    else:
        print("[WARNING] Arduino not detected. Gate control will be simulated.")
except Exception as e:
    print(f"[ERROR] Arduino connection failed: {e}")



payment_processor = PaymentProcessor()
payment_thread = threading.Thread(target=payment_processor.run)
payment_thread.daemon = True
payment_thread.start()



# ======== ULTRASONIC SENSOR MOCK ========
# For testing, always return "vehicle present" distance
def mock_ultrasonic_distance():
    # For debugging purposes, always report a vehicle is present (distance < 50)
    # This ensures the system always tries to detect plates in every frame
    return random.randint(10, 40)

# ======== MAIN LOOP ========
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Failed to open camera")
        exit(1)

    plate_buffer = []
    entry_cooldown = 300  # 5 minutes
    last_saved_plate = None
    last_entry_time = 0

    print("[SYSTEM] Ready. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break

        distance = mock_ultrasonic_distance()
        print(f"[SENSOR] Distance: {distance} cm")

        # Initialize results to None
        results = None

        if distance <= 50:
            try:
                # Lower confidence threshold for detection
                results = model(frame, conf=0.25)  # Reduce confidence threshold to catch more potential plates
                
                # Draw bounding boxes with more info
                annotated_frame = frame.copy()
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Get confidence
                        confidence = float(box.conf[0])
                        print(f"[DEBUG] Detection confidence: {confidence:.4f}")
                        
                        # Draw box with confidence
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"{confidence:.2f}", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # If no results from model, draw a message on frame
                if len(results) == 0 or len(results[0].boxes) == 0:
                    cv2.putText(annotated_frame, "No plate detected", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Check if coordinates are valid
                        if x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0] and x1 < x2 and y1 < y2:
                            plate_img = frame[y1:y2, x1:x2]
                            
                            # Check if plate_img is valid
                            if plate_img.size == 0:
                                continue

                            try:
                                                # Enhanced preprocessing with multiple techniques
                                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                                
                                # Resize image for better OCR if it's small
                                if plate_img.shape[0] < 50 or plate_img.shape[1] < 100:
                                    scale_factor = 2
                                    gray = cv2.resize(gray, (gray.shape[1] * scale_factor, gray.shape[0] * scale_factor), 
                                                     interpolation=cv2.INTER_CUBIC)
                                
                                # Create enhanced versions of the image
                                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                                enhanced_contrast = clahe.apply(gray)
                                
                                # Apply different thresholding techniques
                                thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                                adaptive_thresh = cv2.adaptiveThreshold(enhanced_contrast, 255, 
                                                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                                       cv2.THRESH_BINARY, 11, 2)
                                
                                # Display all preprocessing steps for debugging
                                try:
                                    cv2.imshow("Gray", gray)
                                    cv2.imshow("Enhanced Contrast", enhanced_contrast)
                                    cv2.imshow("Threshold", thresh)
                                    cv2.imshow("Adaptive Threshold", adaptive_thresh)
                                except Exception as e:
                                    print(f"[ERROR] Failed to display preprocessing images: {e}")

                                                # Enhanced OCR with multiple configurations and image preprocessing
                                # First try normal preprocessing
                                plate_text = pytesseract.image_to_string(
                                    thresh,
                                    config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                                ).strip().replace(" ", "")
                                
                                # If result is too short, try alternate preprocessing
                                if len(plate_text) < 4:
                                    # Try enhancing contrast
                                    enhanced = cv2.equalizeHist(gray)
                                    plate_text = pytesseract.image_to_string(
                                        enhanced,
                                        config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                                    ).strip().replace(" ", "")
                                    
                                    # If still short, try different PSM mode
                                    if len(plate_text) < 4:
                                        plate_text = pytesseract.image_to_string(
                                            thresh,
                                            config='--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                                        ).strip().replace(" ", "")

                                # Print raw OCR result for debugging
                                print(f"[DEBUG] Raw OCR text: '{plate_text}'")
                                
                                # Improved plate validation with more flexible pattern matching
                                valid_plate = None
                                
                                # First look for exact "RAH972U" pattern (the plate in your image)
                                # if "RAH972U" in plate_text:
                                #     valid_plate = "RAH972U"
                                #     print(f"[DEBUG] Found exact match: {valid_plate}")
                                
                                # Then try to find any "RA" pattern
                                if "RA" in plate_text:
                                    start_idx = plate_text.find("RA")
                                    plate_candidate = plate_text[start_idx:]
                                    print(f"[DEBUG] 'RA' found, candidate: {plate_candidate}")
                                    
                                    # More flexible length check (6 to 8 characters)
                                    if len(plate_candidate) >= 6:
                                        # Take up to 8 chars to account for different formats
                                        plate_candidate = plate_candidate[:min(8, len(plate_candidate))]
                                        
                                        # Check different possible formats
                                        if len(plate_candidate) >= 7:
                                            # Try standard format: 3 letters + 3 digits + 1 letter (RAH972U)
                                            prefix, middle, suffix = plate_candidate[:3], plate_candidate[3:-1], plate_candidate[-1]
                                            if (prefix.isalpha() and any(c.isdigit() for c in middle) and suffix.isalpha()):
                                                valid_plate = plate_candidate
                                                print(f"[DEBUG] Valid standard format: {valid_plate}")
                                
                                # If still not found, try more general patterns
                                if valid_plate is None and len(plate_text) >= 6:
                                    print("[DEBUG] Trying general pattern matching")
                                    
                                    # Look for pattern with 2-3 letters followed by numbers and maybe a letter
                                    for i in range(len(plate_text) - 5):
                                        # Try different lengths (6-8 chars)
                                        for length in range(6, min(9, len(plate_text) - i + 1)):
                                            candidate = plate_text[i:i+length]
                                            
                                            # Count letters and digits
                                            letters = sum(c.isalpha() for c in candidate)
                                            digits = sum(c.isdigit() for c in candidate)
                                            
                                            # Accept if has both letters and digits with reasonable distribution
                                            if letters >= 2 and digits >= 1 and letters + digits == len(candidate):
                                                valid_plate = candidate
                                                print(f"[DEBUG] Found general pattern: {valid_plate}")
                                                break
                                        if valid_plate:
                                            break
                                
                                if valid_plate:
                                    print(f"[VALID] Plate Detected: {valid_plate}")
                                    plate_buffer.append(valid_plate)

                                    # Save plate image
                                    try:
                                        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
                                        image_filename = f"{valid_plate}_{timestamp_str}.jpg"
                                        save_path = os.path.join(save_dir, image_filename)
                                        cv2.imwrite(save_path, plate_img)
                                        print(f"[IMAGE SAVED] {save_path}")
                                    except Exception as e:
                                        print(f"[ERROR] Failed to save image: {e}")

                                    # Print plate buffer for debugging
                                    print(f"[DEBUG] Current plate buffer: {plate_buffer}")
                                    
                                    # Decision after 2 captures (reduced from 3)
                                    if len(plate_buffer) >= 2:
                                        most_common = Counter(plate_buffer).most_common(1)[0][0]
                                        current_time = time.time()
                                        print(f"[DEBUG] Most common plate: {most_common}")

                                        if most_common != last_saved_plate or (current_time - last_entry_time) > entry_cooldown:
                                            # Log to CSV
                                            try:
                                                with open(csv_file, 'a', newline='') as f:
                                                    writer = csv.writer(f)
                                                    writer.writerow([most_common, 0, time.strftime('%Y-%m-%d %H:%M:%S')])
                                                print(f"[SAVED] {most_common} logged to CSV.")
                                            except Exception as e:
                                                print(f"[ERROR] Failed to write to CSV: {e}")

                                            # Gate control
                                            if arduino:
                                                try:
                                                    arduino.write(b'1')
                                                    print("[GATE] Opening gate (sent '1')")
                                                    time.sleep(15)
                                                    arduino.write(b'0')
                                                    print("[GATE] Closing gate (sent '0')")
                                                except Exception as e:
                                                    print(f"[ERROR] Arduino communication error: {e}")
                                            else:
                                                print("[GATE] Gate control simulated (Arduino not connected)")

                                            last_saved_plate = most_common
                                            last_entry_time = current_time
                                        else:
                                            print("[SKIPPED] Duplicate within 5 min window.")

                                        plate_buffer.clear()

                                try:
                                    cv2.imshow("Plate", plate_img)
                                    cv2.imshow("Processed", thresh)
                                except Exception as e:
                                    print(f"[ERROR] Failed to display image: {e}")
                            
                            except Exception as e:
                                print(f"[ERROR] Image processing error: {e}")
            
            except Exception as e:
                print(f"[ERROR] Model inference error: {e}")
                annotated_frame = frame
        else:
            annotated_frame = frame

        # Show frame
        try:
            cv2.imshow('Webcam Feed', annotated_frame)
        except Exception as e:
            print(f"[ERROR] Failed to display frame: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("[SYSTEM] Program stopped by user")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
finally:
    # ======== CLEANUP ========
    if 'cap' in locals() and cap is not None:
        cap.release()
    if arduino is not None:
        arduino.close()
    cv2.destroyAllWindows()
    payment_processor.stop()
    print("[SYSTEM] Cleanup complete")

