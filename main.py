import cv2
import time
import torch
from ultralytics import YOLO

class PhoneAlertDemo:
    def __init__(self, capture_index=0):
        # 1. Hardware Setup
        print("Is cuda available ", torch.cuda.is_available())
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Running on: {torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU'}")

        # 2. Load Models (Using Extra-Large models for best accuracy)
        print("Loading Object Detection Model (Extra-Large)...")
        self.model_obj = YOLO('yolo11x.pt')  # Most accurate - best for distinguishing phones from remotes
        
        print("Loading Pose Estimation Model (Large)...")
        self.model_pose = YOLO('yolo11l-pose.pt') 

        # 3. Camera Setup
        self.cap = cv2.VideoCapture(capture_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # COCO Class IDs
        self.CELL_PHONE_CLASS_ID = 67
        self.REMOTE_CLASS_ID = 65  # TV remote - often confused with phones 

    def run(self):
        print("Starting Demo... Press 'q' to exit, 'f' to toggle fullscreen.")
        prev_time = 0
        
        # Create named window and set to fullscreen
        window_name = 'DGX Spark - Phone Alert System'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            success, frame = self.cap.read()
            if not success:
                break

            # --- A. INFERENCE ---
            # Run Object Detection with higher confidence and larger image size
            results_obj = self.model_obj.predict(
                frame, 
                conf=0.5,  # Higher confidence threshold to reduce false positives
                iou=0.45,
                imgsz=1280,  # Larger image size for better small object detection
                device=self.device, 
                verbose=False
            )
            
            # Run Pose Detection
            results_pose = self.model_pose.predict(frame, conf=0.5, device=self.device, verbose=False)

            # --- B. ALERT LOGIC ---
            phone_detected = False
            phone_box = None
            phone_confidence = 0

            # Check if any detected object is a cell phone (and NOT a remote)
            if results_obj[0].boxes:
                for box in results_obj[0].boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Only accept cell phone class with good confidence
                    if cls_id == self.CELL_PHONE_CLASS_ID and confidence > 0.5:
                        phone_detected = True
                        phone_confidence = confidence
                        # Get coordinates to draw a special box later
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        phone_box = (int(x1), int(y1), int(x2), int(y2))
                        break # Found one, that's enough to trigger alert

            # --- C. VISUALIZATION ---
            # 1. Draw Skeletons (Pose)
            annotated_frame = results_pose[0].plot()

            # 2. Draw Objects (Standard YOLO boxes)
            # We overlay object boxes on top of the pose frame
            annotated_frame = results_obj[0].plot(img=annotated_frame)

            # 3. Draw CUSTOM ALERT if Phone Found
            if phone_detected:
                # Big Red Text Alert with confidence
                alert_text = f"WARNING: PHONE DETECTED ({phone_confidence:.1%})"
                cv2.putText(annotated_frame, alert_text, (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                
                # Draw a thick red rectangle specifically around the phone
                if phone_box:
                    cv2.rectangle(annotated_frame, (phone_box[0], phone_box[1]), 
                                  (phone_box[2], phone_box[3]), (0, 0, 255), 5)

            # 4. FPS Counter
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # --- D. DISPLAY ---
            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                # Toggle fullscreen
                current_mode = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                if current_mode == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = PhoneAlertDemo()
    app.run()
