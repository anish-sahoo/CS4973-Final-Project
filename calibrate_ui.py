#!/usr/bin/env python3
"""
Real-time Gaze Tracker with Calibration

This script uses a trained gaze estimation model and calibration data
to track where the user is looking on the screen in real-time.
"""

import sys
import os
import argparse

# Add src to path to enable imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import cv2
    import torch
    import pyautogui
    import numpy as np
    import mediapipe as mp
    from models.gaze_model import GazeNet
    from calibration.calibration import Calibration
    from utils import crop_eye_from_frame, extract_eye_landmarks, estimate_head_pose
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("\nPlease install required dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

mp_face_mesh = mp.solutions.face_mesh

def main():
    parser = argparse.ArgumentParser(description='Real-time gaze tracker')
    parser.add_argument('--model', default='gaze_two_eye.pth', 
                       help='Path to trained model')
    parser.add_argument('--calibration', default='calibration.npz', 
                       help='Path to calibration file')
    parser.add_argument('--no-cursor', action='store_true',
                       help='Disable cursor movement (just show visualization)')
    args = parser.parse_args()
    
    # Check if calibration file exists
    if not os.path.exists(args.calibration):
        print(f"Error: Calibration file '{args.calibration}' not found!")
        print("\nPlease run calibration first:")
        print("  python collect_calibration.py")
        sys.exit(1)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = GazeNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print("✓ Model loaded")
    
    # Load calibration
    print("Loading calibration...")
    calib = Calibration()
    calib.load(args.calibration)
    print("✓ Calibration loaded")
    
    # Setup webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)
    
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    
    print("\n" + "="*60)
    print("GAZE TRACKER RUNNING")
    print("="*60)
    print("Press ESC to quit")
    print("="*60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame from webcam")
            break
        
        # Process frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            left_idxs = [33, 133, 160, 159, 158, 144, 145, 153, 154]
            right_idxs = [263, 362, 387, 386, 385, 373, 374, 380, 381]
            
            try:
                # Extract eye regions
                left_lm, right_lm = extract_eye_landmarks(lm, left_idxs, right_idxs)
                left_crop = crop_eye_from_frame(frame, left_lm)
                right_crop = crop_eye_from_frame(frame, right_lm)
                
                # Prepare tensors
                left_t = torch.from_numpy(left_crop).unsqueeze(0).to(device).float()
                right_t = torch.from_numpy(right_crop).unsqueeze(0).to(device).float()
                
                # Estimate head pose
                head_pitch, head_yaw = estimate_head_pose(lm, frame.shape[:2])
                head = torch.tensor([head_pitch, head_yaw], dtype=torch.float32).unsqueeze(0).to(device)
                
                # Get gaze prediction
                with torch.no_grad():
                    gaze_pred = model(left_t, right_t, head).detach().cpu().numpy()[0]
                
                # Apply calibration to get screen coordinates
                screen_point = calib.predict(gaze_pred)
                screen_x, screen_y = int(screen_point[0]), int(screen_point[1])
                
                # Move cursor (if enabled)
                if not args.no_cursor:
                    try:
                        pyautogui.moveTo(screen_x, screen_y)
                    except:
                        pass  # Ignore pyautogui errors
                
                # Draw gaze point on frame
                h, w = frame.shape[:2]
                # Scale screen coordinates to frame coordinates
                frame_x = int((screen_x / pyautogui.size()[0]) * w)
                frame_y = int((screen_y / pyautogui.size()[1]) * h)
                
                # Draw visualization
                cv2.circle(frame, (frame_x, frame_y), 10, (0, 255, 0), 2)
                cv2.line(frame, (frame_x - 15, frame_y), (frame_x + 15, frame_y), (0, 255, 0), 2)
                cv2.line(frame, (frame_x, frame_y - 15), (frame_x, frame_y + 15), (0, 255, 0), 2)
                
                # Display gaze info
                cv2.putText(frame, f"Gaze: ({screen_x}, {screen_y})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Pitch: {gaze_pred[0]:.3f}, Yaw: {gaze_pred[1]:.3f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            except Exception as e:
                # Display error on frame
                cv2.putText(frame, "Processing error", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # No face detected
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow("Gaze Tracker (Press ESC to quit)", frame)
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Gaze tracker stopped")

if __name__ == '__main__':
    main()
