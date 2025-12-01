#!/usr/bin/env python3
"""
Gaze Calibration Data Collection Tool

This script collects calibration data for gaze estimation by having the user
look at specific points on the screen while recording their gaze vectors.
"""

import sys
import os
import argparse
import time
import traceback

# Add src to path to enable imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import cv2
    import torch
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

class CalibrationCollector:
    def __init__(self, model_path, screen_width=1920, screen_height=1080):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading model...")
        self.model = GazeNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("✓ Model loaded")
        
        # Setup webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1, 
            refine_landmarks=True
        )
        
        # Calibration data
        self.gaze_vectors = []
        self.screen_points = []
        
        # Screen size
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # 9-point calibration grid
        self.cal_points = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),  # Top row
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),  # Middle row
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),  # Bottom row
        ]
        self.current_point = 0
    
    def get_gaze_prediction(self, frame):
        """Get gaze prediction from current frame"""
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        lm = results.multi_face_landmarks[0]
        left_idxs = [33, 133, 160, 159, 158, 144, 145, 153, 154]
        right_idxs = [263, 362, 387, 386, 385, 373, 374, 380, 381]
        
        left_lm, right_lm = extract_eye_landmarks(lm, left_idxs, right_idxs)
        left_crop = crop_eye_from_frame(frame, left_lm)
        right_crop = crop_eye_from_frame(frame, right_lm)
        
        left_t = torch.from_numpy(left_crop).unsqueeze(0).to(self.device).float()
        right_t = torch.from_numpy(right_crop).unsqueeze(0).to(self.device).float()
        
        head_pitch, head_yaw = estimate_head_pose(lm, (h, w))
        head = torch.tensor([head_pitch, head_yaw], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            gaze_pred = self.model(left_t, right_t, head).cpu().numpy()[0]
        
        return gaze_pred
    
    def collect_calibration(self):
        """Collect calibration data with visual feedback"""
        print("\n" + "="*60)
        print("CALIBRATION INSTRUCTIONS")
        print("="*60)
        print("1. Look at each RED DOT when it appears")
        print("2. Press SPACE to record your gaze for that point")
        print("3. Keep your head still during recording (~1 second)")
        print("4. Press ESC to cancel")
        print("="*60 + "\n")
        
        input("Press ENTER to start calibration...")
        
        # Create fullscreen window
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        while self.current_point < len(self.cal_points):
            # Get current calibration point
            norm_x, norm_y = self.cal_points[self.current_point]
            screen_x = int(norm_x * self.screen_width)
            screen_y = int(norm_y * self.screen_height)
            
            # Create blank screen with calibration point
            screen = np.ones((self.screen_height, self.screen_width, 3), dtype=np.uint8) * 255
            
            # Draw calibration target
            cv2.circle(screen, (screen_x, screen_y), 30, (0, 0, 255), -1)  # Red dot
            cv2.circle(screen, (screen_x, screen_y), 35, (0, 0, 0), 3)      # Black outline
            cv2.circle(screen, (screen_x, screen_y), 5, (255, 255, 255), -1)  # White center
            
            # Add instructions
            text = f"Point {self.current_point + 1} of {len(self.cal_points)}"
            cv2.putText(screen, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            cv2.putText(screen, "Look at the RED DOT", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            cv2.putText(screen, "Press SPACE when ready  |  ESC to cancel", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            
            cv2.imshow("Calibration", screen)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("\nCalibration cancelled")
                cv2.destroyAllWindows()
                return None
            
            elif key == 32:  # SPACE
                print(f"\nCollecting point {self.current_point + 1}/{len(self.cal_points)}...")
                
                # Visual countdown
                for i in range(3, 0, -1):
                    screen_countdown = screen.copy()
                    cv2.putText(screen_countdown, f"Recording in {i}...", (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                    cv2.imshow("Calibration", screen_countdown)
                    cv2.waitKey(1000)
                
                # Collect samples
                samples = []
                print("  Recording", end="", flush=True)
                
                for _ in range(30):  # Collect 30 samples (~1 second at 30 FPS)
                    ret, frame = self.cap.read()
                    if not ret:
                        continue
                    
                    gaze = self.get_gaze_prediction(frame)
                    if gaze is not None:
                        samples.append(gaze)
                    
                    # Show "recording" on screen
                    screen_rec = screen.copy()
                    cv2.putText(screen_rec, "RECORDING...", (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.imshow("Calibration", screen_rec)
                    cv2.waitKey(33)  # ~30 FPS
                    
                    print(".", end="", flush=True)
                
                print()  # New line
                
                if len(samples) >= 10:  # Need at least 10 good samples
                    # Average the samples
                    avg_gaze = np.mean(samples, axis=0)
                    self.gaze_vectors.append(avg_gaze)
                    self.screen_points.append([screen_x, screen_y])
                    print(f"  ✓ Recorded {len(samples)} samples")
                    print(f"    Average gaze: pitch={avg_gaze[0]:.3f}, yaw={avg_gaze[1]:.3f}")
                    self.current_point += 1
                    
                    # Brief pause before next point
                    time.sleep(0.5)
                else:
                    print(f"  ✗ Only got {len(samples)} samples - need at least 10")
                    print("    Make sure your face is visible. Try again.")
        
        cv2.destroyAllWindows()
        
        # Fit calibration model
        print("\n" + "="*60)
        print("Fitting calibration model...")
        calib = Calibration()
        gaze_array = np.array(self.gaze_vectors)
        screen_array = np.array(self.screen_points)
        calib.fit(gaze_array, screen_array)
        
        print("✓ Calibration complete!")
        print("="*60 + "\n")
        
        return calib
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect gaze calibration data')
    parser.add_argument('--model', default='gaze_two_eye.pth', 
                       help='Path to trained model')
    parser.add_argument('--output', default='calibration.npz', 
                       help='Output calibration file')
    parser.add_argument('--width', type=int, default=1920, 
                       help='Screen width in pixels')
    parser.add_argument('--height', type=int, default=1080, 
                       help='Screen height in pixels')
    args = parser.parse_args()
    
    try:
        collector = CalibrationCollector(args.model, args.width, args.height)
        calib = collector.collect_calibration()
        
        if calib is not None:
            calib.save(args.output)
            print(f"✓ Calibration saved to {args.output}")
            print(f"\nYou can now run: python calibrate_ui.py")
        else:
            print("Calibration was cancelled or failed")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()