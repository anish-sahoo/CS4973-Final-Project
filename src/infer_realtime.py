import os
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import mediapipe as mp
import torch
import numpy as np
from pathlib import Path

import config
from models.gaze_model import GazeNet
from utils import draw_point_on_frame, crop_eye_from_frame, extract_eye_landmarks, estimate_head_pose, draw_gaze

mp_face_mesh = mp.solutions.face_mesh

class RealtimeGazeEngine:
    def __init__(self, model_path, device=None, img_size=(36,60)):
        self.device = device if device else torch.device(config.DEVICE if config.DEVICE != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.img_size = img_size
        self.cap = cv2.VideoCapture(0)
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1, 
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print(f"Loading model from {model_path}...")
        self.model = GazeNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        print(f"Model loaded on {self.device}")

    def get_gaze(self, frame):
        """
        Process a single frame and return gaze vector (pitch, yaw).
        Returns None if face not detected.
        """
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            return None, frame

        lm = results.multi_face_landmarks[0]
        
        # Extract eye landmarks
        left_idxs = [33, 133, 160, 159, 158, 144, 145, 153, 154]
        right_idxs = [263, 362, 387, 386, 385, 373, 374, 380, 381]
        
        left_lm, right_lm = extract_eye_landmarks(lm, left_idxs, right_idxs)
        
        # Crop eyes
        left_crop = crop_eye_from_frame(frame, left_lm, img_size=self.img_size)
        right_crop = crop_eye_from_frame(frame, right_lm, img_size=self.img_size)
        
        # Estimate head pose
        head_pitch, head_yaw = estimate_head_pose(lm, (h,w))
        head = np.array([head_pitch, head_yaw], dtype=np.float32)
        
        # Prepare tensors
        left_t = torch.from_numpy(left_crop).unsqueeze(0).to(self.device).float()
        right_t = torch.from_numpy(right_crop).unsqueeze(0).to(self.device).float()
        head_t = torch.from_numpy(head).unsqueeze(0).to(self.device).float()
        
        # Inference
        with torch.no_grad():
            pred = self.model(left_t, right_t, head_t).cpu().numpy().squeeze(0)
        
        gaze_pitch, gaze_yaw = pred[0], pred[1]
        
        # Visualization (optional, can be moved out)
        for x, y in left_lm:
            draw_point_on_frame(frame, x*w, y*h, color=(0, 255, 0))
        for x, y in right_lm:
            draw_point_on_frame(frame, x*w, y*h, color=(0, 255, 0))
        draw_gaze(frame, gaze_pitch, gaze_yaw, color=(0, 0, 255))
        
        return (gaze_pitch, gaze_yaw), frame

    def run(self):
        print("Starting inference loop. Press 'q' to exit.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read from webcam")
                break

            gaze, frame = self.get_gaze(frame)
            
            if gaze:
                gaze_pitch, gaze_yaw = gaze
                cv2.putText(frame, f"Gaze: P={gaze_pitch:.2f}, Y={gaze_yaw:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Realtime Gaze Tracking', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

def find_best_model():
    """Find the best model in the checkpoints directory."""
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        return None

    models = [
        checkpoint_dir / 'gaze_best_train_complete.pth',
        checkpoint_dir / 'gaze_final_complete.pth',
        checkpoint_dir / 'gaze_best_short.pth',
        checkpoint_dir / 'gaze_final_short.pth'
    ]
    
    for model_path in models:
        if model_path.exists():
            return str(model_path)
            
    # Fallback to any .pth file
    pth_files = list(checkpoint_dir.glob('*.pth'))
    if pth_files:
        return str(pth_files[0])
        
    return None

if __name__ == '__main__':
    model_path = find_best_model()
    
    if not model_path:
        print(f"No model found in {config.CHECKPOINT_DIR}. Please run training first.")
        sys.exit(1)
        
    engine = RealtimeGazeEngine(model_path)
    engine.run()
