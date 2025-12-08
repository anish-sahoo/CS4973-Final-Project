import numpy as np
import pickle
import cv2
import time
import pyautogui
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './..')))

class Calibration:
    """
    Maps gaze vector (pitch, yaw) to screen coordinates (x, y) using polynomial regression.
    """
    def __init__(self):
        self.weights_x = None
        self.weights_y = None

    def _get_features(self, gaze):
        """
        Construct feature vector from gaze (pitch, yaw).
        Uses 2nd degree polynomial features: [1, p, y, p*y, p^2, y^2]
        """
        p, y = gaze
        return np.array([1, p, y, p*y, p**2, y**2])

    def fit(self, gaze_samples, screen_points):
        """
        Fit the calibration model.
        
        Args:
            gaze_samples: List or array of (pitch, yaw) tuples
            screen_points: List or array of (x, y) screen coordinates
        """
        gaze_samples = np.array(gaze_samples)
        screen_points = np.array(screen_points)
        
        if len(gaze_samples) != len(screen_points):
            raise ValueError("Number of gaze samples must match number of screen points")
            
        X = np.array([self._get_features(g) for g in gaze_samples])
        
        # Solve linear least squares for X and Y coordinates separately
        # X * w = target
        self.weights_x, _, _, _ = np.linalg.lstsq(X, screen_points[:, 0], rcond=None)
        self.weights_y, _, _, _ = np.linalg.lstsq(X, screen_points[:, 1], rcond=None)
        
        print("Calibration fitted successfully.")

    def predict(self, gaze):
        """
        Predict screen coordinates for a given gaze vector.
        """
        if self.weights_x is None or self.weights_y is None:
            raise RuntimeError("Calibration model is not fitted")
            
        feat = self._get_features(gaze)
        x = np.dot(feat, self.weights_x)
        y = np.dot(feat, self.weights_y)
        return int(x), int(y)

    def save(self, filename):
        """Save calibration weights to file."""
        with open(filename, 'wb') as f:
            pickle.dump({'x': self.weights_x, 'y': self.weights_y}, f)
        print(f"Calibration saved to {filename}")

    def load(self, filename):
        """Load calibration weights from file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.weights_x = data['x']
            self.weights_y = data['y']
        print(f"Calibration loaded from {filename}")

# Local imports
from src.infer_realtime import RealtimeGazeEngine, find_best_model

def main():
    # Get screen size
    try:
        screen_w, screen_h = pyautogui.size()
    except:
        print("Could not get screen size, defaulting to 1920x1080")
        screen_w, screen_h = 1920, 1080
    
    # Initialize Gaze Engine
    model_path = find_best_model()
    if not model_path:
        print("No model found! Please train the model first.")
        return
    
    print("Initializing Gaze Engine...")
    engine = RealtimeGazeEngine(model_path)
    
    # Initialize Calibration
    calib = Calibration()
    
    # Calibration grid (denser for better fit)
    # 5x5 grid = 25 points; adjust MARGIN to keep points away from screen edges
    GRID_POINTS = 5
    MARGIN = 120
    xs = np.linspace(MARGIN, screen_w - MARGIN, GRID_POINTS, dtype=int)
    ys = np.linspace(MARGIN, screen_h - MARGIN, GRID_POINTS, dtype=int)
    points = [(x, y) for y in ys for x in xs]
    
    # Setup Window
    window_name = "Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Data collection
    gaze_samples = []
    screen_targets = []
    
    print("Starting Calibration. Look at the red circles and press SPACE.")
    
    for i, (tx, ty) in enumerate(points):
        while True:
            # Create black background
            bg = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            
            # Draw target
            cv2.circle(bg, (tx, ty), 20, (0, 0, 255), -1)
            cv2.circle(bg, (tx, ty), 5, (255, 255, 255), -1)
            
            # Instructions
            cv2.putText(bg, f"Point {i+1}/{len(points)}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(bg, "Look at the red circle and press SPACE", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            # Show webcam feed in corner (optional, for feedback)
            ret, frame = engine.cap.read()
            if ret:
                gaze, processed_frame = engine.get_gaze(frame)
                small_frame = cv2.resize(processed_frame, (320, 240))
                # Place in bottom right
                bg[screen_h-240:screen_h, screen_w-320:screen_w] = small_frame
            
            cv2.imshow(window_name, bg)
            key = cv2.waitKey(1)
            
            if key == 32: # Space
                # Collect samples
                print(f"Collecting samples for point {i+1}...")
                samples = []
                start_time = time.time()
                
                # Visual feedback: Turn circle green
                cv2.circle(bg, (tx, ty), 20, (0, 255, 0), -1)
                cv2.imshow(window_name, bg)
                cv2.waitKey(1)
                
                # Collect for longer to reduce noise (2 seconds)
                while time.time() - start_time < 2.0:
                    ret, frame = engine.cap.read()
                    if ret:
                        gaze, _ = engine.get_gaze(frame)
                        if gaze:
                            samples.append(gaze)
                    cv2.waitKey(10)
                
                if len(samples) > 0:
                    avg_gaze = np.mean(samples, axis=0)
                    gaze_samples.append(avg_gaze)
                    screen_targets.append((tx, ty))
                    print(f"Captured {len(samples)} samples.")
                    break
                else:
                    print("No face detected! Try again.")
            
            elif key == 27: # ESC
                engine.cap.release()
                cv2.destroyAllWindows()
                return

    # Fit calibration
    print("Fitting calibration model...")
    calib.fit(gaze_samples, screen_targets)
    calib.save("calibration.pkl")
    print("Calibration saved!")
    
    # Test Loop
    print("Starting Test Mode. Press 'q' to quit.")
    
    # Smoothing buffer
    history = []
    SMOOTHING_WINDOW = 5
    
    while True:
        bg = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        
        ret, frame = engine.cap.read()
        if ret:
            gaze, processed_frame = engine.get_gaze(frame)
            
            if gaze:
                # Predict screen coordinates
                sx, sy = calib.predict(gaze)
                
                # Smoothing
                history.append((sx, sy))
                if len(history) > SMOOTHING_WINDOW:
                    history.pop(0)
                
                avg_sx = int(np.mean([p[0] for p in history]))
                avg_sy = int(np.mean([p[1] for p in history]))
                
                # Draw predicted point
                cv2.circle(bg, (avg_sx, avg_sy), 30, (0, 255, 0), 2) # Green circle
                cv2.line(bg, (avg_sx-10, avg_sy), (avg_sx+10, avg_sy), (0, 255, 0), 2)
                cv2.line(bg, (avg_sx, avg_sy-10), (avg_sx, avg_sy+10), (0, 255, 0), 2)
            
            # Show webcam
            small_frame = cv2.resize(processed_frame, (320, 240))
            bg[screen_h-240:screen_h, screen_w-320:screen_w] = small_frame
            
        cv2.putText(bg, "Test Mode - Look around! (Press 'q' to quit)", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow(window_name, bg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    engine.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
