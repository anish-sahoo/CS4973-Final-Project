import cv2
import numpy as np
import sys
import os
import pyautogui
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.infer_realtime import RealtimeGazeEngine, find_best_model
from calibration import Calibration

def get_displays():
    """Get list of connected displays (works on macOS with xrandr or system_profiler)."""
    try:
        # Try macOS first
        result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                              capture_output=True, text=True, timeout=5)
        if 'Resolution' in result.stdout:
            return True  # Multiple displays detected
    except:
        pass
    
    try:
        # Try Linux/X11
        result = subprocess.run(['xrandr', '--query'], 
                              capture_output=True, text=True, timeout=5)
        connected = result.stdout.count(' connected')
        return connected > 1
    except:
        pass
    
    return False

def main():
    print("==================================================")
    print("           Gaze Tracker Demo Interface            ")
    print("==================================================")

    # 1. Load Model
    model_path = find_best_model()
    if not model_path:
        print("Error: No model found. Please train the model first using 'python main.py'")
        return
    
    print(f"Loading model: {os.path.basename(model_path)}")
    engine = RealtimeGazeEngine(model_path)

    # 2. Load Calibration
    calib = Calibration()
    calib_path = "calibration.pkl"
    
    if os.path.exists(calib_path):
        print(f"Loading calibration from {calib_path}")
        calib.load(calib_path)
    else:
        print("Warning: 'calibration.pkl' not found in current directory.")
        print("Please run 'python src/calibration.py' first.")
        print("Running in uncalibrated mode.")
        calib = None
        
    try:
        screen_w, screen_h = pyautogui.size()
    except:
        screen_w, screen_h = 1920, 1080

    window_name = "Gaze Tracker Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set to fullscreen but allow moving to other displays with keyboard shortcut
    fullscreen_mode = True
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\nControls:")
    print("  q: Quit")
    print("  c: Recalibrate (launches calibration tool)")
    print("  f: Toggle fullscreen mode")

    # Smoothing buffer
    history = []
    SMOOTHING_WINDOW = 5

    while True:
        # Create a black canvas representing the screen
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        
        # Get Gaze from Webcam
        ret, frame = engine.cap.read()
        if not ret:
            print("Failed to read from webcam")
            break
            
        gaze, processed_frame = engine.get_gaze(frame)
        
        if gaze and calib:
            # Predict Screen Coordinates
            sx, sy = calib.predict(gaze)
            
            # Apply Smoothing
            history.append((sx, sy))
            if len(history) > SMOOTHING_WINDOW:
                history.pop(0)
            
            avg_x = int(np.mean([p[0] for p in history]))
            avg_y = int(np.mean([p[1] for p in history]))
            
            # Clamp to screen bounds
            avg_x = max(0, min(avg_x, screen_w))
            avg_y = max(0, min(avg_y, screen_h))
            
            # Draw Gaze Point (Green Circle)
            cv2.circle(canvas, (avg_x, avg_y), 30, (0, 255, 0), 2)
            cv2.circle(canvas, (avg_x, avg_y), 5, (0, 255, 0), -1)
            
            # Draw Crosshair
            cv2.line(canvas, (avg_x-20, avg_y), (avg_x+20, avg_y), (0, 255, 0), 2)
            cv2.line(canvas, (avg_x, avg_y-20), (avg_x, avg_y+20), (0, 255, 0), 2)
            
            # Display Coordinates (top right corner)
            coord_text = f"Screen: ({avg_x}, {avg_y})"
            text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.putText(canvas, coord_text, (screen_w - text_size[0] - 50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif gaze:
            # Uncalibrated mode
            cv2.putText(canvas, "Uncalibrated Mode", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(canvas, f"Gaze: P={gaze[0]:.2f}, Y={gaze[1]:.2f}", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Overlay Webcam Feed (Bottom Right)
        h, w = processed_frame.shape[:2]
        scale = 0.3
        small_h, small_w = int(h * scale), int(w * scale)
        small_frame = cv2.resize(processed_frame, (small_w, small_h))
        
        # Calculate position
        y_offset = screen_h - small_h - 20
        x_offset = screen_w - small_w - 20
        
        # Ensure it fits
        if y_offset > 0 and x_offset > 0:
            canvas[y_offset : y_offset + small_h, 
                   x_offset : x_offset + small_w] = small_frame
            
            # Draw border around webcam
            cv2.rectangle(canvas, (x_offset, y_offset), 
                         (x_offset + small_w, y_offset + small_h), (255, 255, 255), 2)

        # UI Text
        cv2.putText(canvas, "Gaze Tracker Demo", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(canvas, "Press 'q' to quit", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(canvas, "Press 'c' to calibrate", (50, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(canvas, "Press 'f' to toggle fullscreen", (50, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.imshow(window_name, canvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Close current window and run calibration
            engine.cap.release()
            cv2.destroyAllWindows()
            print("Launching calibration...")
            os.system("python3 src/calibration.py")
            # Restart this script to reload calibration
            os.execv(sys.executable, ['python3'] + sys.argv)
        elif key == ord('f'):
            # Toggle fullscreen mode
            fullscreen_mode = not fullscreen_mode
            if fullscreen_mode:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                # Move window to center for easier repositioning
                cv2.moveWindow(window_name, 100, 100)

    engine.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
