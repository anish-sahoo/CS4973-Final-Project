import cv2
import torch
import pyautogui
from models.gaze_model import GazeCNN
from calibration.calibration import Calibration
from utils import extract_eye

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GazeCNN().to(device)
model.load_state_dict(torch.load("gaze_cnn.pth", map_location=device))
model.eval()

calib = Calibration()
# Load precomputed calibration coefficients here (or collect live)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    eye_img = extract_eye(frame)  # implement face + eye detection
    eye_tensor = torch.tensor(eye_img).unsqueeze(0).unsqueeze(0).float().to(device)/255.0
    gaze_pred = model(eye_tensor).detach().cpu().numpy()[0]
    
    screen_point = calib.predict(gaze_pred)
    pyautogui.moveTo(*screen_point)  # or draw on frame
    cv2.imshow("Eye Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
