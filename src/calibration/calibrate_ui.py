import cv2
import torch
import pyautogui
import mediapipe as mp
from models.gaze_model import GazeNet
from calibration.calibration import Calibration
from utils import crop_eye_from_frame, extract_eye_landmarks, estimate_head_pose

mp_face_mesh = mp.solutions.face_mesh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GazeNet().to(device)
model.load_state_dict(torch.load("gaze_two_eye.pth", map_location=device))
model.eval()

calib = Calibration()
# Load precomputed calibration coefficients here (or collect live)

cap = cv2.VideoCapture(0)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        continue
    lm = results.multi_face_landmarks[0]
    left_idxs = [33, 133, 160, 159, 158, 144, 145, 153, 154]
    right_idxs = [263, 362, 387, 386, 385, 373, 374, 380, 381]
    left_lm, right_lm = extract_eye_landmarks(lm, left_idxs, right_idxs)
    left_crop = crop_eye_from_frame(frame, left_lm)
    right_crop = crop_eye_from_frame(frame, right_lm)
    left_t = torch.from_numpy(left_crop).unsqueeze(0).to(device).float()
    right_t = torch.from_numpy(right_crop).unsqueeze(0).to(device).float()
    # estimate head pose using our utility
    head_pitch, head_yaw = estimate_head_pose(lm, frame.shape[:2])
    head = torch.tensor([head_pitch, head_yaw], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        gaze_pred = model(left_t, right_t, head).detach().cpu().numpy()[0]
    
    screen_point = calib.predict(gaze_pred)
    pyautogui.moveTo(*screen_point)  # or draw on frame
    cv2.imshow("Eye Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
