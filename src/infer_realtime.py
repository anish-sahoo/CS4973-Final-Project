import cv2
import mediapipe as mp
import torch
import numpy as np
from models.gaze_model import GazeNet
from utils import draw_point_on_frame, crop_eye_from_frame, extract_eye_landmarks, estimate_head_pose
from calibration.calibration import Calibration

mp_face_mesh = mp.solutions.face_mesh

class RealtimeGazeEngine:
    def __init__(self, model_path, device='cpu', img_size=(36,60)):
        self.device = device
        self.img_size = img_size
        self.cap = cv2.VideoCapture(0)
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.model = GazeNet()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device).eval()
        self.calib = None

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Webcam read error")
        return frame

    def _crop_eye(self, frame, lm_coords, w_pad=0.15):
        # lm_coords: list of (x_norm, y_norm) landmarks bounding the eye; using min/max to crop
        h, w = frame.shape[:2]
        xs = [int(x * w) for x,y in lm_coords]
        ys = [int(y * h) for x,y in lm_coords]
        x1, x2 = max(min(xs), 0), min(max(xs), w-1)
        y1, y2 = max(min(ys), 0), min(max(ys), h-1)
        # pad
        pad_x = int((x2-x1)*w_pad) + 2
        pad_y = int((y2-y1)*w_pad) + 2
        x1 = max(x1-pad_x, 0)
        x2 = min(x2+pad_x, w-1)
        y1 = max(y1-pad_y, 0)
        y2 = min(y2+pad_y, h-1)
        crop = frame[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, self.img_size)
        crop = crop.astype(np.float32)/255.0
        crop = (crop - 0.5)/0.5
        crop = np.expand_dims(crop, axis=0) # channel dim
        return crop

    def predict_frame(self, frame):
        try:
            # returns gaze pitch,yaw (radians)
            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                return np.array([0.0, 0.0], dtype=np.float32)
            lm = results.multi_face_landmarks[0]
            # landmarks indices around eyes (MediaPipe iris/eyes):
            # left eye approx: landmarks 33..133 region; iris: 468..471; right eye similar
            # We'll pick sets roughly around the eyes (coarse)
            left_idxs = [33, 133, 160, 159, 158, 144, 145, 153, 154]  # approximate
            right_idxs = [263, 362, 387, 386, 385, 373, 374, 380, 381]
            left_lm, right_lm = extract_eye_landmarks(lm, left_idxs, right_idxs)
            left_crop = crop_eye_from_frame(frame, left_lm, img_size=self.img_size)
            right_crop = crop_eye_from_frame(frame, right_lm, img_size=self.img_size)
            # head pose estimation using 6-point model
            head_pitch, head_yaw = estimate_head_pose(lm, (h,w))
            head = np.array([head_pitch, head_yaw], dtype=np.float32)
            # to torch
            left_t = torch.from_numpy(left_crop).unsqueeze(0).to(self.device).float()
            right_t = torch.from_numpy(right_crop).unsqueeze(0).to(self.device).float()
            head_t = torch.from_numpy(head).unsqueeze(0).to(self.device).float()
            with torch.no_grad():
                pred = self.model(left_t, right_t, head_t).cpu().numpy().squeeze(0)
            return pred  # pitch,yaw
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.array([0.0, 0.0], dtype=np.float32)


    def set_calibration(self, calib: Calibration):
        self.calib = calib

    def predict_screen_point(self, gaze_pred):
        if self.calib is None:
            return None
        xy = self.calib.predict(gaze_pred.reshape(1,2))[0]
        return tuple(int(x) for x in xy)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--calib', default=None)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    engine = RealtimeGazeEngine(args.model, device='cpu')
    if args.calib:
        from calibration.calibration import Calibration
        c = Calibration()
        c.load(args.calib)
        engine.set_calibration(c)

    while True:
        frame = engine.read_frame()
        gaze = engine.predict_frame(frame)
        if engine.calib is not None:
            pt = engine.predict_screen_point(gaze)
            if pt is not None:
                draw_point_on_frame(frame, pt[0], pt[1])
        cv2.imshow("gaze", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    engine.cap.release()
    cv2.destroyAllWindows()
