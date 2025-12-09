import numpy as np
import cv2
import math

def rad2deg(x):
    return x * 180.0 / np.pi

def deg2rad(x):
    return x * np.pi / 180.0

def draw_point_on_frame(frame, x, y, color=(0,0,255)):
    cv2.circle(frame, (int(x), int(y)), 8, color, -1)

def crop_eye_from_frame(frame, lm_coords, img_size=(36,60), w_pad=0.15):
    # lm_coords: list of (x_norm, y_norm) in [0,1] relative to image
    h, w = frame.shape[:2]
    xs = [int(x * w) for x,y in lm_coords]
    ys = [int(y * h) for x,y in lm_coords]
    x1, x2 = max(min(xs), 0), min(max(xs), w-1)
    y1, y2 = max(min(ys), 0), min(max(ys), h-1)
    pad_x = int((x2-x1)*w_pad) + 2
    pad_y = int((y2-y1)*w_pad) + 2
    x1 = max(x1-pad_x, 0)
    x2 = min(x2+pad_x, w-1)
    y1 = max(y1-pad_y, 0)
    y2 = min(y2+pad_y, h-1)
    crop = frame[y1:y2, x1:x2]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (img_size[1], img_size[0]))
    crop = crop.astype(np.float32)/255.0
    crop = (crop - 0.5)/0.5
    crop = np.expand_dims(crop, axis=0)  # channel dim
    return crop

def extract_eye_landmarks(face_landmarks, left_idxs, right_idxs):
    def lm_list(idxs):
        return [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in idxs]
    return lm_list(left_idxs), lm_list(right_idxs)

def estimate_head_pose(face_landmarks, image_shape, camera_matrix=None, dist_coeffs=None):
    """Estimate head pitch and yaw using solvePnP with a 6-point face model.

    face_landmarks: mediapipe face landmarks object
    image_shape: (height, width)
    Returns: head_pitch, head_yaw (radians)
    """
    import config
    import numpy as np
    import cv2

    inds = getattr(config, 'HEAD_POSE_LANDMARKS', None)
    if inds is None or len(inds) < 6:
        return 0.0, 0.0
    h, w = image_shape
    # approximate 3D model points (in mm) — using OpenCV sample model
    model_points = np.array([
        (0.0, 0.0, 0.0),  # nose tip
        (-225.0, 170.0, -135.0),  # left eye outer
        (225.0, 170.0, -135.0),   # right eye outer
        (-150.0, -150.0, -125.0), # left mouth
        (150.0, -150.0, -125.0),  # right mouth
        (0.0, -330.0, -65.0)      # chin
    ], dtype=np.float32)

    image_points = []
    for i in inds:
        lm = face_landmarks.landmark[i]
        x = int(lm.x * w)
        y = int(lm.y * h)
        image_points.append((x,y))
    image_points = np.array(image_points, dtype=np.float32)

    # if not provided, construct a basic camera_matrix
    if camera_matrix is None:
        focal_length = w
        camera_matrix = np.array([[focal_length, 0, w/2.0], [0, focal_length, h/2.0], [0,0,1]], dtype=np.float32)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4,1))
    # solvePnP
    try:
        success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return 0.0, 0.0
        R, _ = cv2.Rodrigues(rvec)
        Zv = R[:, 2]
        head_pitch = math.asin(Zv[1])
        head_yaw = math.atan2(Zv[0], Zv[2])
        return float(head_pitch), float(head_yaw)
    except Exception:
        return 0.0, 0.0

def draw_gaze(frame, pitch, yaw, thickness=2, color=(0, 255, 0), length=200):
    """Draw gaze vector on the frame."""
    h, w = frame.shape[:2]
    # Start point (center of image for now, or use face center)
    cx, cy = w // 2, h // 2
    
    # Calculate end point
    # pitch is rotation around x-axis (up/down)
    # yaw is rotation around y-axis (left/right)
    dx = -length * np.sin(yaw)
    dy = -length * np.sin(pitch)
    
    cv2.arrowedLine(frame, (cx, cy), (int(cx + dx), int(cy + dy)), color, thickness, tipLength=0.3)

def pitchyaw_to_vector(pitch, yaw):
    """Convert pitch and yaw to 3D vector."""
    vector = np.zeros((3, pitch.shape[0]))
    vector[0, :] = np.cos(pitch) * np.sin(yaw)
    vector[1, :] = np.sin(pitch)
    vector[2, :] = np.cos(pitch) * np.cos(yaw)
    return vector.T

def compute_angular_error(pred, target):
    """Compute angular error between predicted and target gaze vectors."""
    pred_v = pitchyaw_to_vector(pred[:, 0], pred[:, 1])
    target_v = pitchyaw_to_vector(target[:, 0], target[:, 1])
    
    # Dot product
    dot_product = np.sum(pred_v * target_v, axis=1)
    
    # Clamp to [-1, 1] to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Angular error in degrees
    angular_error = np.arccos(dot_product) * 180.0 / np.pi
    return angular_error


def angular_error_loss(pred, target):
    """Compute angular error loss for training (PyTorch version).
    
    Args:
        pred: (batch, 2) tensor of [pitch, yaw] in radians
        target: (batch, 2) tensor of [pitch, yaw] in radians
    
    Returns:
        Mean angular error in radians
    """
    import torch
    
    # Clamp input angles to reasonable range to prevent extreme values
    pred = torch.clamp(pred, -np.pi, np.pi)
    target = torch.clamp(target, -np.pi, np.pi)
    
    # Convert pitch/yaw to 3D unit vectors
    pred_x = -torch.cos(pred[:, 0]) * torch.sin(pred[:, 1])
    pred_y = -torch.sin(pred[:, 0])
    pred_z = -torch.cos(pred[:, 0]) * torch.cos(pred[:, 1])
    pred_vec = torch.stack([pred_x, pred_y, pred_z], dim=1)
    
    target_x = -torch.cos(target[:, 0]) * torch.sin(target[:, 1])
    target_y = -torch.sin(target[:, 0])
    target_z = -torch.cos(target[:, 0]) * torch.cos(target[:, 1])
    target_vec = torch.stack([target_x, target_y, target_z], dim=1)
    
    # Normalize vectors with better numerical stability
    pred_norm = torch.norm(pred_vec, dim=1, keepdim=True)
    target_norm = torch.norm(target_vec, dim=1, keepdim=True)
    pred_vec = pred_vec / torch.clamp(pred_norm, min=1e-6)
    target_vec = target_vec / torch.clamp(target_norm, min=1e-6)
    
    # Compute angular error with numerical stability
    dot_product = torch.sum(pred_vec * target_vec, dim=1)
    # Clamp more aggressively to avoid acos gradient issues near -1 and 1
    dot_product = torch.clamp(dot_product, -0.999999, 0.999999)
    angular_error = torch.acos(dot_product)
    
    # Check for any NaN in the loss before returning
    if torch.isnan(angular_error).any():
        print("Warning: NaN detected in angular error computation")
        angular_error = angular_error[~torch.isnan(angular_error)]
        if len(angular_error) == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    return angular_error.mean()

