import os
import glob
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
import scipy.io as sio

class MPIIGazeDataset(Dataset):
    """Dataset that provides left and right eye crops, optional head pose, and gaze labels.

    CSV format expected:
    image_path,l_x1,l_y1,l_x2,l_y2,r_x1,r_y1,r_x2,r_y2,pitch,yaw,head_pitch,head_yaw
    """
    def __init__(self, csv_file, transform=None, img_size=(36,60)):
        df = pd.read_csv(csv_file)
        self.data = df.to_dict('records')
        self.img_size = img_size  # (height, width)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        img = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Could not read image: {row['image_path']}")

        # Crop coordinates
        lx1, ly1, lx2, ly2 = int(row['l_x1']), int(row['l_y1']), int(row['l_x2']), int(row['l_y2'])
        rx1, ry1, rx2, ry2 = int(row['r_x1']), int(row['r_y1']), int(row['r_x2']), int(row['r_y2'])

        # Crop eyes
        left = img[ly1:ly2, lx1:lx2]
        right = img[ry1:ry2, rx1:rx2]

        target_w, target_h = self.img_size[1], self.img_size[0]
        left = cv2.resize(left, (target_w, target_h))
        right = cv2.resize(right, (target_w, target_h))

        left = left.astype(np.float32) / 255.0
        right = right.astype(np.float32) / 255.0

        # (x - mean) / std with mean=0.5, std=0.5 => (x - 0.5) / 0.5
        left = (left - 0.5) / 0.5
        right = (right - 0.5) / 0.5

        # (H, W) -> (1, H, W)
        left = torch.from_numpy(left).unsqueeze(0)
        right = torch.from_numpy(right).unsqueeze(0)

        gaze = torch.tensor([row['pitch'], row['yaw']], dtype=torch.float32)
        head = torch.tensor([row['head_pitch'], row['head_yaw']], dtype=torch.float32)
        return left, right, head, gaze


class MPIIGazeNormalizedDataset(Dataset):
    """Dataset that loads normalized eye patches directly from MPIIGaze normalized MAT files.
    
    Preloads all data for fast access during training.
    """

    def __init__(self, normalized_root, use_right=True):
        print("Loading normalized dataset into memory...")
        self.use_right = use_right
        left_images_list = []
        right_images_list = []
        head_list = []
        gaze_list = []

        mat_files = sorted(glob.glob(os.path.join(normalized_root, 'p*', '*.mat')))
        if not mat_files:
            raise FileNotFoundError(f"No MAT files found in {normalized_root}. Ensure normalized data is present.")

        from tqdm import tqdm
        for mf in tqdm(mat_files, desc="Loading MAT files"):
            try:
                mat = sio.loadmat(mf, squeeze_me=True, struct_as_record=False)
            except Exception as e:
                print(f"Warning: could not read {mf}: {e}")
                continue

            data = mat.get('data', None)
            if data is None:
                continue

            left_struct = getattr(data, 'left', None)
            right_struct = getattr(data, 'right', None)
            if left_struct is None or right_struct is None:
                continue

            try:
                left_imgs = np.array(left_struct.image, dtype=np.float32)
                right_imgs = np.array(right_struct.image, dtype=np.float32)
                gaze_vecs = np.array(right_struct.gaze if use_right else left_struct.gaze, dtype=np.float32)
                pose_vecs = np.array(right_struct.pose if use_right else left_struct.pose, dtype=np.float32)
            except Exception as e:
                continue

            if left_imgs.ndim == 2:
                left_imgs = left_imgs[None, ...]
                right_imgs = right_imgs[None, ...]
                gaze_vecs = gaze_vecs[None, ...]
                pose_vecs = pose_vecs[None, ...]
            
            if left_imgs.ndim != 3 or right_imgs.ndim != 3:
                continue

            # Normalize images to [-1, 1]
            left_imgs = (left_imgs / 255.0 - 0.5) / 0.5
            right_imgs = (right_imgs / 255.0 - 0.5) / 0.5

            gaze_angles = self._vec_to_pitch_yaw(gaze_vecs)
            head_angles = self._vec_to_pitch_yaw(pose_vecs)

            left_images_list.append(left_imgs)
            right_images_list.append(right_imgs)
            gaze_list.append(gaze_angles)
            head_list.append(head_angles)

        if not gaze_list:
            raise RuntimeError("No valid samples loaded from normalized data.")

        # Concatenate all data into contiguous arrays
        print("Concatenating arrays...")
        self.left_images = np.concatenate(left_images_list, axis=0)
        self.right_images = np.concatenate(right_images_list, axis=0)
        self.gaze = np.concatenate(gaze_list, axis=0)
        self.head = np.concatenate(head_list, axis=0)
        
        print(f"Loaded {len(self.gaze)} samples")

    @staticmethod
    def _vec_to_pitch_yaw(vec):
        """Convert 3D gaze/head vectors to pitch and yaw angles."""
        pitch = np.arcsin(vec[:, 1])
        yaw = np.arctan2(vec[:, 0], vec[:, 2])
        return np.stack([pitch, yaw], axis=1)

    def __len__(self):
        return len(self.gaze)

    def __getitem__(self, idx):
        # Convert to tensors - numpy arrays are shared across workers via copy-on-write
        left = torch.from_numpy(self.left_images[idx].copy()).unsqueeze(0)
        right = torch.from_numpy(self.right_images[idx].copy()).unsqueeze(0)
        gaze = torch.from_numpy(self.gaze[idx].copy())
        head = torch.from_numpy(self.head[idx].copy())
        return left, right, head, gaze
