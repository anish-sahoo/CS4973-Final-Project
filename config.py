
import os
"""
Global configuration for device selection and common hyperparameters.
Single variable change to select device: set DEVICE = 'cuda' or 'mps' or 'cpu'
"""

# Device: 'cuda' (NVIDIA), 'mps' (Apple Metal / MPS for macOS), 'cpu'
# Set to 'auto' to choose the best available one.
DEVICE = 'auto'

# Training defaults
BATCH_SIZE = 512  # Increased for 4090
LR = 1e-3
EPOCHS = 40
NUM_WORKERS = 16  # Increased workers
PIN_MEMORY = True  # Speeds up data transfer to CUDA
LOG_INTERVAL = 10  # Log more frequently since batches are larger
PLOT_SAVE_INTERVAL = 1  # how often to save the png plots (in epochs)

# Training Mode: 'short' (quick test on small subset) or 'complete' (full dataset)
TRAIN_MODE = 'complete'
# Tensorboard: True to enable logging, False to disable
USE_TENSORBOARD = True

# Evaluation defaults
EVAL_SAMPLE_SIZE = 1000
EVAL_BATCH_SIZE = 64

# Paths
DATA_DIR = 'data'
LOG_DIR = 'runs'
CHECKPOINT_DIR = 'checkpoints'

# MPIIGaze specific paths
ANNOT_DIR = os.path.join(DATA_DIR, 'MPIIGaze', 'Annotation Subset')
IMG_ROOT = os.path.join(DATA_DIR, 'MPIIGaze', 'Data', 'Original')
OUTPUT = os.path.join(DATA_DIR, 'mpiigaze_landmarks.csv')

# Default CSV path to dataset labels (set to None to use RandomEyeDataset fallback)
CSV_PATH = os.path.join(DATA_DIR, 'mpiigaze_two_eye.csv')

# Mediapipe face mesh indices for head pose estimation (6pt model)
# [nose_tip, left_eye_outer, right_eye_outer, left_mouth, right_mouth, chin]
HEAD_POSE_LANDMARKS = [1, 33, 263, 61, 291, 199]

# Debug mode: if True, uses a smaller dataset to step through the pipeline
DEBUG = False