
import os
"""
Global configuration for device selection and common hyperparameters.
Single variable change to select device: set DEVICE = 'cuda' or 'mps' or 'cpu'
"""

# Device: 'cuda' (NVIDIA), 'mps' (Apple Metal / MPS for macOS), 'cpu'
# Set to 'auto' to choose the best available one.
DEVICE = 'auto'

# Training defaults
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 30
NUM_WORKERS = 4
PIN_MEMORY = False
LOG_INTERVAL = 10  # Log more frequently since batches are larger
PLOT_SAVE_INTERVAL = 1  # how often to save the png plots (in epochs)

# Optimizer: 'adam', 'adamw', 'sgd'
OPTIMIZER = 'adamw'  # AdamW generally works better with weight decay
WEIGHT_DECAY = 1e-4  # L2 regularization

# Gradient clipping to prevent exploding gradients
GRAD_CLIP_MAX_NORM = 1.0  # Set to None to disable

# Learning rate scheduler: 'cosine', 'step', 'plateau', 'none'
LR_SCHEDULER = 'cosine'  # Cosine annealing with warm restarts
LR_STEP_SIZE = 10  # For 'step' scheduler: decay every N epochs
LR_GAMMA = 0.1  # For 'step' scheduler: multiply LR by this factor
LR_MIN = 1e-6  # Minimum LR for cosine scheduler

# Loss function: 'mse' or 'angular'
LOSS_FUNCTION = 'angular'
TRAIN_MODE = 'complete'
USE_TENSORBOARD = True
USE_NORMALIZED = True  # if True, load normalized eye patches from .mat files

EVAL_SAMPLE_SIZE = 1000
EVAL_BATCH_SIZE = 64

DATA_DIR = 'data'
LOG_DIR = 'runs'
CHECKPOINT_DIR = 'checkpoints'
ANNOT_DIR = os.path.join(DATA_DIR, 'MPIIGaze', 'Annotation Subset')
IMG_ROOT = os.path.join(DATA_DIR, 'MPIIGaze', 'Data', 'Original')
OUTPUT = os.path.join(DATA_DIR, 'mpiigaze_landmarks.csv')
NORMALIZED_ROOT = os.path.join(DATA_DIR, 'MPIIGaze', 'Data', 'Normalized')
CSV_PATH = os.path.join(DATA_DIR, 'mpiigaze_two_eye.csv')

# Mediapipe face mesh indices for head pose estimation (6pt model)
# [nose_tip, left_eye_outer, right_eye_outer, left_mouth, right_mouth, chin]
HEAD_POSE_LANDMARKS = [1, 33, 263, 61, 291, 199]

# Debug mode: if True, uses a smaller dataset to step through the pipeline
DEBUG = False