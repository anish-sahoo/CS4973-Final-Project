"""
Global configuration for device selection and common hyperparameters.
Single variable change to select device: set DEVICE = 'cuda' or 'mps' or 'cpu'
"""

# Device: 'cuda' (NVIDIA), 'mps' (Apple Metal / MPS for macOS), 'cpu'
# Set to 'auto' to choose the best available one.
DEVICE = 'auto'

# Training defaults
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 10
LOG_INTERVAL = 10  # how often (in steps) to log
PLOT_SAVE_INTERVAL = 100  # how often to save the png plots (in steps)

# Paths
DATA_DIR = 'data'
LOG_DIR = 'runs'
CHECKPOINT_DIR = 'checkpoints'

# Default CSV path to dataset labels (set to None to use RandomEyeDataset fallback)
CSV_PATH = None

# Debug mode: if True, uses a smaller dataset to step through the pipeline
DEBUG = False

# Add other config variables as needed
