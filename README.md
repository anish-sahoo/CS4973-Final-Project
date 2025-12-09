Dataset - https://www.kaggle.com/datasets/dhruv413/mpiigaze

```
./utils/download_data.sh
```

## Goals of the project:
- train model using mpiigaze dataset
- calibrate this model for the screen
- create an inference gui that displays the tracking

## some development standards
- all settings will be in config.py
- no argparsing or argv use, everything should be put in config
- everything must support switch between cuda (nvidia) and mps (mac) using the DEVICE variable in config.py
- try to modularize as much as possible
- try to use a data_collector that collects the training metrics and stores diagrams every certain interval (this will be important for the paper)

## Setup
1. Create a Python venv and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Download MPIIGaze dataset and prepare a CSV (two-eye):
   - Format: image_path,l_x1,l_y1,l_x2,l_y2,r_x1,r_y1,r_x2,r_y2,pitch,yaw,head_pitch,head_yaw
   - Each image_path points to a full frame in `Data/Original`. Bounding boxes define left & right eye crop.
   - Use `utils/create_two_eye_csv.py` to generate `data/mpiigaze_two_eye.csv`

3. Train: (TBD)
```
tensorboard --logdir runs
```

1. Calibrate:

After training, you have to calibrate the gaze tracking model to your specific screen and seating position. It maps the raw gaze vectors (Pitch/Yaw) predicted by the neural network to 2D pixel coordinates on your monitor.

### Prerequisites

1.  **Trained Model**: You must have a trained model checkpoint (e.g., `checkpoints/gaze_best_complete.pth`). Run the training pipeline first if you haven't.
2.  **Webcam**: A working webcam connected to your computer.
3.  **Screen**: The calibration script assumes you are running it on the screen you intend to track.

### How to Run

Run the interactive calibration UI from the project root:

```bash
python3 src/calibration.py
```

### The Calibration Process

1.  **Initialization**: The script will load your best model and open a full-screen window.
2.  **25-Point Grid**: You will be presented with a sequence of 25 targets (Red Circles) covering the screen (corners, edges, and center).
3.  **Capture**:
    *   Look steadily at the **Red Circle**.
    *   Press the **SPACEBAR**.
    *   The circle will turn **Green** for 1 second while it collects gaze samples.
    *   Once finished, the next target will appear.
4.  **Fitting**: After all 25 points are collected, the system calculates a mapping function (Polynomial Regression).
5.  **Testing**: The system immediately enters "Test Mode". A green crosshair/circle will appear on screen indicating where the model thinks you are looking.
6.  **Save**: The calibration parameters are automatically saved to `calibration.pkl` in the current directory.

### Controls

*   **SPACE**: Start capturing samples for the current target.
*   **ESC**: Abort calibration and exit.
*   **q**: Quit the test mode after calibration.

### Tips for Best Accuracy

*   **Lighting**: Ensure your face is evenly lit. Avoid strong backlighting (windows behind you).
*   **Stability**: Try to keep your head relatively still during calibration, though the model is robust to some head movement.
*   **Distance**: Sit at a comfortable, normal working distance (approx. 50-70cm) and try to maintain that distance.
*   **Eyes**: Open your eyes normally. Squinting or widening them unnaturally can affect predictions.


## Demo/Inference:
1) Ensure you have calibrated the model, otherwise results will be incomprehensible
2) Run `python3 src/demo.py` to use the inference ui

## Project Structure
```bash
.
├── calibration.pkl         # Saved calibration parameters (generated after running calibration.py)
├── config.py               # Global configuration settings (paths, hyperparameters, device)
├── main.py                 # Main entry point for training the model
├── paper/
│   ├── main.pdf
│   ├── main.tex
│   └── neurips_2020.sty
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── src/
│   ├── calibration.py      # Interactive calibration tool (collects points to map gaze to screen)
│   ├── datasets/           # PyTorch dataset implementations for MPIIGaze
│   ├── demo.py             # User-facing demo application (controls cursor with eyes)
│   ├── evaluate.py         # Script to evaluate model performance on test set
│   ├── infer_realtime.py   # Core inference engine (Webcam -> Face Mesh -> Gaze Vector)
│   ├── models/             # Neural network architecture definitions (GazeNet)
│   ├── train.py            # Training loop implementation
│   ├── utils.py            # Helper functions for image processing and geometry
│   └── visualizer.py       # Visualization utilities for training monitoring
├── utils/
│   ├── create_csv.py       # Helper to parse dataset structure
│   ├── download_data.sh    # Script to download MPIIGaze dataset
│   └── two_eye_csv_generator.py # Generates the main training CSV from raw data
└── visualization/
    ├── create_model_diagram.py # Generates the network architecture diagram
    ├── gaze_net_architecture.png # Visual representation of the model
    ├── generated_plot.png  # Sample output plots
    └── plot_generator.py   # Script to create training/evaluation plots
```
