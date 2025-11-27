# Eye/Gaze Tracking - Starter Project

A modular starter project for building an eye/gaze tracker using modern CV techniques.

Key features:
- Modular layout with separate data retriever, model, and training loop
- TensorBoard integration + a visualizer to save charts to disk at intervals
- Device selection via a single config variable (CUDA, MPS/Metal, CPU)

Quick start (macOS):

1. Create and activate a Python environment
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2. Edit `config.py` to set `DEVICE` to `cuda` (if NVIDIA GPU), `mps` (Mac Metal/Apple silicon), or `cpu` — or set the `DEVICE` environment variable to override at runtime.

3. Run the data retriever using a direct URL (if needed):
    ```bash
    python -m data.retriever --url https://example.com/path/to/dataset.zip --dest data/unityeyes
    ```

4. Run training using `config.py` configuration:
    ```bash
    # Edit `config.py` to set:
    #   - CSV_PATH = 'data/unityeyes/labels.csv' if you downloaded a dataset
    #   - EPOCHS = 1 and DEBUG = True for a short debug run
    # Then simply run:
    python3 train.py
    ```

Files:
- `config.py` - device & config variables
- `data/retriever.py` - data fetching & download helpers
- `models/eye_gaze_net.py` - sample lightweight CNN
- `utils/visualizer.py` - TensorBoardWriter + Matplotlib plotter that saves PNGs periodically
- `train.py` - training orchestration with logging

Notes:
- The dataset retriever expects a URL to the dataset archive (zip/tar/zip.gz) and will download and unpack it to the destination directory.
- The provided model is a placeholder to get started; replace with a model of your choice.
