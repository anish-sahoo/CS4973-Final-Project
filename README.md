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

2. Download MPIIGaze dataset and prepare a CSV:
   - Format: left_img_path,right_img_path,pitch,yaw,head_pitch,head_yaw
   - left/right paths should point to the eye crop images (or whole face crops if you modify loader accordingly)

3. Train: (TBD)

4. Calibrate: (TBD)

5. Real-time demo: (TBD)