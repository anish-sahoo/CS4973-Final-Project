#!/bin/zsh
# Start TensorBoard server pointing at the log directory

LOG_DIR=${1:-runs}

tensorboard --logdir "$LOG_DIR" --bind_all
