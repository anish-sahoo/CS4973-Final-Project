#!/bin/bash

if [ -d "data/MPIIGaze" ]; then
  echo "MPIIGaze data already exists. Exiting."
  exit 0
fi

mkdir -p data

curl -L -o data/mpiigaze.zip \
  https://www.kaggle.com/api/v1/datasets/download/dhruv413/mpiigaze

cd data

unzip mpiigaze.zip

rm mpiigaze.zip

cd ..

python3 utils/create_csv.py
