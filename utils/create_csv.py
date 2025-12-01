"""Wrapper script for creating a two-eye CSV (replaces old landmarks create_csv).
This will generate `data/mpiigaze_two_eye.csv` using `create_two_eye_csv.py` implementation.
"""
from utils.two_eye_csv_generator import generate_csv

if __name__ == '__main__':
    generate_csv()
