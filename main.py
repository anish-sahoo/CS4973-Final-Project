import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.train import train
from src.evaluate import evaluate

def main():
    print("==================================================")
    print("       Gaze Tracking Project - Main Pipeline      ")
    print("==================================================")
    
    # Step 1: Train
    print("\n>>> Step 1: Training Model")
    train()
    
    # Step 2: Evaluate
    print("\n>>> Step 2: Evaluating Model")
    evaluate()
    
    print("\n==================================================")
    print("               Pipeline Complete                  ")
    print("==================================================")

if __name__ == "__main__":
    main()
