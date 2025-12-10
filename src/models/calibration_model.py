import numpy as np
import pickle

class Calibration:
    """
    Maps gaze vector (pitch, yaw) to screen coordinates (x, y) using polynomial regression.
    """
    def __init__(self):
        self.weights_x = None
        self.weights_y = None

    def _get_features(self, gaze):
        """
        Construct feature vector from gaze (pitch, yaw).
        Uses 2nd degree polynomial features: [1, p, y, p*y, p^2, y^2]
        """
        p, y = gaze
        return np.array([1, p, y, p*y, p**2, y**2])

    def fit(self, gaze_samples, screen_points):
        """
        Fit the calibration model.
        
        Args:
            gaze_samples: List or array of (pitch, yaw) tuples
            screen_points: List or array of (x, y) screen coordinates
        """
        gaze_samples = np.array(gaze_samples)
        screen_points = np.array(screen_points)
        
        if len(gaze_samples) != len(screen_points):
            raise ValueError("Number of gaze samples must match number of screen points")
            
        X = np.array([self._get_features(g) for g in gaze_samples])
        
        # Solve linear least squares for X and Y coordinates separately
        # X * w = target
        self.weights_x, _, _, _ = np.linalg.lstsq(X, screen_points[:, 0], rcond=None)
        self.weights_y, _, _, _ = np.linalg.lstsq(X, screen_points[:, 1], rcond=None)
        
        print("Calibration fitted successfully.")

    def predict(self, gaze):
        """
        Predict screen coordinates for a given gaze vector.
        """
        if self.weights_x is None or self.weights_y is None:
            raise RuntimeError("Calibration model is not fitted")
            
        feat = self._get_features(gaze)
        x = np.dot(feat, self.weights_x)
        y = np.dot(feat, self.weights_y)
        return int(x), int(y)

    def save(self, filename):
        """Save calibration weights to file."""
        with open(filename, 'wb') as f:
            pickle.dump({'x': self.weights_x, 'y': self.weights_y}, f)
        print(f"Calibration saved to {filename}")

    def load(self, filename):
        """Load calibration weights from file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.weights_x = data['x']
            self.weights_y = data['y']
        print(f"Calibration loaded from {filename}")
