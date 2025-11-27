import numpy as np

class Calibration:
    def __init__(self):
        self.Ax = None
        self.Ay = None

    def fit(self, gaze_vectors, screen_points):
        # gaze_vectors: Nx2 (pitch, yaw)
        # screen_points: Nx2 (x, y)
        G = np.hstack([gaze_vectors, np.ones((len(gaze_vectors),1))])
        Sx, Sy = screen_points[:,0], screen_points[:,1]
        self.Ax, _, _, _ = np.linalg.lstsq(G, Sx, rcond=None)
        self.Ay, _, _, _ = np.linalg.lstsq(G, Sy, rcond=None)

    def predict(self, gaze_vector):
        gv = np.append(gaze_vector, 1)
        x = gv @ self.Ax
        y = gv @ self.Ay
        return np.array([x, y])

    def save(self, path):
        # save coefficient vectors
        np.savez(path, Ax=self.Ax, Ay=self.Ay)

    def load(self, path):
        d = np.load(path)
        self.Ax = d['Ax']
        self.Ay = d['Ay']
