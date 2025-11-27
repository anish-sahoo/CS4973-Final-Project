import os
import csv
import math
import cv2
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

DATA_ROOT = os.path.join(config.DATA_DIR, 'MPIIGaze', 'Data', 'Original')
OUTPUT = os.path.join(config.DATA_DIR, 'mpiigaze_two_eye.csv')

def gaze_vector_to_angles(gx, gy, gz):
    # clamp values to [-1,1] before asin
    val = max(-1.0, min(1.0, -gy))
    theta = math.asin(val)
    phi = math.atan2(-gx, -gz)
    return float(theta), float(phi)

def rotation_vector_to_head_angles(rvec):
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float32))
    Zv = R[:, 2]
    head_pitch = math.asin(Zv[1])
    head_yaw = math.atan2(Zv[0], Zv[2])
    return float(head_pitch), float(head_yaw)

def generate_csv(output_path=OUTPUT):
    rows = []
    for person in sorted(os.listdir(DATA_ROOT)):
        pdir = os.path.join(DATA_ROOT, person)
        if not os.path.isdir(pdir):
            continue
        for day in sorted(os.listdir(pdir)):
            day_dir = os.path.join(pdir, day)
            ann_file = os.path.join(day_dir, 'annotation.txt')
            if not os.path.exists(ann_file):
                continue
            # list images and sort to map lines to filenames
            images = sorted([f for f in os.listdir(day_dir) if f.lower().endswith('.jpg')])
            with open(ann_file, 'r') as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) < 41:
                        continue
                    if idx >= len(images):
                        continue
                    img_name = images[idx]
                    img_path = os.path.join(day_dir, img_name)
                    try:
                        # parse first 24 numbers as 12 (x,y) pairs
                        pts = [(float(parts[i]), float(parts[i+1])) for i in range(0, 24, 2)]
                        # sort by x coordinate and split into two clusters (left/right)
                        pts_sorted = sorted(pts, key=lambda p: p[0])
                        mid = len(pts_sorted)//2
                        left_pts = pts_sorted[:mid]
                        right_pts = pts_sorted[mid:]
                        lx1 = min([p[0] for p in left_pts])
                        lx2 = max([p[0] for p in left_pts])
                        ly1 = min([p[1] for p in left_pts])
                        ly2 = max([p[1] for p in left_pts])
                        rx1 = min([p[0] for p in right_pts])
                        rx2 = max([p[0] for p in right_pts])
                        ry1 = min([p[1] for p in right_pts])
                        ry2 = max([p[1] for p in right_pts])
                        # add padding (10%)
                        wleft = max(1, lx2-lx1)
                        hleft = max(1, ly2-ly1)
                        padx = max(2, int(0.12 * wleft))
                        pady = max(2, int(0.12 * hleft))
                        lx1 = max(0, int(lx1 - padx))
                        ly1 = max(0, int(ly1 - pady))
                        lx2 = int(lx2 + padx)
                        ly2 = int(ly2 + pady)
                        wright = max(1, rx2-rx1)
                        hright = max(1, ry2-ry1)
                        padx2 = max(2, int(0.12 * wright))
                        pady2 = max(2, int(0.12 * hright))
                        rx1 = max(0, int(rx1 - padx2))
                        ry1 = max(0, int(ry1 - pady2))
                        rx2 = int(rx2 + padx2)
                        ry2 = int(ry2 + pady2)
                        # gaze 3D vector (parts[26:29])
                        gx, gy, gz = float(parts[26]), float(parts[27]), float(parts[28])
                        pitch, yaw = gaze_vector_to_angles(gx, gy, gz)
                        # head rotation vector parts[29:32]
                        rvec = [float(parts[29]), float(parts[30]), float(parts[31])]
                        head_pitch, head_yaw = rotation_vector_to_head_angles(rvec)
                        rows.append([
                            img_path,
                            lx1, ly1, lx2, ly2,
                            rx1, ry1, rx2, ry2,
                            pitch, yaw, head_pitch, head_yaw
                        ])
                    except (ValueError, IndexError, cv2.error):
                        continue

    # Write CSV
    header = ['image_path','l_x1','l_y1','l_x2','l_y2', 'r_x1','r_y1','r_x2','r_y2', 'pitch','yaw','head_pitch','head_yaw']
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print('Wrote', len(rows), 'rows to', output_path)

if __name__ == '__main__':
    generate_csv()

