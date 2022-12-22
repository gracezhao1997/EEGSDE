import argparse
import qm9.visualizer as vis
import os
from configs.datasets_config import qm9_with_h
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    dataset_info = qm9_with_h
    vis.visualize_chain(args.data_path, args.save_path, dataset_info, spheres_3d=True)


