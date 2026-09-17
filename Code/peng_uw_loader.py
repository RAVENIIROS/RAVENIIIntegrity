
import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

# [ARTIFACT] prep | Data loading and column-mapping check (paper Sec. II-A)
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
Peng UW dataset loader.
Based on the mapping verified empirically in probe_peng_v2.py.

Column mapping (VERIFIED; the dataset README is off by one):
    Col 0     : GT elapsed time
    Cols 1:4  : GT joint pos (3 values, in RAD)
    Col 4     : CRTK Unix timestamp
    Cols 5:11 : CRTK joint pos (6 values, in RAD)
    Col 12    : Robot state Unix timestamp
    Cols 13:21: Robot state jpos (8 values, in DEG)
    Cols 34:37: End-effector position (encoder counts)
    Cols 114:122 : Motor torque
    Cols 130:138 : Motor position
    Cols 146:154 : Motor velocity
    Cols 162:170 : Joint velocity (real values, not zero)
"""

import glob
import numpy as np
import pandas as pd


# === Verified column mapping ===
COL = {
    'gt_time'      : 0,             # elapsed seconds
    'gt_jpos'      : (1, 4),        # rad, 3 joints
    'crtk_time'    : 4,             # Unix timestamp
    'crtk_jpos'    : (5, 11),       # rad, 6 joints
    'robot_time'   : 12,            # Unix timestamp
    'robot_jpos'   : (13, 21),      # 8 joints, joints 1-2 in DEG, joint 3 likely in mm
    'ee_pos'       : (34, 37),      # encoder counts, 3 axes
    'motor_torque' : (114, 122),    # 8 motors
    'motor_pos'    : (130, 138),    # 8 motors (mixed units)
    'motor_vel'    : (146, 154),    # 8 motors
    'joint_vel'    : (162, 170),    # 8 joints
}


def _all_needed_cols():
    """Return sorted list of all column indices we actually need."""
    idx = set()
    for k, v in COL.items():
        if isinstance(v, tuple):
            idx.update(range(v[0], v[1]))
        else:
            idx.add(v)
    return sorted(idx)


class PengUWLoader:
    """Memory-efficient loader for Peng UW CSV files."""

    NEEDED_COLS = _all_needed_cols()   # only load these

    @classmethod
    def load_csv(cls, path, subsample=5, verbose=False):
        """
        Load a single Peng CSV.

        Args:
            path      : CSV file path
            subsample : keep every N-th row (default 5 = ~132 Hz from 660 Hz)
                        Peng data is 660 Hz, most surgical dynamics are much slower.
            verbose   : print loading progress

        Returns:
            dict with numpy arrays for each field.
        """
        # Read only the needed columns to save memory
        df = pd.read_csv(path, header=None, usecols=cls.NEEDED_COLS)
        if subsample > 1:
            df = df.iloc[::subsample].reset_index(drop=True)

        def _slice(key):
            v = COL[key]
            if isinstance(v, tuple):
                return df[list(range(v[0], v[1]))].values.astype(np.float32)
            return df[[v]].values.astype(np.float32).ravel()

        data = {
            'gt_time'      : _slice('gt_time'),
            'gt_jpos_rad'  : _slice('gt_jpos'),
            'robot_time'   : _slice('robot_time'),
            'robot_jpos'   : _slice('robot_jpos'),      # deg (1,2), mixed (rest)
            'ee_pos'       : _slice('ee_pos'),          # encoder counts
            'motor_torque' : _slice('motor_torque'),
            'motor_pos'    : _slice('motor_pos'),
            'motor_vel'    : _slice('motor_vel'),
            'joint_vel'    : _slice('joint_vel'),
            'source'       : path,
            'n_samples'    : len(df),
        }

        # === Unit conversion ===
        # Ground-truth joint positions: joints 1 and 2 from radians to degrees
        gt = data['gt_jpos_rad'].copy()
        gt[:, 0] *= 180.0 / np.pi
        gt[:, 1] *= 180.0 / np.pi
        # Joint 3 (prismatic): radian-equivalent scale to degree-equivalent
        # Empirically matches robot_jpos column 15 (both in the 20-24 range)
        gt[:, 2] *= 180.0 / np.pi
        data['gt_jpos_deg'] = gt

        if verbose:
            fname = path.split('/')[-1]
            print(f"  {fname}  n={data['n_samples']:,}")

        return data

    @classmethod
    def load_directory(cls, directory, subsample=5, max_files=None,
                       max_samples_per_file=None, verbose=True):
        """
        Load all CSVs in a directory.

        Args:
            directory              : path
            subsample              : row subsample factor
            max_files              : optional limit on number of files
            max_samples_per_file   : optional row cap per file (after subsample)
            verbose                : print progress
        """
        csvs = sorted(glob.glob(f'{directory}/*.csv'))
        if not csvs:
            return []
        if max_files:
            csvs = csvs[:max_files]

        if verbose:
            print(f"  loading {len(csvs)} CSVs from {directory}")

        all_data = []
        for p in csvs:
            data = cls.load_csv(p, subsample=subsample, verbose=verbose)
            if max_samples_per_file and data['n_samples'] > max_samples_per_file:
                for k, v in data.items():
                    if isinstance(v, np.ndarray):
                        data[k] = v[:max_samples_per_file]
                data['n_samples'] = max_samples_per_file
            all_data.append(data)
        return all_data


if __name__ == '__main__':
    # Quick sanity check
    import sys
    ROOT = _DATA if len(sys.argv) < 2 else sys.argv[1]
    csvs = sorted(glob.glob(f'{ROOT}/record_1_different_directions/*.csv'))
    print(f"Test loading first CSV: {csvs[0]}")
    d = PengUWLoader.load_csv(csvs[0], subsample=5, verbose=True)
    print(f"\nLoaded {d['n_samples']:,} samples")
    print(f"\nField shapes and ranges:")
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            print(f"  {k:<15}  shape={str(v.shape):<15}  "
                  f"range=[{v.min():>10.3f}, {v.max():>10.3f}]")
