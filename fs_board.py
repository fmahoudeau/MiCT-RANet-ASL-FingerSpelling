# ==============================================================================
# Copyright 2025 Florent Mahoudeau. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import h5py
import json
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset


class FSBoard(Dataset):
    """
    ASL Fingerspelling Board Dataset.
    """
    def __init__(self, split, img_dir, annots, vocab_map, transform, 
                 img_size=224, map_size=14, **kwargs):
        """
        :param split: the dataset split must be 'test'.
        :param img_dir: the dataset RGB root directory.
        :param vocab_map: maps label's chars to integers.
        :param transform: image transformations.
        :param img_size: model image input size.
        :param map_size: prior map size, which is equal to the CNN features map
            size in output
        :param annots: the dataset's annotations JSON filename.
        """
        assert split == 'test', 'Unknown split: %s'.format(split)
        self.split = split
        self.img_dir = img_dir
        self.vocab_map = vocab_map
        self.transform = transform
        self.img_size = img_size
        self.map_size = map_size
        with open(annots, 'r') as f:
            self.annots = json.load(f)
            self.imdirs = list(self.annots.keys())
            self.imdirs = [key for key, value in self.annots.items() if value.get('label').replace(' ', '').isalpha()]

        print(f'{len(self.imdirs)} {split} samples')

    def __len__(self):
        return len(self.imdirs)

    def __getitem__(self, idx):
        """Loads a sample video at the scale specified by the `scale_x` instance attribute."""
        subdir = self.imdirs[idx]
        sample = self.annots[subdir]
        label = list(map(lambda x: self.vocab_map[x], sample['label']))
        
        rgb_h5_file = os.path.join(self.img_dir, f"{subdir}.h5")
        frames = self.load_frames_from_hdf5(rgb_h5_file, kind='rgb')

        imgs, grays = [], []
        for frame in frames:
            patch_rgb = cv.resize(
                frame, (self.img_size, self.img_size), 
                interpolation=cv.INTER_LINEAR
            )
            patch_gray = cv.cvtColor(patch_rgb, cv.COLOR_RGB2GRAY)
            imgs.append(patch_rgb)
            grays.append(patch_gray)

        imgs, gray = np.stack(imgs), np.stack(grays)[..., np.newaxis]
        sample = {'imgs': imgs, 'gray': gray, 'label': label}

        return self.transform(sample)

    def to_cartesian_coord(self, polar_coord):
        """Convert a squared-box from polar-like coordinates to cartesian coordinates."""
        cx, cy, r = polar_coord['cx'], polar_coord['cy'], polar_coord['r']
        return [cx - r, cy - r, cx + r, cy + r]

    def load_frames_from_hdf5(self, input_h5, kind):
        """
        Load and decode frames from an HDF5 file.

        Args:
            input_h5 (str): Path to the input HDF5 file.

        Returns:
            List of frames as numpy arrays.
        """
        assert kind in ('rgb', 'opt'), 'Argument `kind` must be `rgb` or `opt`'
        frames = []
        with h5py.File(input_h5, 'r') as f:
            dset = f["frames"]
            for byte_frame in dset:
                # Convert stored bytes back into a numpy array.
                arr = np.frombuffer(byte_frame.tobytes(), dtype=np.uint8)
                # Decode the JPEG bytes back to an image.
                if kind == 'rgb':
                    frame = cv.imdecode(arr, cv.IMREAD_COLOR)
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                else:
                    frame = cv.imdecode(arr, cv.IMREAD_GRAYSCALE)
                frames.append(frame)
        
        return frames
