# coding=utf-8
# Copyright 2021 Santiago Hincapie-Potes & The HuggingFace Team All rights reserved.
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

import csv
import json
import random
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image

class MIMICDataset(VisionDataset):
    """
    Dataset for loading image-text data for tasks like CLIP training, Image Captioning.

    Args:
        root: (string): The root path where the dataset is stored
        file_path: (string): Path to the file containing the image_paths and associated captions.
            The expected format is jsonlines where each line is a json object containing to keys.
            `image_path`: The path to the image.
            `captions`: An `array` of captions.
        mode: (string): target format:
            * 'longest': return the longest sections
            * 'docs': return findings and impressions
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        file_path: str,
        mode: str = 'longest',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        root = Path(root)

        if not mode in {'longest', 'docs'}:
            raise ValueError('Invalid mode')

        self.mode = mode

        with open(root / file_path, "r") as f:
            examples = [json.loads(line) for line in f.readlines()]

        self.captions = []
        self.image_paths = []

        for example in examples:
            self.captions.append(example["caption"])
            self.image_paths.append(str(root / example["image_path"]))

    def _load_image(self, idx: int):
        path = self.image_paths[idx]
        return read_image(path, mode=ImageReadMode.RGB)

    def _load_target(self, idx) -> str:
        sections = self.captions[idx]

        if self.mode == 'docs':
            _collection = []
            if 'impression' in sections:
                _collection.append(sections['impression'])

            if 'findings' in sections:
                _collection.append(sections['findings'])

            if len(_collection) == 1:
                output = _collection[0]
            if len(_collection) == 2:
                output = random.choice(_collection)

        if self.mode == 'longest' or len(_collection) == 0:
            longest_section = max(
                filter(lambda x: isinstance(x, str), sections.values()), 
                key=len
            )

            output = longest_section

        return output

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.captions)


class ROCODataset(VisionDataset):
    """
    Dataset for loading image-text data for tasks like CLIP training, Image Captioning.

    Args:
        root: (string): The root path where the dataset is stored
        file_path: (string): Path to the file containing the image_paths and associated captions.
            The expected format is jsonlines where each line is a json object containing to keys.
            `image_path`: The path to the image.
            `captions`: An `array` of captions.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        root = Path(root) / f"{split}/radiology/"
        file_path = f"{split}.csv"

        self.captions = []
        self.image_paths = []

        with open((root / file_path).resolve(), 'r') as buf:
            csv_reader = csv.reader(buf)
            next(csv_reader) # skip header

            for row in csv_reader:
                if len(row) == 3:
                    _, fname, caption = row
                else:
                    print(row)
                self.captions.append(caption.strip())
                self.image_paths.append(str(root / 'images' / fname.strip()))

    def _load_image(self, idx: int):
        path = self.image_paths[idx]
        return read_image(path, mode=ImageReadMode.RGB)

    def _load_target(self, idx: int) -> str:
        return self.captions[idx]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.captions)
