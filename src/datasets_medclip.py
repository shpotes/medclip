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

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image

TextTarget = Union[str, Dict[str, str]]

class MIMICDataset(VisionDataset):
    """
    Dtaset for loading image-text data for tasks like CLIP training, Image Captioning.

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

    def _load_target(self, idx) -> TextTarget:
        sections = self.captions[idx]

        num_sections = 0

        if self.mode == 'docs':
            output = {}
            if 'impression' in sections:
                output['impression'] = sections['impression']
                num_sections += 1

            if 'findings' in sections:
                output['findings'] = sections['findings']
                num_sections += 1

            if num_sections == 1:
                output = next(output.values())

        if self.mode == 'longest' or num_sections == 0:
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
