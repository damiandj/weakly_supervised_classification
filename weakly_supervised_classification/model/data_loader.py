from __future__ import annotations

import os
import random

from typing import Optional

import PIL.Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torch.nn.utils.rnn import pad_sequence


class BagDistribution:
    AVERAGE = 10
    STD = 2
    MIN = 5
    MAX = 250000000

    @staticmethod
    def _get_random_value() -> float:
        return random.normalvariate(BagDistribution.AVERAGE, BagDistribution.STD)

    @classmethod
    def get_bag_size(cls) -> int:
        x = BagDistribution._get_random_value()
        while x > BagDistribution.MAX or x < BagDistribution.MIN:
            x = BagDistribution._get_random_value()
        return int(x)


class MnistInstance:
    def __init__(self, image: PIL.Image.Image, label: int):
        self.image = image
        self.label = label


class Bag(list):
    def __init__(self, items=None):
        super().__init__()
        if items:
            for __item in items:
                self.append(__item)
        self.label = None

    @classmethod
    def from_directory(cls, directory_path: str) -> Bag:
        bag = Bag()
        for img_path in os.listdir(directory_path):
            mnist_instance = MnistInstance(image=PIL.Image.open(os.path.join(directory_path, img_path)),
                                           label=int(os.path.splitext(img_path)[0].split('_')[1]))
            bag.append(mnist_instance)

        return bag

    @property
    def images(self):
        return [mnist.image for mnist in self]

    def digit_in_bag(self, digit: int):
        return any(mnist.label == digit for mnist in self)

    def append(self, __object: MnistInstance) -> None:
        if not isinstance(__object, MnistInstance):
            raise Exception(type(__object))
        super(Bag, self).append(__object)

    def bag_dir(self, save_path: str) -> str:
        bag_name = "_".join([str(item.label) for item in self])
        if self.label is not None:
            bag_dir = os.path.join(save_path, str(self.label), bag_name)
        else:
            bag_dir = os.path.join(save_path, bag_name)

        return bag_dir

    def save(self, save_path: str):
        bag_dir = self.bag_dir(save_path=save_path)
        os.makedirs(bag_dir, exist_ok=True)
        for position, item in enumerate(self):
            item: MnistInstance
            _item_count = 0
            item_name = f"{position}_{item.label}_{_item_count}.jpg"
            while os.path.exists(os.path.join(bag_dir, item_name)):
                _item_count += 1
                item_name = f"{position}_{item.label}_{_item_count}.jpg"
            item.image.save(os.path.join(bag_dir, item_name))


class BagCollections(list):
    def __init__(self, *items):
        super().__init__()
        for __item in items:
            self.append(__item)

    def append(self, __object: Bag) -> None:
        if not isinstance(__object, Bag):
            raise Exception()
        super(BagCollections, self).append(__object)


class MnistBagsDataset(data.Dataset):
    DECISION_DIGIT = 7

    def __init__(self, root: str, number_of_bags: int, train: bool = True,
                 transforms: Optional[torchvision.transforms.Compose] = None):
        self.mnist_data = torchvision.datasets.MNIST(root=root, train=train, download=True)
        self.number_of_bags = number_of_bags
        self.transforms = transforms

        self.mnist_data_all_ids = self._prepare_mnist_data_all_ids()
        self.mnist_data_without_decision_digit_ids = self._prepare_mnist_data_without_decision_digit_ids()

        self.__bags = BagCollections()

    @property
    def bags(self):
        return self.__bags

    def __len__(self):
        return len(self.bags)

    def _prepare_mnist_data_all_ids(self) -> list[int]:
        return list(range(len(self.mnist_data.targets)))

    def _prepare_mnist_data_without_decision_digit_ids(self) -> list[int]:
        return [idx for idx in range(len(self.mnist_data.targets)) if
                self.mnist_data.targets[idx] != MnistBagsDataset.DECISION_DIGIT]

    def _prepare_positive_bag(self) -> Bag:
        bag_size = BagDistribution.get_bag_size()
        bag_mnist_ids = random.sample(self.mnist_data_all_ids, bag_size)
        bag = Bag([MnistInstance(*self.mnist_data[idx]) for idx in bag_mnist_ids])
        while not bag.digit_in_bag(MnistBagsDataset.DECISION_DIGIT):
            bag_mnist_ids = random.sample(self.mnist_data_all_ids, bag_size)
            bag = Bag([MnistInstance(*self.mnist_data[idx]) for idx in bag_mnist_ids])
        bag.label = 1

        return bag

    def _prepare_positive_bags(self):
        number_of_positive_bags = int(self.number_of_bags / 2)
        for _ in range(number_of_positive_bags):
            self.__bags.append(self._prepare_positive_bag())

    def _prepare_negative_bag(self) -> Bag:
        bag_size = BagDistribution.get_bag_size()
        bag_mnist_ids = random.sample(self.mnist_data_without_decision_digit_ids, bag_size)
        bag = Bag([MnistInstance(*self.mnist_data[idx]) for idx in bag_mnist_ids])
        bag.label = 0

        return bag

    def _prepare_negative_bags(self):
        number_of_negative_bags = int(self.number_of_bags / 2)
        for _ in range(number_of_negative_bags):
            self.__bags.append(self._prepare_negative_bag())

    def _remove_bags(self):
        self.__bags = BagCollections()

    def _prepare_bags(self):
        self._prepare_positive_bags()
        self._prepare_negative_bags()

    def regenerate_bags(self):
        self._remove_bags()
        self._prepare_bags()

    def save_bags(self, save_path: str):
        for bag in self.bags:
            bag.save(save_path=save_path)

    def __getitem__(self, item: int):
        bag = self.bags[item]
        bag_images = bag.images

        if self.transforms:
            bag_images = [self.transforms(img) for img in bag_images]
        bag_images = torch.stack(bag_images)
        label = torch.from_numpy(np.array([bag.label])).to(torch.int64)

        return bag_images, label


def prepare_train_dataloader(number_of_bags: int = 1000, batch_size: int = 16):
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    train_data = MnistBagsDataset(root=os.path.join('data'),
                                  train=True,
                                  number_of_bags=number_of_bags,
                                  transforms=train_transforms)
    train_data.regenerate_bags()
    loader = data.DataLoader(dataset=train_data,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_fn
                             )

    return loader


def collate_fn(_data):
    inputs = [d[0] for d in _data]
    labels = [d[1] for d in _data]
    inputs = pad_sequence(inputs, batch_first=True)
    labels = torch.tensor(labels)

    return inputs, labels


def prepare_val_dataloader(number_of_bags: int = 100, batch_size: int = 16):
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    val_data = MnistBagsDataset(root=os.path.join('data'),
                                train=True,
                                number_of_bags=number_of_bags,
                                transforms=val_transforms)
    val_data.regenerate_bags()
    loader = data.DataLoader(dataset=val_data,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=collate_fn)

    return loader
