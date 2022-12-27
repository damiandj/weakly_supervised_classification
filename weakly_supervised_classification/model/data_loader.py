import random

import PIL.Image
import torch.utils.data as data
import torchvision


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
    def __init__(self, *items):
        super().__init__()
        self.extend(items)
        self.label = 0

    def append(self, __object: MnistInstance) -> None:
        if not isinstance(__object, MnistInstance):
            raise Exception()
        super(Bag, self).append(__object)


class BagCollections(list):
    def __init__(self, *items):
        super().__init__()
        self.extend(items)

    def append(self, __object: Bag) -> None:
        if not isinstance(__object, Bag):
            raise Exception()
        super(BagCollections, self).append(__object)


class MnistBagsDataset(data.Dataset):
    DECISION_DIGIT = 7

    def __init__(self, root: str, number_of_bags: int, train: bool = True):
        self.mnist_data = torchvision.datasets.MNIST(root=root, train=train, download=True)
        self.number_of_bags = number_of_bags

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

    def _prepare_bags(self):
        self._prepare_positive_bags()
        self._prepare_negative_bags()

    def __getitem__(self, item: int) -> Bag:

        return self.bags[item]


def prepare_train_dataloader(number_of_bags: int = 1000, batch_size: int = 16, num_workers: int = 4):
    train_transforms = ...
    train_data = MnistBagsDataset(root=os.path.join('..', '..', 'data', 'data'),
                                  train=True,
                                  number_of_bags=number_of_bags)

    loader = data.DataLoader(dataset=train_data,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=True)

    return loader


def prepare_val_dataloader(number_of_bags: int = 100, batch_size: int = 16, num_workers: int = 4):
    val_transforms = ...
    val_data = MnistBagsDataset(root=os.path.join('..', '..', 'data', 'data'),
                                train=True,
                                number_of_bags=number_of_bags)

    loader = data.DataLoader(dataset=val_data,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    return loader


import os

asd = MnistBagsDataset(os.path.join('..', '..', 'data', 'data'))
asd._prepare_positive_bag()
