import glob
import os

import PIL.Image
import torch
import torchvision

from weakly_supervised_classification.model.data_loader import Bag


class Evaluator:
    def __init__(self, model_path: str, batch_size: int = 64):
        self.batch_size = batch_size
        self.model = self.load_model(model_path=model_path)

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    @staticmethod
    def load_model(model_path: str):
        """
        Loads model model from specified path.
        :param model_path: (str) path to save TorchScript model
        :return: model object
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        return torch.jit.load(model_path, map_location=device)

    def _preprocess_batch(self, images_batch):
        bag_images = [self.transforms(img) for img in images_batch]
        bag_images = torch.stack(bag_images)
        bag_images = bag_images.unsqueeze(0)

        return bag_images

    def evaluate_dir(self, images_dir):
        bag = Bag.from_directory(images_dir)

        return self.evaluate_bag(bag)

    def evaluate_bag(self, bag: Bag):
        batch = bag.images

        return self.evaluate_batch(batch)

    @torch.no_grad()
    def evaluate_batch(self, images_batch):
        """
        Evaluates passed list of images.
        :param images_batch: (List[Tuple[PIL.Image, PIL.Image]]) list of PIL.Image objects
        :return: out_class: List[Tuple[float, float]] - probabilities for each class [different, correct] for each pair
        is added to the list
        """
        self.model.eval()

        images_batch = self._preprocess_batch(images_batch)
        outputs = []
        for batch in images_batch.split(self.batch_size):
            outputs += self.model(batch.to(next(self.model.parameters()).device))

        return outputs


# eval = Evaluator(os.path.join('asd/checkpoints1', 'model.pt'))
# imgs = [PIL.Image.open(os.path.join('bags_eval', '7_7_7_2_1_6_7_7_0_0_0', i)) for i in os.listdir(os.path.join('bags_eval', '7_7_7_2_1_6_7_7_0_0_0'))]
# out = eval.evaluate_batch(imgs)
# print(out)