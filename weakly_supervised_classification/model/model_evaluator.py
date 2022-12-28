import os

import PIL.Image
import torch
import torchvision


class Evaluator:
    def __init__(self, model_path: str):
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
        device = torch.device('cpu')
        return torch.jit.load(model_path, map_location=device)

    def _preprocess_batch(self, images_batch):
        bag_images = [self.transforms(img) for img in images_batch]
        bag_images = torch.stack(bag_images)
        bag_images = bag_images.unsqueeze(0)

        return bag_images

    @torch.no_grad()
    def evaluate_batch(self, images_batch, batch_size=64):
        """
        Evaluates passed list of images.
        :param images_batch: (List[Tuple[PIL.Image, PIL.Image]]) list of PIL.Image objects
        :param batch_size: (int) number of triples to be processed in one batch
        :return: out_class: List[Tuple[float, float]] - probabilities for each class [different, correct] for each pair
        is added to the list
        """
        self.model.eval()

        images_batch = self._preprocess_batch(images_batch)
        outputs = []
        for batch in images_batch.split(batch_size):
            outputs += self.model(batch.to(next(self.model.parameters()).device))

        return outputs


eval = Evaluator(os.path.join('checkpoints', 'model.pt'))
imgs = [PIL.Image.open(os.path.join('bags_eval', '2_9_0_4_6_2_3', i)) for i in os.listdir(os.path.join('bags_eval', '2_9_0_4_6_2_3'))]
out = eval.evaluate_batch(imgs)
print(out)