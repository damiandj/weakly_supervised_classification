import os
import time
import datetime

from torch import optim

import torch
from torch.utils.tensorboard import SummaryWriter

from weakly_supervised_classification.model.data_loader import prepare_train_dataloader, prepare_val_dataloader
from weakly_supervised_classification.model.multi_attention_mil import MultiAttentionMIL


class Logger:
    def __init__(self):
        self._epoch = 0
        self.epochs = 0
        self.start_training_time = None
        self.writer = SummaryWriter()

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, val: int):
        self._epochs = val

    def update_epoch(self):
        self._epoch += 1

    def start_training(self):
        self.start_training_time = time.time()

    def end_training(self, best_acc: float, best_acc_epoch: int):
        self.log(f"Finish learning. Total time {datetime.timedelta(seconds=round(time.time() - self.start_training_time))}. "
                 f"Best eval accuracy: {best_acc} from epoch {best_acc_epoch}")

    @staticmethod
    def log(message: str):
        print(message)

    def log_epoch(self):
        self.log(f"Epoch {self._epoch} / {self.epochs}")

    def log_train_loss_accuracy(self, loss: float, accuracy: float):
        self.writer.add_scalar("Loss/train", loss, self._epoch)
        self.writer.add_scalar("accuracy/train", accuracy, self._epoch)
        self.writer.flush()
        self.log(f"\t train accuracy: {accuracy}, train loss: {loss}")

    def log_eval_loss_accuracy(self, loss: float, accuracy: float):
        self.writer.add_scalar("Loss/eval", loss, self._epoch)
        self.writer.add_scalar("accuracy/eval", accuracy, self._epoch)
        self.writer.flush()
        self.log(f"\t eval accuracy: {accuracy}, eval loss: {loss}")

    def log_model_graph(self, model, example):
        self.writer.add_graph(model, example)


class Trainer:
    def __init__(self, number_train_of_bags: int = 1000, number_eval_of_bags: int = 1000, num_attentions: int = 2,
                 lr: float = 0.00005, models_path: str = 'checkpoints'):
        self.lr = lr
        self.models_path = models_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataloader = prepare_train_dataloader(batch_size=4, number_of_bags=number_train_of_bags)
        self.eval_dataloader = prepare_val_dataloader(batch_size=4, number_of_bags=number_eval_of_bags)

        self.model = MultiAttentionMIL(num_attentions=num_attentions)
        self.model.to(self.device)
        self.optimizer = self._prepare_adam_optimizer(lr=lr)
        self.criterion = self._prepare_criterion()
        self._best_acc = 0.0
        self._best_acc_epoch = None

        self.logger = Logger()
        self._log_model()

    def _log_model(self):
        example = torch.zeros(1, 1, 1, 28, 28).to(self.device)
        self.logger.log_model_graph(self.model, example)

    def _prepare_adam_optimizer(self, lr: float):
        return optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=10e-5)

    @staticmethod
    def _prepare_criterion():
        return torch.nn.CrossEntropyLoss()

    def train(self, epochs):
        """
        Main training method. Runs training and validation loop
        :param epochs: Number of epochs to train (int)
        :return:
        """
        self.logger.epochs = epochs
        self.logger.start_training()
        for epoch in range(epochs):
            self.train_dataloader.dataset.regenerate_bags()
            self.logger.update_epoch()
            self.logger.log_epoch()
            self._train_epoch()
            self._eval_epoch(epoch)
        self.logger.end_training(best_acc=self._best_acc, best_acc_epoch=self._best_acc_epoch)

    def _train_epoch(self):
        self.train_total_loss = 0
        self.correct_predictions = 0
        self.model.train()
        for batch_num, (inputs, labels) in enumerate(self.train_dataloader):
            self._train_iteration(inputs, labels)
        self.logger.log_train_loss_accuracy(self.train_total_loss / len(self.train_dataloader.dataset),
                                            accuracy=self.correct_predictions / len(self.train_dataloader.dataset))

    def _train_iteration(self, inputs, labels):
        """
        Train single iteration (batch)
        :param inputs: batch of images to train
        :param labels: batch of labels to train
        """
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()
        output, _ = self.model.forward(inputs)
        loss = self.criterion(output, torch.flatten(labels))
        self.train_total_loss += loss.item()
        loss.backward()
        self.optimizer.step()

        predictions = output.argmax(axis=1)
        self.correct_predictions += sum(prediction == int(label) for prediction, label in zip(predictions, labels))

    def _eval_epoch(self, epoch: int):
        """
        Validate single epoch
        :return:
        """
        self.model.eval()
        self.eval_total_loss = 0
        self.correct_predictions = 0

        for batch_num, (inputs, labels) in enumerate(self.eval_dataloader):
            self._eval_batch(inputs, labels)
        acc = self.correct_predictions / len(self.eval_dataloader.dataset)
        self.logger.log_eval_loss_accuracy(loss=self.eval_total_loss / len(self.eval_dataloader.dataset),
                                           accuracy=acc)

        if acc >= self._best_acc:
            self._best_acc = acc
            self._best_acc_epoch = epoch
            self._save_model()

    @torch.no_grad()
    def _eval_batch(self, inputs_raw, labels_raw):
        """
        Validate single iteration
        :param inputs_raw: batch of images to validate
        :param labels_raw: batch of labels to validate
        :return:
        """
        inputs = inputs_raw.to(self.device)
        labels = labels_raw.to(self.device)
        output, _ = self.model.forward(inputs)
        loss = self.criterion(output, labels)
        self.eval_total_loss += loss.item()

        predictions = output.argmax(axis=1)
        self.correct_predictions += sum(prediction == int(label) for prediction, label in zip(predictions, labels))

    def _save_model(self):
        """
        Save trainer.model to specified direction (create directory if necessary).
        :return:
        """
        os.makedirs(self.models_path, exist_ok=True)
        self.model.eval()
        example = torch.zeros(1, 1, 1, 28, 28).to(self.device)

        with torch.no_grad():
            traced_script_module = torch.jit.trace(self.model, example)
            torch.jit.save(traced_script_module, os.path.join(self.models_path, 'model.pt'))

        print(f"Save TorchScript checkpoint to {os.path.join(self.models_path, 'model.pt')}")
        torch.save(self.model, os.path.join(self.models_path, "model.pth"))
        print(f'Save checkpoint to {os.path.join(self.models_path, "model.pth")}')
