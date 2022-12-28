import os

from torch import optim
from torchvision import transforms
import torch
from torch.utils.tensorboard import SummaryWriter

from weakly_supervised_classification.model.data_loader import prepare_train_dataloader, prepare_val_dataloader
from weakly_supervised_classification.model.multi_attention_mil import MultiAttentionMIL, Attention


class Logger:
    def __init__(self):
        self._epoch = 0
        self.epochs = 0
        self.writer = SummaryWriter()

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, val: int):
        self._epochs = val

    def update_epoch(self):
        self._epoch += 1

    def log_epoch(self):
        print(f"Epoch {self._epoch} / {self.epochs}")

    def log_train_loss_accuracy(self, loss: float, accuracy: float):
        self.writer.add_scalar("Loss/train", loss, self._epoch)
        self.writer.add_scalar("accuracy/train", accuracy, self._epoch)
        self.writer.flush()
        print(f"\t train accuracy: {accuracy}, train loss: {loss}")

    def log_eval_loss_accuracy(self, loss: float, accuracy: float):
        self.writer.add_scalar("Loss/eval", loss, self._epoch)
        self.writer.add_scalar("accuracy/eval", accuracy, self._epoch)
        self.writer.flush()
        print(f"\t eval accuracy: {accuracy}, eval loss: {loss}")


class Trainer:
    def __init__(self, number_train_of_bags: int = 1000, number_eval_of_bags: int = 1000, num_attentions: int = 3,
                 lr: float = 0.0001,
                 models_path: str = 'checkpoints'):
        self.lr = lr
        self.models_path = models_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataloader = prepare_train_dataloader(batch_size=4, number_of_bags=number_train_of_bags)
        self.eval_dataloader = prepare_val_dataloader(batch_size=4, number_of_bags=number_eval_of_bags)
        self.eval_dataloader.dataset.save_bags('bags_eval')
        self.model = MultiAttentionMIL(num_attentions=num_attentions)
        self.model.to(self.device)
        self.optimizer = self._prepare_adam_optimizer(lr=lr)
        self.scheduler = None
        self.criterion = self._prepare_criterion()
        self._best_acc = 0.0

        self.logger = Logger()

    def _prepare_adam_optimizer(self, lr: float):
        return optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))

    def _prepare_sgd_optimizer(self, lr: float):
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=lr,
                                    weight_decay=0.005,
                                    momentum=0.9)

        return optimizer

    def _prepare_scheduler(self):
        """
        Prepare scheduler based on optimizer
        :return scheduler: scheduler object
        """
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        return scheduler

    @staticmethod
    def _prepare_criterion():
        return torch.nn.CrossEntropyLoss()

    def _new_optimizer(self):
        self.optimizer = self._prepare_sgd_optimizer(self.lr)
        self.scheduler = self._prepare_scheduler()

    def train(self, epochs):
        """
        Main training method. Runs training and validation loop
        :param epochs: Number of epochs to train (int)
        :return:
        """
        self.logger.epochs = epochs
        for epoch in range(epochs):
            # if epoch == int(epochs / 2):
            #     self._new_optimizer()
            #     print('Prepare a new optimizer and scheduler')
            self.train_dataloader.dataset.regenerate_bags()
            self.logger.update_epoch()
            self.logger.log_epoch()
            self._train_epoch()
            self._eval_epoch()

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
        # self.scheduler.step()

        predictions = output.argmax(axis=1)
        self.correct_predictions += sum(prediction == int(label) for prediction, label in zip(predictions, labels))

    def _eval_epoch(self):
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


trainer = Trainer(number_train_of_bags=100, number_eval_of_bags=100,num_attentions=3)
trainer.train(50)
