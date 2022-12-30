import glob
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
from statistics import stdev

from weakly_supervised_classification.model.data_loader import Bag
from weakly_supervised_classification.model.model_evaluator import Evaluator


class Tester:
    def __init__(self, models_path: str, data_path: str, output_path: str, save_correct_attentions: bool = False):
        self.models_path = models_path
        self.data_path = data_path
        self.output_path = output_path
        self.save_correct_attentions = save_correct_attentions

        self.models_metrics = {}
        self.models_paths = self._collect_models()

    def _collect_models(self) -> dict[str]:
        out = {}
        if os.path.exists(os.path.join(self.models_path, "model.pt")):
            out[os.path.basename(self.models_path)] = os.path.join(self.models_path, "model.pt")
            return out

        for model_dir in os.listdir(self.models_path):
            if not os.path.exists(os.path.join(self.models_path, model_dir, "model.pt")):
                raise Exception(f'model.pt not found in {os.path.join(self.models_path, model_dir)}')
            out[model_dir] = os.path.join(self.models_path, model_dir, "model.pt")

        return out

    def _save_bag(self, bag: Bag, model_name: str, dir_name: str):
        save_dir = os.path.join(self.output_path, model_name, dir_name)
        os.makedirs(save_dir, exist_ok=True)
        bag.save(save_dir)

    @staticmethod
    def plot_attention(bag: Bag, attentions: list[float], prediction: int, label: int, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig = plt.figure(figsize=(20, 8), constrained_layout=True)
        fig.suptitle(f"True Label: {label}    Predicted Label: {prediction}", size=20)
        subfigs = fig.subplots(nrows=1, ncols=len(bag))
        for col, subfig in enumerate(subfigs):
            subfig.imshow(bag.images[col])
            subfig.set_xlabel(round(attentions[col], 4))
        fig.savefig(save_path)
        plt.close(fig)

    def compute_metrics(self):
        headers = ['Model', 'precision', 'recall', 'accuracy', 'F1']
        body = []

        def _precision(tp, fp):
            if not tp + fp:
                return 0
            return tp / (tp + fp)

        def _recall(tp, fn):
            if not tp + fn:
                return 0
            return tp / (tp + fn)

        def _accuracy(prec, rec):
            """
            Balanced data
            """
            return (prec + rec) / 2

        def _f1(prec, rec):
            """
            Balanced data
            """
            if not prec + rec:
                return 0
            return 2 * (prec * rec) / (prec + rec)

        precisions, recalls, accuracies, f1s = [], [], [], []
        for model_name, model_metrics in self.models_metrics.items():
            precision = _precision(model_metrics['tp'], model_metrics['fp'])
            precisions.append(precision)
            recall = _recall(model_metrics['tp'], model_metrics['fn'])
            recalls.append(recall)
            accuracy = _accuracy(precision, recall)
            accuracies.append(accuracy)
            f1 = _f1(precision, recall)
            f1s.append(f1)
            body.append([model_name, round(precision, 4), round(recall, 4), round(accuracy, 4), round(f1, 4)])
        if len(precisions) > 1:
            body.append(['**average**',
                         round(sum(precisions) / len(self.models_metrics), 4),
                         round(sum(recalls) / len(self.models_metrics), 4),
                         round(sum(accuracies) / len(self.models_metrics), 4),
                         round(sum(f1s) / len(self.models_metrics), 4)])
            body.append(['**std**', round(stdev(precisions), 4), round(stdev(recalls), 4), round(stdev(accuracies), 4),
                         round(stdev(f1s), 4)])
        results_summary = tabulate(body, headers, tablefmt="github")

        print(results_summary)

    def test(self):
        for model_name, model_path in self.models_paths.items():
            self.models_metrics[model_name] = {
                'tp': 0,
                'tn': 0,
                'fp': 0,
                'fn': 0
            }
            evaluator = Evaluator(model_path=model_path)
            positive_bags = glob.glob(os.path.join(self.data_path, '1', '*'))
            negative_bags = glob.glob(os.path.join(self.data_path, '0', '*'))
            for bag_path in positive_bags + negative_bags:
                self.test_single_bag(bag_path, evaluator, model_name, positive_bags)
        self.compute_metrics()

    def test_single_bag(self, bag_path, evaluator, model_name, positive_bags):
        bag = Bag.from_directory(bag_path)
        probs, attentions = evaluator.evaluate_bag(bag)
        model_class = probs.argmax(axis=1)[0].item()
        if bag_path in positive_bags:
            if model_class == 0:
                self.models_metrics[model_name]['fn'] += 1
                self._save_bag(bag=bag, model_name=model_name, dir_name='fn')
                save_path = f"{bag.bag_dir(os.path.join(self.output_path, model_name, 'fn'))}.jpg"
                self.plot_attention(bag=bag, attentions=attentions.squeeze().tolist(), prediction=model_class,
                                    label=1, save_path=save_path)
            else:
                self.models_metrics[model_name]['tp'] += 1
                if self.save_correct_attentions:
                    save_path = f"{bag.bag_dir(os.path.join(self.output_path, model_name, 'tp'))}.jpg"
                    self.plot_attention(bag=bag, attentions=attentions.squeeze().tolist(), prediction=model_class,
                                        label=1, save_path=save_path)
        else:
            if model_class == 0:
                self.models_metrics[model_name]['tn'] += 1
                if self.save_correct_attentions:
                    save_path = f"{bag.bag_dir(os.path.join(self.output_path, model_name, 'tn'))}.jpg"
                    self.plot_attention(bag=bag, attentions=attentions.squeeze().tolist(), prediction=model_class,
                                        label=0, save_path=save_path)
            else:
                self.models_metrics[model_name]['fp'] += 1
                self._save_bag(bag=bag, model_name=model_name, dir_name='fp')
                save_path = f"{bag.bag_dir(os.path.join(self.output_path, model_name, 'fp'))}.jpg"
                self.plot_attention(bag=bag, attentions=attentions.squeeze().tolist(), prediction=model_class,
                                    label=0, save_path=save_path)
