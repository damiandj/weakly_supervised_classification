import os
import random
import argparse

from weakly_supervised_classification.model.data_loader import MnistBagsDataset
from weakly_supervised_classification.model.tester import Tester
from weakly_supervised_classification.model.trainer import Trainer


def prepare_test_sets():
    random.seed(12345)
    parser = argparse.ArgumentParser('Preparing two test datasets for weakly-supervised-learning',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bags', help='Amount bags for each dataset', type=int, default=10000)
    parser.add_argument('--output-directory', help='Output directory path', type=str, default='data')
    args = parser.parse_args()

    test_path_1 = os.path.join(args.output_directory, "test_1")
    if os.path.exists(test_path_1):
        raise Exception(f"Test dataset 1 exists in {test_path_1}")
    os.makedirs(test_path_1)

    test_path_2 = os.path.join(args.output_directory, "test_2")
    if os.path.exists(test_path_2):
        raise Exception(f"Test dataset 2 exists in {test_path_2}")
    os.makedirs(test_path_2)

    test_data_1 = MnistBagsDataset(root=os.path.join('data'),
                                   train=False,
                                   number_of_bags=args.bags)
    test_data_1.regenerate_bags()
    test_data_1.save_bags(test_path_1)

    test_data_2 = MnistBagsDataset(root=os.path.join('data'),
                                   train=False,
                                   number_of_bags=args.bags)
    test_data_2.regenerate_bags()
    test_data_2.save_bags(test_path_2)

    print(f'Save {len(test_data_1)} test bags to {test_path_1} and '
          f'{len(test_data_2)} test bags to {test_path_2}')


def train_test_models():
    parser = argparse.ArgumentParser('Learning and testing models for weakly-supervised-learning',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--training-bags', help='Amount bags for each training epoch', type=int, default=200)
    parser.add_argument('--validation-bags', help='Amount bags for validation', type=int, default=1000)
    parser.add_argument('--attentions', help='Amount additional attentions in model', type=int, default=2)
    parser.add_argument('--epochs', help='Training epochs', type=int, default=200)

    parser.add_argument('--models', help='Models amount', type=int, default=10)
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.00005)
    parser.add_argument('--output-directory', help='Output directory path', type=str,
                        default='checkpoints')
    parser.add_argument('--test-data', help='Testing data path', type=str,
                        default='data/test_1')

    args = parser.parse_args()

    for num_model in range(args.models):
        trainer = Trainer(number_train_of_bags=args.training_bags, number_eval_of_bags=args.validation_bags,
                          num_attentions=args.attentions, lr=args.lr,
                          models_path=os.path.join(args.output_directory, str(num_model)))
        trainer.train(args.epochs)
    tester = Tester(models_path=args.output_directory, data_path=args.test_data, output_path='results')
    tester.test()


def test_models():
    parser = argparse.ArgumentParser('Testing models for weakly-supervised-learning',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--models-directory', help='Models directory, or path to model directory.',
                        type=str, default='checkpoints')
    parser.add_argument('--output-directory', help='Output directory path', type=str,
                        default='results_best_model')
    parser.add_argument('--test-data', help='Testing data path', type=str,
                        default='data/test_2')

    args = parser.parse_args()
    tester = Tester(models_path=args.models_directory, data_path=args.test_data, output_path=args.output_directory)
    tester.test()
