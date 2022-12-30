# weakly_supervised_classification

Module to training, testing and evaluation of model.

Research report: [Summary.md](Summary.md).

## Installation
For using `pytorch` with GPU follow the installation instruction from [pytorch.org](https://pytorch.org/)

In a directory with `setup.py` file run
```commandline
pip install .
```
## Problem statement
The module can be used to preparing model solving Weakly-supervised classification problem using 
Multi-Scale Attention-based Multiple Instance Learning [arXiv:2209.03041](https://arxiv.org/abs/2209.03041).

In this specific case each training, validating and testing bag consists of on average `10` images with standard deviation of `2`, but no less
than `5` and no more than `250000000`. Label of bag is positive (`1`) if there is at least one `7` in a data point and zero (`0`) otherwise.

The trained model should classify bags with and without number `7`.
Notably, an application of the Multi-Scale Attention-based approach provides insight into the contribution of each instance to the bag label. 

The proposed approach uses resnet18 as a feature extractor.

## Usage
The module contains several CLI endpoints
### `prepare_test_sets`
```
usage: Preparing two test datasets for weakly-supervised-learning [-h] [--bags BAGS] [--output-directory OUTPUT_DIRECTORY]

optional arguments:
  -h, --help            show this help message and exit
  --bags BAGS           Amount bags for each dataset (default: 10000)
  --output-directory OUTPUT_DIRECTORY
                        Output directory path (default: data)
```
Creates two testing sets of bags based on testing [MNIST dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html).

Each dataset contains `bags / 2` positive and negative bags.


### `train_test_models`
```
usage: Learning and testing models for weakly-supervised-learning [-h] [--training-bags TRAINING_BAGS] [--validation-bags VALIDATION_BAGS] [--attentions ATTENTIONS] [--epochs EPOCHS] [--models MODELS] [--lr LR]
                                                                  [--output-directory OUTPUT_DIRECTORY] [--test-data TEST_DATA]

optional arguments:
  -h, --help            show this help message and exit
  --training-bags TRAINING_BAGS
                        Amount bags for each training epoch (default: 200)
  --validation-bags VALIDATION_BAGS
                        Amount bags for validation (default: 1000)
  --attentions ATTENTIONS
                        Amount additional attentions in model (default: 2)
  --epochs EPOCHS       Training epochs (default: 200)
  --models MODELS       Models amount (default: 10)
  --lr LR               Learning rate (default: 5e-05)
  --output-directory OUTPUT_DIRECTORY
                        Output directory path (default: checkpoints)
  --test-data TEST_DATA
                        Testing data path (default: data/test_1)
```
Main training and testing CLI. 
Training and validation bags are crated based on training [MNIST dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html).

Each dataset (training and validation) contains `bags / 2` positive and negative bags.

Training data is recreated at the beginning of each training epoch. 

From each learning process only one model is saved (best `validation_accuracy`):
* `model.pt` - traced script module (using [torch.jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html)),
can be used to model evaluation locally or on inference server (like [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server))
* `model.pth` - model weights, using to continue training

### `test_models`
```
usage: Testing models for weakly-supervised-learning [-h] [--models-directory MODELS_DIRECTORY] [--save-correct-attentions] [--output-directory OUTPUT_DIRECTORY] [--test-data TEST_DATA]

optional arguments:
  -h, --help            show this help message and exit
  --models-directory MODELS_DIRECTORY
                        Models directory, or path to model directory. (default: checkpoints)
  --save-correct-attentions
                        Saves all attentions images (default: False)
  --output-directory OUTPUT_DIRECTORY
                        Output directory path (default: results_best_model)
  --test-data TEST_DATA
                        Testing data path (default: data/test_2)
```

### Demo
```
prepare_test_sets --bags 1000 --output-directory data
```
Creates testing data: `data\test_1` and `data\test_2` datasets.
```
train_test_models --models 10 --test-data data\test_1
```
Prepares 10 models and tests them on `data\test_1` dataset.
```
test_models --models-directory checkpoints/0 --test-data data\test_2
```
Tests model `0` on `data\test_2` dataset.