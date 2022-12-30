# Summary
## Problem statement
In this research the method to solve Weakly-supervised classification problem using 
Multi-Scale Attention-based Multiple Instance Learning will be presented. Based on [arXiv:2209.03041](https://arxiv.org/abs/2209.03041).

In this case the data point is set (bag) of [MNIST](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html) images
with only one label: correct if at least one `7` is in bag and negative otherwise.

The trained model should classify bags into those with and without `7`.
Notably, an application of the Multi-Scale Attention-based approach provides insight into the contribution of each instance to the bag label.
In other words, the probability that a particular number in the bag is `7` is returned.

## Data
In this specific case each training, validating and testing bag consists of on average `10` images with standard deviation of `2` (normal distribution),
but no less than `5` and no more than `250000000`. Label of bag is positive (`1`) if there is at least one `7` in a data point and zero (`0`) otherwise.

Data samples in [assets/data/test_1](assets/data/test_1).

| ![](assets/test_1_distribution.png)       |
|:-----------------------------------------:|
|  Bag size distribution in test_1 dataset  |


Moreover images are normalized with `mean=0.1307` and `std=0.3081`. Normalization is part of the model definition.

## Model structure
The proposed approach uses `resnet18` as a feature extractor. Input images dimension is changed to `1`. 
Then three Attention blocks computes attentions and images features weighted by attentions.
The final fully connected layer returns values (Softmax) for each class: positive and negative.
Both Softmax values and last attentions are returned from model.

| ![](assets/moded_grpah.png) |
|:---------------------------:|
|     Simple model graph      |

Full model graph in [assets/model_graph_full.pdf](assets/model_graph_full.pdf).

### Attention function 
An attention layer is defined as a weighted sum:

$$z=\sum_{k=1}^K a_k x_k $$

where

$$ a_k = \frac{e^{w^\mathtt{T} \tanh(Vx_k^\mathtt{T})}}{S} $$

and

$$ S = \sum_{j=1}^K e^{w^\mathtt{T} \tanh(Vx_j^{T})} $$

## Learning process
All experiments were conducted on workstation with Intel(R) Core(TM) i7-10750H CPU (2.60GHz), 32 GB RAM and single GPU NVIDAIA Quadro T1000 (4 GB).

Training and validation code details in [README.md](README.md).

At the beginning of each training epoch new set of training bags are created. This prevents overwriting of the model 
and allows for more frequent validation of the model.
The model are trained with learning rate of `5e-05`, weight_decay of `10e-5` and with Adam optimization algorithm.

From each learning process only one model is saved (after an epoch with best `validation_accuracy`).

10 models with the following parameters have been trained:
* epochs: `200`
* training_bags: `200` (for epoch * `200` epochs = `40 000` different bags)
* validation_bags: `1 000`

The learning process of one model took about 33 minutes. Tensorboard outputs in [assets/runs](assets/runs)

| ![](assets/train_loss.png) |
|:--------------------------:|
|       Learning loss        |

| ![](assets/eval_loss.png)  |
|:--------------------------:|
|      Evaluation loss       |

| ![](assets/eval_acc.png)  |
|:-------------------------:|
|    Evaluation accuracy    |

## Metrics for Testing
Each model were tested on testing sets of `10 000` bags based on testing MNIST Dataset. Used metrics:
* `presicion = tp / (tp + fp)`
* `recall = tp / (tp + fn)`
* `accuracy = (presicion + recall) / 2` - testing data is balanced, `5 000` positive and negative bags
* `F1 = 2 * (presicion * recall) / (presicion + recall)` - testing data is balanced

## Results
| Model       | precision | recall  | accuracy | F1         |
|-------------|-----------|---------|----------|------------|
| 0           | 0.9777    | 0.9554  | 0.9665   | 0.9664     |
| 1           | 0.9778    | 0.9616  | 0.9697   | 0.9696     |
| 2           | 0.9691    | 0.9486  | 0.9589   | 0.9588     |
| 3           | 0.9777    | 0.9474  | 0.9626   | 0.9623     |
| 4           | 0.991     | 0.9256  | 0.9583   | 0.9572     |
| 5           |  0.9906   |  0.9498 | 0.9702   | **0.9698** |
| 6           | 0.9762    | 0.9588  | 0.9675   | 0.9674     |
| 7           | 0.9738    | 0.9584  | 0.9661   | 0.966      |
| 8           | 0.9743    | 0.9632  | 0.9688   | 0.9687     |
| 9           | 0.985     | 0.9458  | 0.9654   | 0.965      |
| **average** | 0.9793    | 0.9515  | 0.9654   | 0.9651     |
| **std**     | 0.0072    | 0.011   | 0.0042   | 0.0044     |

All models obtained high metric values (`~98% precision` and `~95% recall`) and standard deviation is low.

The model with highest `F1` value (model `5`) were tested on second testing dataset (`5 000` positive and negative bags).

|   Model |   precision |   recall |   accuracy |     F1 |
|---------|-------------|----------|------------|--------|
|       5 |      0.9907 |   0.9562 |     0.9734 | 0.9731 |

The average time of model evaluation on single bag is `6ms` and maximum time is `32ms` and depends on the size of bag. 

### Examples

| ![](assets/tp/0_2_2_6_5_8_5_7_3_1.jpg)   ![](assets/tp/0_2_2_7_0_2_7_5_6_1_3_9_9.jpg) |
|:-------------------------------------------------------------------------------------:|
|                            True positives with attentions                             |

| ![](assets/fp/0_1_3_9_6_2_3.jpg)  ![](assets/fp/2_5_3_5_0.jpg)  |
|:---------------------------------------------------------------:|
|                 False positives with attentions                 |

| ![](assets/fn/0_7_3_1_1_8_4_6_8.jpg)  ![](assets/fn/5_0_0_9_8_4_0_7.jpg) |
|:------------------------------------------------------------------------:|
|                     False negatives with attentions                      |


## Summary
Proposed method to solve given problem obtain high metric values (`99% precision` and `96% recall`).

Probabilities given form attention layer in most cases allow to correctly indicate the searched number.

Using resnet18 as a feature extractor is enough for MNIST data, but for more complex data it may be need to use a different model
(other resnet structure or different structure).

Possible future improvements:
* choosing the best parameters (using [ray-tune framework](https://www.ray.io/ray-tune)):
  * optimizer (Adam, SGD)
  * learning_rate
  * weight_decay
  * number of epochs and training samples
* check other feature extractors
* add augmentations
  * flips
  * rotations
  * dropout, coarse dropout
* code refactoring and unittests


## References
* Made Satria Wibawa, Kwok-Wai Lo, Lawrence Young, Nasir Rajpoot, **Multi-Scale Attention-based Multiple Instance Learning for Classification of Multi-Gigapixel Histology Images**
[arXiv:2209.03041](https://arxiv.org/abs/2209.03041) and https://github.com/mdsatria/MultiAttentionMIL
* Maximilian Ilse, Jakub M. Tomczak, Max Welling, **Attention-based Deep Multiple Instance Learning**
[arXiv:1802.04712](https://arxiv.org/abs/1802.04712) and https://github.com/AMLab-Amsterdam/AttentionDeepMIL
* Jonathan Glaser, **Attention-Based Deep Multiple Instance Learning**, https://towardsdatascience.com/attention-based-deep-multiple-instance-learning-1bb3df857e24
* Lori Sheng, **Multiple Instance Learning**, https://medium.com/swlh/multiple-instance-learning-c49bd21f5620
* Pytorch 1.13 documentation, https://pytorch.org/docs/stable/index.html