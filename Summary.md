# Summary
## Problem statement
In this research, we proposed method, which can be used to preparing model solving Weakly-supervised classification problem using 
Multi-Scale Attention-based Multiple Instance Learning based on [arXiv:2209.03041](https://arxiv.org/abs/2209.03041).


The trained model should classify bags with and without number `7`.
Notably, an application of the Multi-Scale Attention-based approach provides insight into the contribution of each instance to the bag label. 

## Data
In this specific case each training, validating and testing bag consists of on average `10` images with standard deviation of `2`, but no less
than `5` and no more than `250000000`. Label of bag is positive (`1`) if there is at least one `7` in a data point and zero (`0`) otherwise.

Moreover images are normalized with `mean=0.1307` and `std=0.3081`. Normalization is part of the model definition.

## Model structure
The proposed approach uses resnet18 as a feature extractor. Input images dimension was changed to `1`. 

![](assets/moded_grpah.png)


| Model       |   precision |   recall |   accuracy |     F1 |
|-------------|-------------|----------|------------|--------|
| 0           |      0.9777 |   0.9554 |     0.9665 | 0.9664 |
| 1           |      0.9778 |   0.9616 |     0.9697 | 0.9696 |
| 2           |      0.9691 |   0.9486 |     0.9589 | 0.9588 |
| 3           |      0.9777 |   0.9474 |     0.9626 | 0.9623 |
| 4           |      0.991  |   0.9256 |     0.9583 | 0.9572 |
| 5           |      0.9906 |   0.9498 |     0.9702 | 0.9698 |
| 6           |      0.9762 |   0.9588 |     0.9675 | 0.9674 |
| 7           |      0.9738 |   0.9584 |     0.9661 | 0.966  |
| 8           |      0.9743 |   0.9632 |     0.9688 | 0.9687 |
| 9           |      0.985  |   0.9458 |     0.9654 | 0.965  |
| **average** |      0.9793 |   0.9515 |     0.9654 | 0.9651 |
| **std**     |      0.0072 |   0.011  |     0.0042 | 0.0044 |
