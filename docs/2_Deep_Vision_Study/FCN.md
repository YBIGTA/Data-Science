# Fully Convolutional Networks for Semantic Segmentation
Jonathan Long, Evan Shelhamer, Trevor Darrell. "Fully Convolutional Networks for Semantic Segmentation." 2015.

## Models

Using Dataset

| Implementation | Accuracy | Weights | Memory | Conv Ops | etc | link |
|---|---|---|---|---|---|
| Keras |   | FCN8: 427,127,544, FCN32 : 4,234,357,544 |  |   |  | [Keras](https://github.com/YBIGTA/DL_Models/tree/master/models/fcn/keras)   |
| Tensorflow Slim |   | 256,445,037 | 256,445,037 * 4bytes |   |   | [Slim](https://github.com/YBIGTA/DL_Models/tree/master/models/fcn/tensorflow%20slim) |
| Pytorch | 85.62% (Pixel-Acc) |  134,489,759 | 134,489,759 * 4bytes |  | W.I.P (Voc2012)| [Pytorch](https://github.com/YBIGTA/DL_Models/tree/master/models/fcn/pytorch)

## Tip & Trick

| name | for What | reference |
|---|---|---|
| Skip Layers | to produce accurate and detailed segmentations | - |
| Becoming Fully Convolutional | 1. to be free from the size of image <br> 2. to store location information of each pixel | - |
| Dilated ConV | 1. Dilated convolution for extract global feature | https://arxiv.org/abs/1511.07122 |
|  |  |  |
|  |  |  |


## Error of paper
- add this if any errors
