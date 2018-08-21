# ImageNet Classification with Deep Convolutional Neural Networks
Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.

## Models

Using Dataset

| Implementation | Accuracy | Weights | Memory | Conv Ops | etc | link |
|---|---|---|---|---|---|---|
| Keras |   |  35,659,688 | 35,659,688 * 4bytes |   |  |   [link](https://github.com/YBIGTA/DL_Models/blob/master/models/alexnet/keras/Alexnet_keras.ipynb)  |
| Tensorflow Slim |   | 62,378,344  | 62,378,344 * 4bytes |  |   | [link](https://github.com/YBIGTA/DL_Models/blob/master/models/alexnet/tensorflow%20slim/Alexnet%20(Slim).ipynb)  |
| Pytorch | 90.1% (CIFAR 10, added BN) | 58,323,690  | 58,323,690 * 4bytes |  |   |[link](https://github.com/Jooong/DLCV/blob/master/classification/models.py#L71)|

## Tip & Trick

| name | for What | reference |
|---|---|---|
| Relu | faster train / *training deeper network* | [Paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.165.6419&rep=rep1&type=pdf) |
| LRN  | better generalization (replaced with BN later)  | [LRN in caffe](http://caffe.berkeleyvision.org/tutorial/layers/lrn.html) |
| Overlapping Pooling | to avoid overfitting | - |
| Drop Out | to avoid overfitting | [Paper](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf) |
| Color PCA augmentation | to avoid overfitting, data augmentation | - |


## Error of paper
- 224x224 is actually 227x227
