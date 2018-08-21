# SSD: Single Shot MultiBox Detector
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. ECCV 2016.

## Models

Using Dataset

| Implementation | Accuracy | Weights | Memory | Conv Ops | etc |
|---|---|---|---|---|---|
| Keras |   |   |  |   |    |
| TensorFlow |   |   |  |   |   |
| PyTorch |   |   | |   |   |

## Tip & Trick

| name | for What | reference |
| ---  | ---      |    ---    |
| Multi-scale featuremap | allow prediction at multiple scales |  -  |
| Default boxes | allow separately detect objects with different ratio |  'Anchor' in [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)  |
|  jaccard overlay  |  to get best box   |  a.k.a. [IOU](https://cdn-images-1.medium.com/max/800/1*_Xf5FUbuUgq8GNyITM3Dwg.png)  |
|   Smooth L1 loss   |  loss function for bbox regression  |  [image](https://www.researchgate.net/publication/322582664/figure/fig5/AS:584361460121600@1516334037062/The-curve-of-the-Smooth-L1-loss.png), [paper](https://arxiv.org/abs/1504.08083)       |
| mAP | eval metric | [explanation](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173) |
| hard negative mining | to sample from a lot of negative boxes | [paper](https://arxiv.org/pdf/1604.03540.pdf) |
|  a trous algorithm  | filling the 'holes' while transforming FC weights into CONV weight | [wikipedia](https://en.wikipedia.org/wiki/Stationary_wavelet_transform) |
| NMS(Nom-Max Suppression | to leave only one bbox (with highest confidence level) during inference time | [coursera - deep learning](https://www.coursera.org/lecture/convolutional-neural-networks/non-max-suppression-dvrjH) |

## Error of paper
- 
