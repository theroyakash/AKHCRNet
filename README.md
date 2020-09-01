# AKHCRNet: Bengali handwritten character recognition using deep learning
Source code for the AKHCRNet Paper: Deep neural architecture on bengali hand written character.

## Abstract
Proposal of a state of the art deep neural architectural solution for handwritten character recognition for Bengali alphabets, compound alphabets as well as numerical digits that achieves state-of-the-art accuracy 96.8% in just 11 epochs. Similar work has been done before by Chatterjee, Dutta, et al. 2019 but they achieved 96.12% accuracy in about 47 epochs. The deep neural architecture used in that paper was fairly large considering the inclusion of the weights of the ResNet 50 model which is a 50 layer Residual Network. This proposed model achieves higher accuracy as compared to any previous work & in a little number of epochs. ResNet50 is a good model trained on the ImageNet dataset, but I propose an HCR network that is trained from the scratch on Bengali characters without the "Ensemble Learning" that can outperform previous architectures.

#### Preprints
[arXiv](https://arxiv.org/abs/2008.12995)

#### Downloads:
Weight files can be downloaded from [here](https://github.com/theroyakash/AKHCRNet/releases/download/v1.0.0/model.h5). Or try the following code in a jupyter cell.
```python
!wget https://github.com/theroyakash/AKHCRNet/releases/download/v1.0.0/model.h5
```
