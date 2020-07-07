# Source Code for the AKHCRNet Paper

Deep neural architecture on bengali hand written character.

This is a proposal of a state of the art deep neural architectural solution for handwritten character recognitionfor Bengali alphabets, compound alphabets as well as numerical digits that achieves state-of-the-artaccuracy 96.8% in just 11 epochs.  Similar work has been done before by Chatterjee, Dutta, et al.2019[1] but they achieved 96.12% accuracy in about 47 epochs. The deep neural architecture used inthat paper was fairly large considering the inclusion of the weights of the ResNet 50 model whichis a 50 layer Residual Network. This proposed model achieves higher accuracy as compared to anyprevious work & in a little number of epochs. ResNet50 is a good model trained on the ImageNetdataset, but I propose an HCR network that is trained from the scratch on Bengali characters withoutthe "Ensemble Learning" that can outperform previous architectures.
