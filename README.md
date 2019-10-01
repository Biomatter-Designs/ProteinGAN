# ProteinGAN
Generative network architecture that may be used to produce de-novo protein sequences.

## Motivation

One of the most challenging task in synthetic biology is to create novel synthetic parts, 
in particular synthetic catalytic biomolecules.

Catalytic biomolecules are a cleaner and greener substitute to chemical catalyzers used all over 
the world. The catalytic biomolecules offer an enormous 17 orders of magnitude chemical reaction 
acceleration as well as excellent stereo-, chemo- and regio-selectivity in aqueous environments. 
Yet, one of the main drawbacks of using enzymes is that for many important chemical reactions, 
efficient enzymes have not yet been discovered or engineered. However, identifying enzyme amino acid 
sequence with the required novel or optimized reaction is a challenging task because the sequence 
space is incomprehensibly large.

### ProteinGAN

ProteinGAN - generative adversarial network architecture that we developed and optimized to generate 
de novo proteins. Once trained, it outputs a desirable amount of proteins belonging to a functional 
class(es) it was trained for.

## Licenses

All material is made available under Creative Commons BY-NC 4.0 license. You can use, redistribute, 
and adapt the material for non-commercial purposes, as long as you give appropriate credit by citing our paper 
and indicating any changes that you've made.

## System requirements
- Operating System: Linux.
- 64-bit Python 3.7 installation.
- blastp: 2.6.0+
- TensorFlow 1.13.1 or newer with GPU support.
- One or more NVIDIA GPUs. Recomendation: NVIDIA at least P100 GPU with 16GB.
- NVIDIA driver 418.87 or newer, CUDA toolkit 10.1 or newer, cuDNN 7.6.2 or newer.

## Datas for training
ProteinGAN expects a number of files in order to be able to train and evaluate the network.

|  File name  |   Data |
|---|---|
| properties.json | File should contain information about max length of sequences and enzyme class.|
| db_train.phr | Output of makeblastdb script using training sequences. Used to evaluate the network during the training. | 
| db_train.pin | Output of makeblastdb script using training sequences. Used to evaluate the network during the training. |  
| db_train.psq | Output of makeblastdb script using training sequences. Used to evaluate the network during the training. |  
| db_val.phr | Output of makeblastdb script using validation sequences. Used to evaluate the network during the training. | 
| db_val.pin | Output of makeblastdb script using validation sequences. Used to evaluate the network during the training. | 
| db_val.psq | Output of makeblastdb script using validation sequences. Used to evaluate the network during the training. | 
| train/{1}_{2}_{3}.tfrecords | Multiple tfrecords containing training sequences. {2}, {3} - are upsampling factors used to balance training dataset | 


## Training networks
Once data is ready, you can train your own ProteinGAN for chosen set of sequences as follows:

1. Edit gan/parameters.py to specify the dataset and training configuration.
2. Run the training script with python train_gan.
3. The results, weights will be stored in specified location.
This location is printed once training script is executed. You can use tensorboard to view all the details.
4. The training may take several days (or weeks) to complete, depending on the configuration.
5. Once training is completed, you can use generate.py to generate chosen number of sequences.
6. Once training is completed, you can use discriminator_scores.py to get discriminator scores for all provided sequences.
7. Once training is completed, you can use test_gan.py to investigate GAN performance via interpolation. 


## Useful links

- Database of annotated enzymes by its function - http://www.uniprot.org/. 
- Database of enzyme reactions: https://www.expasy.org/.
- Paper on generating DNA sequences using GANs - https://arxiv.org/pdf/1712.06148.pdf
- Paper on generating peptides: https://arxiv.org/pdf/1804.01694.pdf

Papers influenced final solution:
- An Empirical Evaluation of Generic Convolutional and Recurrent Networksfor Sequence Modeling: https://arxiv.org/pdf/1803.01271.pdf
- Large Scale GAN Training for High Fidelity Natural Image Synthesis: https://arxiv.org/pdf/1809.11096.pdf
- Progressive Growing of GANs for Improved Quality, Stability, and Variation: https://arxiv.org/pdf/1710.10196.pdf
- Spectral Normalization for Generative Adversarial Networks: https://arxiv.org/abs/1802.05957.pdf
- Improved Techniques for Training GANs: https://arxiv.org/pdf/1606.03498.pdf
- Spectral Normalization for Generative Adversarial Networks: https://arxiv.org/pdf/1802.05957.pdf
- Multi-Scale Context Aggregation by Dilated Convolutions: https://arxiv.org/pdf/1511.07122.pdf
- Self-Attention Generative Adversarial Networks: https://arxiv.org/pdf/1805.08318.pdf
- cGANs with Projection Discriminator: https://arxiv.org/pdf/1802.05637.pdf
- A Style-Based Generator Architecture for Generative Adversarial Networks: https://arxiv.org/pdf/1812.04948.pdf
- Which Training Methods for GANs do actually Converge? https://arxiv.org/pdf/1801.04406


