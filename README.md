# What is this
This repository contains everything needed to train various networks (Alexnet, VGG, Resnet) 
with **Bitpack** and **Bitunpack** on Tensorflow based on the paper Reducing Data 
Motion to Accelerate the Training of Deep Neural Networks (https://arxiv.org/abs/2004.02297)

**Bitpack** and **Bitunpack** routines are machine-specific hence two folders 
are provided for the x86 and POWER versions.

# What do you need to run
* An x86/POWER machine with GPUs
* Tensorflow
* gcc on x86 and XLC on POWER
* CUDA
* ImageNet ILSVRC-2012 dataset in TFRecord format

# How to run
1. Depends on the CPU model go to either bitpack\_x86 or bitpack\_power9, issue 
command "make".
1. Choose either one of the three available networks to run (Alexnet, VGG, 
   Resnet). An example job script file is provided.

Sicong Zhuang
sicong.zhuang@gmail.com
