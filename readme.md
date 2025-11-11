
## Appendix
If you need all the appendix information, including the complete data processing flow, LLMs prompt word scheme, and various additional experiments, please click here (waiting).
## Introduction

The project aims to create a network that understands TPAS to address the issues of rhetorical metonymy, logical reasoning, and metaphorical expression extraction in TPAS.

## Requirements

The code is based on Python 3.9.18 Please install the dependencies as below:  

You should first download `Pytorch` corresponding to the system Cuda version. In the experimental environment of this paper, we used `cuda 11.4`, so the installation code is as follows. You can also go to the [Pytorch](https://pytorch.org/) official website to download other corresponding versions.

Our pytorch== 2.1.2 ; torchvision==0.16.2; torchaudio==2.1.2

Then install other required packages.

```
pip install -r requirements.txt
```

In the experiment, we used the [RoBERTa-wwm](https://huggingface.co/hfl/chinese-roberta-wwm-ext) model and [ChineseBert](https://github.com/ShannonAI/ChineseBert) model. Please download the corresponding model parameters from Huggingface and GitHub.

Please put the pre-trained files in (**\data\pretrained_models**).

## Data

The Data folder contains folders in `Chinese` and `English`. There are `output_all_train_data.txt`, `output_all_dev_data.txt`, and `output_all_test_data.txt` files in each language folder.

## Code

The parameter details of our model are already available in **config. py** and **dataManager. py**. Please run **main/main.py** directly with the data and model installed.

**!!!** 

Our experiment is running on a **40G A100 NVIDIA, please ensure that your graphics card exceeds 20GB at least. Our runtime is 25 hours.**

