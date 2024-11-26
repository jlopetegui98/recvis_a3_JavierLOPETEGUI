## Object recognition and computer vision 2024/2025

Javier Alejandro LOPETEGUI GONZALEZ

*MVA Master, ENS Paris-Saclay*

*This Repository is based on the one provided with the orientation of the assigment 3 for the Object Recognition and Computer Vision class. It contains my implementations to solve this assignment. I obtained a public test accuracy of 92.89%.*

### Assignment 3: Sketch image classification
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jlopetegui98/recvis_a3_JavierLOPETEGUI/blob/main/RecVis24_A3.ipynb)
#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```

#### Dataset
We will be using a dataset containing 500 different classes of sketches adapted from the [ImageNet Sketch](https://github.com/HaohanWang/ImageNet-Sketch).
Download the training/validation/test images from [here](https://www.kaggle.com/competitions/mva-recvis-2024/data). The test image labels are not provided.

#### Training and validating your model

To run the training loop you should follow the next instructions:

1. Download the dataset (kaggle credentials required):
```bash
!mkdir ~/.kaggle #create the .kaggle folder in your root directory
## INSERT CREDENTIALS
!echo '{KAGGLE CREDENTIALS}' > ~/.kaggle/kaggle.json #write kaggle API credentials to kaggle.json
!chmod 600 ~/.kaggle/kaggle.json  # set permissions
!pip install kaggle #install the kaggle library
!kaggle competitions download -c mva-recvis-2024
!unzip mva-recvis-2024.zip
```
2. Run the following line to run the best solution model:
```bash
!python recvis24_a3/main.py --model_name dinov2
```

I added the following parameters to customize the transfer learning approach:
- weight_path: Indicates the version of the Dinov2 model to use as feature extractor. Just used for Dinov2 model, not supported for the Resnet-50 baseline. The possible values are:
  - "facebook/dinov2-giant": ViT-g version of Dinov2 (default value)
  - "facebook/dinov2-large": ViT-L version of Dinov2
  - "facebook/dinov2-base": ViT-B version of Dinov2
- embedding_strategy: Indicates the embedding used as features for classification:
  - "cls": Use the cls token embedding
  - "seq_emb": Use the tokens'embeddings pooled average
  - "cls+seq_emb": Use the concatenation of the two embeddings listed before (default value)
- "frozen_strategy": Indicates the freezing paremeters strategy:
  - "none": Not freezing any parameter
  - "all": Freeze all the feature extractor parameters (default value)
  - "n-1_attention": Freeze all the feature extractor parameters but the last attention head
- "aug_flag": a boolean indicating wheter to use data augmentation or not (default value: False)
- "dropout": a float value indicating the dropout level after the feature extraction (default value: 0.0)

#### Evaluating your model on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```
python evaluate.py --data [data_dir] --model [model_file] --model_name [model_name]
```

#### Logger

The training metrics for all the approaches alreaddy tried are available in the following wand report: [HW3_Training_Report](https://api.wandb.ai/links/nlp-tasks/qr77to53)

#### Report

The report for the solution implemented is available in the file: [Report_JavierLOPETEGUI.pdf](https://github.com/jlopetegui98/recvis_a3_JavierLOPETEGUI/blob/main/HW3_Report_Javier_LOPETEGUI.pdf)

#### Acknowledgments
Adapted from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.<br/>
Origial adaptation done by Gul Varol: https://github.com/gulvarol<br/>
New Sketch dataset and code adaptation done by Ricardo Garcia and Charles Raude: https://github.com/rjgpinel, http://imagine.enpc.fr/~raudec/
