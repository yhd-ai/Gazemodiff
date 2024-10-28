#!/bin/bash


source conda activate
conda create -n humanmac python=3.8
conda activate humanmac_yan

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirement.txt