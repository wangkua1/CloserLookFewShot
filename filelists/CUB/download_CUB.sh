#!/usr/bin/env bash
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
gdown https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45
tar -zxvf CUB_200_2011.tgz
python write_CUB_filelist.py
