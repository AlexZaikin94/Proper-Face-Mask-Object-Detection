#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate main
echo "start job"
nvidia-smi
python main.py
echo "done"