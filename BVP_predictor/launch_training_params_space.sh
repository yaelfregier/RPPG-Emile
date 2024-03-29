# Camilla, Jul 9, 2020

#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

for epoch in 50 100 200 300; do 
  
    for lr in 1e-3 1e-4 1e-5; do

        echo Running: python module_mind/train.py --epochs=$epoch --lr=$lr  '>' train.out_epochs_$epoch'_lr_'$lr
        python train.py --epochs=$epoch --lr=$lr > train.out_epochs_$epoch'_lr_'$lr

   done

done

