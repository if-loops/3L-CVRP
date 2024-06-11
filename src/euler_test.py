# sftp://euler.ethz.ch 
# ssh sschoepf@euler.ethz.ch  
# module load new gcc/4.8.2 python/3.7.1 
# busers -w
# lquota
# bsub -I python src/euler_test.py
# bsub -I python src/euler_train.py
# bsub python src/euler_train.py graph batch lr epochs
# pip freeze > requirements.txt
# pip3 install -r requirements.txt --prefix=$HOME/python
# https://scicomp.ethz.ch/wiki/Python
#mkdir -p lib64/python3.7.1/site-packages
# export PYTHONPATH=$HOME/python/lib64/python3.7.1/site-packages #:$PYTHONPATH
# module load python/3.7.1
# pip3 install --user numba

# TensTenorboard
# cd Documents/GitHub/3L-CVRP/src/src
# tensorboard --logdir Documents/GitHub/3L-CVRP/src/src/logs/EULER --reload_multifile True

import numpy as np
import tensorflow as tf

print(np.arange(10))

# bsub -n 48 python src/euler_train.py graph batch lr epochs
# bsub -n 48 -I python src/euler_train.py 3 10 0.0001 5
# bsub -n 48 python src/euler_train.py 15 25 0.001 100

bsub -W 24:00 -n 4 python src/euler_train.py 10 100 0.0001 10000 
bsub -W 72:00 -n 8 python src/euler_train.py 10 100 0.0001 10000
bsub -W 72:00 -n 8 python src/euler_train.py 15 100 0.0001 10000

# Newest iteration:
bsub -n 8 -I python src/euler_train.py 15 128 0.0001 25 0.5 "testrun" 6 5 12 1

bsub -W 120:00 -n 8 python src/euler_train.py 15 128 0.0001 50000 0.5 "_novehpen_noforce_" 6 5 12 1
bsub -W 120:00 -n 4 python src/euler_train.py 15 100 0.0001 50000 0.5 "_novehpen_noforce_" 6 5 12 1


python src/euler_train.py 15 100 0.0001 50000 0.5 "_novehpen_noforce_" 6 5 12 1

bsub -W 120:00 -n 8 python src/euler_train.py 15 128 0.0001 50000 0.5 "_nomix_forced_" 6 5 12 1
bsub -W 120:00 -n 8 python src/euler_train.py 15 128 0.0001 50000 0.5 "_mix_forced_" 6 5 12 1
bsub -W 120:00 -n 8 python src/euler_train.py 15 128 0.0001 50000 0.5 "_mix_forced_" 6 5 12 7
bsub -W 120:00 -n 8 python src/euler_train.py 15 256 0.0001 50000 0.5 "_mix_forced_" 6 5 12 51

bsub -W 120:00 -n 8 python src/euler_train.py 15 128 0.0001 50000 0.5 "_nomix_fulldemand_" 6 5 12 1
bsub -W 120:00 -n 8 python src/euler_train.py 15 128 0.0001 50000 0.5 "_nomix_fulldemand_" 6 5 12 7
bsub -W 120:00 -n 8 python src/euler_train.py 15 128 0.0001 50000 0.5 "_nomix_fulldemand_" 6 5 12 51

bsub -W 120:00 -n 8 python src/euler_train.py 15 128 0.0001 50000 0.5 "_nomix_v3_" 6 5 12 1
bsub -W 120:00 -n 8 python src/euler_train.py 15 128 0.0001 50000 0.5 "_nomix_v3_" 6 5 12 7
bsub -W 120:00 -n 8 python src/euler_train.py 15 128 0.0001 50000 0.5 "_nomix_v3_" 6 5 12 51

bsub -W 120:00 -n 8 python src/euler_train.py 20 128 0.0001 50000 0.5 "_sizemix_" 6 5 12 1

bsub -W 120:00 -n 16 python src/euler_train.py 10 128 0.0001 50000 0.5 "__" 6 5 12 1
bsub -W 120:00 -n 16 python src/euler_train.py 15 128 0.0001 50000 0.5 "_mixedhwlweight_" 6 5 12
bsub -W 120:00 -n 16 python src/euler_train.py 10 128 0.0001 50000 0.5 "_mixedweight_" 30 25 60
bsub -W 120:00 -n 8 python src/euler_train.py 15 128 0.0001 10000 0.5 "_cnn_sdvrp_norm_" 30 25 60
bsub -W 120:00 -n 8 python src/euler_train.py 10 128 0.0001 10000 0.5 "_hwl_" 6 5 12
bsub -W 120:00 -n 8 python src/euler_train.py 10 128 0.0001 10000 0.5 "_hwl_" 30 25 60
bsub -W 120:00 -n 8 python src/euler_train.py 5 128 0.0001 10000 0.5 "_hwl_" 6 5 12
bsub -W 120:00 -n 8 python src/euler_train.py 5 128 0.0001 10000 0.5 "_hwl_" 30 25 60

bsub -n 12 -I python src/euler_train.py 3 128 0.0001 10 0.5 "cnn" 6 5 12

bsub -n 4 -I python src/euler_train.py 5 100 0.0001 10 0.1 "testing"

pip3 install numba --prefix=$HOME/python
pip3 install --user 'numba<=0.40'
pip3 install --user 'scanpy<=1.0'
pip3 install --user 'llvmlite==0.32.1'
# my order:
# module load new gcc/4.8.2 python/3.7.1 

bsub -W 72:00 -n 10 python src/euler_train.py 25 100 0.0001 10000
