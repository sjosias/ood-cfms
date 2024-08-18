#!/bin/sh

# num_epochs="400"
num_epochs="10"
cov_scale="1.0"
num_repeats="1"


# # GMM base - hparams from hyperparameters sweep
# time python experiments.py --data cifar10 --train --n_epochs $num_epochs --cov_scale 0.8 --base gmm --batch_size 256 --net_channels 128 --repeats $num_repeats --lr 0.0002 
time python experiments.py --data fashionmnist --train --n_epochs $num_epochs --cov_scale 0.6 --base gmm --batch_size 128 --net_channels 128 --repeats $num_repeats --lr 0.0002
# time python experiments.py --data mnist --train --n_epochs $num_epochs --cov_scale 0.6 --base gmm --batch_size 256 --net_channels 128 --repeats $num_repeats --lr 0.0002
# time python experiments.py --data svhn --train --n_epochs $num_epochs --cov_scale 0.8 --base gmm --batch_size 256 --net_channels 128 --repeats $num_repeats --lr 0.0002


# Standard base - hparams from hyperparameters sweep
# time python experiments.py --data cifar10 --train --n_epochs $num_epochs --cov_scale $cov_scale --base standard --batch_size 256 --net_channels 128 --lr 0.0002 --repeats $num_repeats
# time python experiments.py --data fashionmnist --train --n_epochs $num_epochs --cov_scale $cov_scale --base standard --batch_size 256 --net_channels 128 --lr 0.0002 --repeats $num_repeats
# time python experiments.py --data mnist --train --n_epochs $num_epochs --cov_scale $cov_scale --base standard --batch_size 128 --net_channels 128 --lr 0.0002 --repeats $num_repeats
# time python experiments.py --data svhn --train --n_epochs $num_epochs --cov_scale $cov_scale --base standard --batch_size 256 --net_channels 128 --lr 0.0002 --repeats $num_repeats

