#!/bin/sh

num_epochs="10"
cov_scale="1.0"
num_repeats="1"



# GMM base - - hparams from hyperparameters sweep
# time python experiments.py --data cifar10  --cov_scale 0.8 --base gmm --batch_size 128 --net_channels 128 --lr 0.0002  --n_epochs $num_epochs --repeats $num_repeats
time python experiments.py --data fashionmnist  --cov_scale 0.6 --base gmm --batch_size 128 --net_channels 128 --lr 0.0002 --n_epochs $num_epochs --repeats $num_repeats --skip
# time python experiments.py --data mnist  --cov_scale 0.6 --base gmm --batch_size 128 --net_channels 128 --lr 0.0002 --n_epochs $num_epochs --repeats $num_repeats
# time python experiments.py --data svhn  --cov_scale 0.8 --base gmm --batch_size 128 --net_channels 128 --lr 0.0002 --n_epochs $num_epochs --repeats $num_repeats

# Standard base - hparams from hyperparameters sweep
# time python experiments.py --data cifar10  --cov_scale $cov_scale --base standard --batch_size 128 --net_channels 128 --lr 0.0002  --n_epochs $num_epochs --repeats $num_repeats
# time python experiments.py --data fashionmnist  --cov_scale $cov_scale --base standard --batch_size 128 --net_channels 128 --lr 0.0002 --n_epochs $num_epochs --repeats $num_repeats --skip
# time python experiments.py --data mnist  --cov_scale $cov_scale --base standard --batch_size 128 --net_channels 128 --lr 0.0002 --n_epochs $num_epochs --repeats $num_repeats
# time python experiments.py --data svhn  --cov_scale $cov_scale --base standard --batch_size 128 --net_channels 128 --lr 0.0002 --n_epochs $num_epochs --repeats $num_repeats

