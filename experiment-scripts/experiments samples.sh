#!/bin/sh
num_repeats="1"


# Standard base
# time python experiments.py --data fashionmnist  --cov_scale 1.0 --base standard --sample --batch_size 128 --net_channels 128 --lr 0.0002 --n_epochs 150 --repeats $num_repeats
# time python experiments.py --data mnist  --cov_scale 1.0 --base standard --sample --batch_size 128 --net_channels 128 --lr 0.0002 --n_epochs 150 --repeats $num_repeats
# time python experiments.py --data cifar10  --cov_scale 1.0 --base standard --sample --batch_size 128 --net_channels 128 --lr 0.0002 --n_epochs 150 --repeats $num_repeats
# time python experiments.py --data svhn  --cov_scale 1.0 --base standard --sample --batch_size 128 --net_channels 128 --lr 0.0002 --n_epochs 150 --repeats $num_repeats



# GMM base
time python experiments.py --data fashionmnist  --cov_scale 0.6 --base gmm --sample --batch_size 128 --net_channels 128 --lr 0.0002 --n_epochs 10 --repeats $num_repeats --skip
# time python experiments.py --data mnist  --cov_scale 0.6 --base gmm --sample --batch_size 128 --net_channels 128 --lr 0.0002 --n_epochs 150 --repeats $num_repeats
# time python experiments.py --data cifar10  --cov_scale 0.8 --base gmm --sample --batch_size 128 --net_channels 128 --lr 0.0002 --n_epochs 150 --repeats $num_repeats
# time python experiments.py --data svhn  --cov_scale 0.8 --base gmm --sample --batch_size 128 --net_channels 128 --lr 0.0002 --n_epochs 150 --repeats $num_repeats
