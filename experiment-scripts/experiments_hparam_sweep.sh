
#!/bin/sh

time python experiments_hparamsweep.py --data cifar10 --base standard
time python experiments_hparamsweep.py --data fashionmnist --base standard
time python experiments_hparamsweep.py --data mnist --base standard
time python experiments_hparamsweep.py --data svhn --base standard


time python experiments_hparamsweep.py --data cifar10 --base gmm
time python experiments_hparamsweep.py --data fashionmnist --base gmm
time python experiments_hparamsweep.py --data mnist --base gmm
time python experiments_hparamsweep.py --data svhn --base gmm