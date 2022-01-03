set -e
:<<!
*****************Instruction*****************
Here you can easily creat a model by selecting
an arbitray backbone model and global method.
You can fine-tune it on your own datasets by
using a pre-trained model.
Modify the following settings as you wish !
*********************************************
!

#***************Backbone model****************
#Our code provides some mainstream architectures:

#mpncovresnet: mpncovresnet50, mpncovresnet101

#You can also add your own network in src/network
arch=mpncovresnet101
#*********************************************

#***************global method****************
#Our code provides some global method at the end
#of network:
#GAvP (global average pooling),
#MPNCOV (matrix power normalized cov pooling),
#...
#You can also add your own method in src/representation
image_representation=MPNCOV
# short description of method
description=reproduce
#*********************************************

#*******************Dataset*******************
#Choose the dataset folder
benchmark=Aircrafts
datadir=/home/rudy/Downlo