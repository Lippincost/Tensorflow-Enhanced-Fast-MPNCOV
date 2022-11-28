
## Train from scratch
In order to facilitate the performance evaluation, we provide the results on CIFAR100.

### Our experiments are running on
 * Tensorflow 2.0.0b0<br>
 * 1 x 1080Ti<br>
 * Cuda 10.0 with CuDNN 7.5<br>
 
### Our experiments details:
1. GAvP:
```    
    representation = {'function': GAvP,
                      'input_dim': 2048}
```
The networks are trained within 140 epochs with the initial learningrate of 0.1, which is reduced to 0.01 and 0.001 at the 90th and 120th epoch, respectively.The mini-batch size is 128 and weight decay is 1e-4.

2. MPNCOV:
```
    representation = {'function': MPNCOV,
                      'iterNum': 5,
                      'input_dim': 256,
                      'dimension_reduction': 128,
                      'dropout_p': 0.5}
```
The networks are trained within 110 epochs with the initial learningrate of 0.25, which is re