## Train from scratch
By using our code, we reproduce the results of our Fast MPN-COV ResNet models on ImageNet 2012. At the same time, in order to facilitate the performance evaluation, we also provide the results on CIFAR100.

### Our experiments are running on
 * Tensorflow 2.0.0b0<br>
 * 2 x 1080Ti<br>
 * Cuda 10.0 with CuDNN 7.5<br>
 
## Results
#### Classification results (single crop 224x224, %) on **ImageNet 2012** validation set
<table>
<tr>                                      
    <td rowspan="3" align='center'>Network</strong></td>
    <td rowspan="3" align='center'>Dim</td>
    <td colspan="3" align='center'>Top1_err/Top5_err</td>
    <td colspan="2" rowspan="2" align='center'>Pre-trai