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
    <td colspan="2" rowspan="2" align='center'>Pre-trained models<br>(tensorflow)</td>
</tr>
<tr>
    <td rowspan="2" align='center'>paper</td>
    <td colspan="2" align='center'>reproduce</td>
</tr>
<tr>
    <td align='center'><strong>tensorflow</strong></td>
    <td align='center'><a href="https://github.com/jiangtaoxie/fast-MPN-COV" title="标题">pytorch</a></td>
    <td align='center'>GoogleDrive</td>
    <td align='center'>BaiduDrive</td>
</tr>
<tr>
    <td>fast-MPN-COV-VGG-D</td>
    <td rowspan="3" align='center'> 32K</td>
    <td align='center'>26.55/8.94</td>
    <td align='center'><strong>23.98/7.12</strong></td>
    <td align='center'>23.98/7.12</td>
    <td align='center'><a href="https://drive.google.com/open?id=19c8ei0FdeRMfeITBApvrjsV49lp1-2ss" title="标题">650.4M</a></td>
    <td align='center'><a href="https://pan.baidu.com/s/13u1nih7bC1b4Mgn9APYxBA" title="标题">650.4M</a></td>
</tr>
<tr>
    <td>fast-MPN-COV-ResNet50</td>
    <td align='center'>22.14/6.22</td>
    <td align='center'><strong>21.57/6.14</strong></td>
    <td align='center'>21.71/6.13</td>
    <t