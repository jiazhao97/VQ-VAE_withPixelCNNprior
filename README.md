# VQ-VAE_withPixelCNNprior
Implementation of Vector Quantised VAE (VQ-VAE) with PixelCNN prior in Tensorflow.

## Results
<table align='center'>
<tr align='center'>
<td> </td>
<td> Testing data </td>
<td> Reconstruction based on VQ-VAE </td>
<td> Random samples </td>
<td> Samples based on PixelCNN prior </td>
</tr>
<tr align='center'>
<td> MNIST </td>
<td><img src = 'Figures/Test_data_mnist.png' height = '160px'>
<td><img src = 'Figures/Test_recon_mnist.png' height = '160px'>
<td><img src = 'Figures/Sample_random_mnist.png' height = '160px'>
<td><img src = 'Figures/Sample_pixelcnn_mnist.png' height = '160px'>
</tr>
<tr align='center'>
<td> Fashion-MNIST </td>
<td><img src = 'examples/fashionmnist4.png' height = '160px'>
<td><img src = 'examples/fashionmnist8.png' height = '160px'>
<td><img src = 'examples/fashionmnist16.png' height = '160px'>
<td><img src = 'examples/fashionmnist32.png' height = '160px'>
</tr>
</table>
