# VQ-VAE with PixelCNN prior

## Workflow
- Train the Vector Quantised Variational AutoEncoder (VQ-VAE) for discrete representation and reconstruction.
- Use PixelCNN to learn the priors on the discrete latents for image sampling.

## Acknowledgement
- VQ-VAE is originally mentioned in the paper [Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937.pdf).
- PixelCNN is proposed in the papers [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759) and [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328).
- Implementation of VQ-VAE (without priors) is based on [the official codes from Google DeepMind](https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb). Note that the implementation here does not rely on the [Sonnet library](https://github.com/deepmind/sonnet).
- 

## Results
<table align='center'>
<tr align='center'>
<td> </td>
<td> Testing data </td>
<td> Reconstruction </td>
<td> Random samples </td>
<td> Samples based on PixelCNN prior </td>
</tr>
<tr align='center'>
<td> MNIST </td>
<td><img src = 'Figures/Test_data_mnist.png' height = '150px'>
<td><img src = 'Figures/Test_recon_mnist.png' height = '150px'>
<td><img src = 'Figures/Sample_random_mnist.png' height = '150px'>
<td><img src = 'Figures/Sample_pixelcnn_mnist.png' height = '150px'>
</tr>
<tr align='center'>
<td> cifar-10 </td>
<td><img src = 'Figures/Test_data_cifar10.png' height = '150px'>
<td><img src = 'Figures/Test_recon_cifar10.png' height = '150px'>
<td><img src = 'Figures/Sample_random_cifar10.png' height = '150px'>
<td><img src = 'Figures/Sample_pixelcnn_cifar10.png' height = '150px'>
</tr>
</table>
