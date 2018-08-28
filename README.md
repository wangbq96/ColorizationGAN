# Colorization GAN

This is codes for *Image Colorization Based on Conditional Wasserstein Generative Adversarial Networks*

Please read the following blog for details https://yahaha312.github.io/ColorizationGAN/

## Requirement:
* Python 3.6.2
* CUDA 9.0 
* cuDNN 7.0
* Tensorflow 1.6
* Scipy 1.0
* Pillow 5.0

## Usage
### Prepare data
* Download LSUN/bedroom_train data.
* Maximum center crop the data and reshape into 256*256.
* divide them into training data and testing data set.
* put training data into `./img_data/lsun_bedroom/train/`.
* put testing data into `./img_data/lsun_bedroom/test/`.

You can use function `copy_img` in `tools.py` to prepare data easily.

### Train

```
python main.py --is_train=True --some_paramter=some_value
```

### Test

```
python main.py --is_train=False --some_paramter=some_value
```

## Acknowledgments

Codes are based on [pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow)
and [COLORGAN](https://github.com/ccyyatnet/COLORGAN).
Thanks for their excellent work!
