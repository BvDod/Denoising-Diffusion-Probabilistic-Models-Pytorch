## Step 1: Forward Noise Sampling

During the forward process, noise is gradually added to the original image $x_0$, where the new image $x_t$ only depends on the previous image $x_{t-1}$. Noise is added from a Gaussian distribution such that the produced image will gradually move to a distribution with pure noise from the standard normal distribution. The amount of noise added at each step is managed using $\beta_t$, and is usually gradually increased as the steps increase. A single forward transition is achieved using: 

<img src="figures/image.png" alt="drawing" width="300"/>

However, since iteratively adding Gaussians only results in a new Gaussian, we can calculate the accumulated noise scaling over any number of steps, and sample the distorted image at any time step t, directly from the original image. This results in the following formula:

<img src="figures/image-1.png" alt="drawing" width="325"/>

I have implemented the noise scheduler and the actual noise sampling process in *functions/noise_schedulur.py*. The following images are a sample of the noise process, directly calculated at t's of 0, 66, 132 and 200.

<img src="figures/cumulative_noise.png" alt="drawing" width="325"/>

During training, we randomly draw t from a uniform distribution, where t is different for every sample in the batch. For $\beta_t$, I use a schedule that linearly increases from $\beta_{start}$ to $\beta_{end}$ as the steps increase.

## Step 2: Predicting the added Noise: Modified U-NET
<img src="image.png" alt="drawing" width="400"/>

The used model has been implemented from scratch by me, and I aimed to keep its formulation as close to the architecture used in the original paper. The official tensorflow implementation of the authors was used in the process.

The UNET architecture is an auto-encoder, where the spatial dimensions are gradually reduced while the amount of channels are increased. After reaching a bottleneck, the process is reversed, and spatial dimensions are gradually increased while channels are reduced, untill we once again reach the initial dimensions. To prevent a vanishing gradient, and preserve finer information, intermediary states during downsampling are saved, and then slowly concatenated again at each level of upsampling.

My implementation consists of 5 resolution levels (from 128x128 to 8x8). For downsampling, each resolution levels consists of 2 ResNet block, and for upsampling, 3 ResNet blocks. At the resolution levels of 16x16, and in the bottleneck, we also use self-attention layer.

### Timestep embedding: Sinusodial
The same model is used to predict the added noise for all different timesteps, and thus with different amounts of noise. It is thus helpfull to give the model information about which timestep we are currently predicting for each image. To achieve this, a sinusodial embeddings like the one detailed in "Attention is all you need" is used throughout the model. More specifically, this embedding is summed to the current state at the start of each resnet-block.

A sinusodial embedding is a collection of Sin and Cos functions applied to the posiition (or timestep) of the sample. The collection consists of Sin and Cos functions at different frequencies. These different frequencies allow the model to differentiate between timesteps at different scales. The benefits of a sinusodial embedding is that they dont have to be learned, and any learned relationship naturally extends between and even outside of the timesteps the relationships was learned at.

## Step 3: Training the model

## Step 4: Sampling new images

## Results


## How to start Training
```
from train import train_diffusion, DiffusionConfig, ModelConfig

diff_config = DiffusionConfig(
    dataset_name = "FFHQ",
    learning_rate = 0.001,
    batch_size= 16,
    use_augmentation = False,
    timesteps_diff = 200
)

model_config = ModelConfig(
    lowest_resolution_size = 8,
    dim_multiply = [1,2,2,4,8],
    transformer_layers = [3],
    base_dim = 64
)
train_diffusion(diff_config, model_config)
```

