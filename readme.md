## Step 1. Forward Noise Sampling

During the forward process, noise is gradually added to the original image $x_0$, where the new image $x_t$ only depends on the previous image $x_{t-1}$. Noise is added from a Gaussian distribution such that the produced image will gradually move to a distribution with pure noise from the standard normal distribution. The amount of noise added at each step is managed using $\beta_t$, and is usually gradually increased as the steps increase. A single forward transition is achieved using: 

<img src="figures/image.png" alt="drawing" width="300"/>

However, since iteratively adding Gaussians only results in a new Gaussian, we can calculate the accumulated noise scaling over any number of steps, and sample the distorted image at any time step t, directly from the original image. This results in the following formula:

<img src="figures/image-1.png" alt="drawing" width="325"/>

I have implemented the noise scheduler and the actual noise sampling process in *functions/noise_schedulur.py*. The following images are a sample of the noise process, directly calculated at t's of 0, 66, 132 and 200.

<img src="figures/cumulative_noise.png" alt="drawing" width="325"/>

During training, we randomly draw t from a uniform distribution, where t is different for every sample in the batch. For $\beta_t$, I use a schedule that linearly increases from $\beta_{start}$ to $\beta_{end}$ as the steps increase.
