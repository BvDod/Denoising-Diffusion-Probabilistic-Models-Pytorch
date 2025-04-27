import torch

class NoiseScheduler:
        """ Class used to manage beta schedule and add forward noise to samples """

        def __init__(self, train_config):
            self.device = "cuda"

            self.timesteps = train_config.timesteps_diff
            self.beta_start = 0.0001
            self.beta_end = 0.02
            self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps, device=self.device)
            self.alphas_cumulative = torch.cumprod((1 - self.betas), dim=0)


        def get_random_ts(self, x):
            """ Get random t for every sample in batch """

            t = torch.randint(0, self.timesteps, size=(x.shape[0],))
            return t


        def add_forward_noise(self, x, t=None):
            """
            Add forward noise to x, such that noise is accumlated noise at timestamp t
            if no t is supplied, draw random t's for every sample
            """

            if t is None:
                t = self.get_random_ts(x).to(self.device)
                alpha_cum = self.alphas_cumulative[t]
                alpha_cum = alpha_cum.reshape((x.shape[0], 1, 1, 1))

            else:
                alpha_cum = self.alphas_cumulative[t]

            noise = torch.randn_like(x).to(self.device)

            return (torch.sqrt(alpha_cum) * x) + (torch.sqrt(1-alpha_cum) * noise), noise, t


        def reverse_noise(self, x, predicted_noise, t):
            """
            Given an image, and the predicted noise at timestep t, move to x_t-1 from x
            Performed by performing the reverse of the forward process.
            """

            denoised_x_mean = (1/torch.sqrt(1-self.betas[t])) * (x - self.betas[t] * predicted_noise / torch.sqrt(1 - self.alphas_cumulative[t]))

            if t <= 1:
                return denoised_x_mean
            else:
                posterior_variance = self.betas[t] * (1-self.alphas_cumulative[t-1]) / (1-self.alphas_cumulative[t])
                noise = torch.randn_like(x).to(self.device)
                denoised_x = denoised_x_mean + torch.sqrt(posterior_variance) * noise
                return denoised_x