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
                print(t.shape)
                print(self.alphas_cumulative.shape)
                alpha_cum = self.alphas_cumulative[t]
                print(alpha_cum.shape)
                alpha_cum = alpha_cum.reshape((x.shape[0], 1, 1, 1))
                print(alpha_cum.shape)
            
            else:
                alpha_cum = self.alphas_cumulative[t]

            noise = torch.randn_like(x).to(self.device)
            print(noise.shape)

            return (torch.sqrt(alpha_cum) * x) + (torch.sqrt(1-alpha_cum) * noise), noise
