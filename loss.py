import torch
import numpy as np

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class LSEP_Loss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            multi_weights=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.weights_start = multi_weights[0]
        self.weights_end = multi_weights[1]
        self.n_weights = multi_weights[2]
        self.weights_type = multi_weights[3]
        
    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def make_time_masks(self, time_, n):
        masks = []
        for i in range(n):
            lower = i / n
            upper = (i + 1) / n
            if i < n - 1:
                mask = (time_ >= lower) & (time_ < upper)
            else:
                mask = (time_ >= lower)
            masks.append(mask)
        return masks

    def __call__(self, model, images, model_kwargs=None):
        if self.weights_type == 'constant':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() 

        model_output, zs_class  = model(model_input, time_input.flatten(), **model_kwargs)

        denoising_loss = mean_flat((model_output - model_target) ** 2)

        if self.weights_type == 'constant':
            loss_class = self.loss_fn(zs_class, model_kwargs['y'])

        elif self.weights_type == 'steps':
            n_chunk = self.n_weights
            masks = self.make_time_masks(time_input.flatten(), n_chunk)
            weights = torch.linspace(self.weights_start, self.weights_end, steps=n_chunk).to(model_output.device)

            loss_class = 0.0
            total_count = 0
            for mask, weight in zip(masks, weights):
                if mask.any():
                    idx = mask.nonzero(as_tuple=True)[0]
                    
                    y_sub = model_kwargs['y'][idx]
                    zs_sub = zs_class[idx]

                    loss = self.loss_fn(zs_sub, y_sub)
                    weighted_loss = weight * loss

                    loss_class += weighted_loss.sum()
                    total_count += loss.numel()

            loss_class /= total_count


        return denoising_loss, loss_class

