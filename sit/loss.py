import torch
import numpy as np
import torch.nn.functional as F

def lognormal_timestep_density(t, path_type="linear"):
    """
    Calculate the probability density function for timesteps sampled using
    the lognormal method in SILoss.
    
    Args:
        t: torch.Tensor, batch of timesteps in [0, 1]
        path_type: str, either "linear" or "cosine"
    
    Returns:
        torch.Tensor: density values corresponding to input timesteps
    """
    if path_type == "linear":
        # For linear path: t = σ/(1+σ), so σ = t/(1-t)
        # We need to be careful about t=1 (singularity)
        t_safe = torch.clamp(t, 0.0, 0.999999)  # Avoid t=1
        sigma = t_safe / (1.0 - t_safe)
        
        # Jacobian of transformation: dt/dσ = 1/(1+σ)²
        # So dσ/dt = (1+σ)² = (1+t/(1-t))² = 1/(1-t)²
        jacobian = 1.0 / ((1.0 - t_safe) ** 2)
        
    elif path_type == "cosine":
        # For cosine path: t = (2/π) * atan(σ), so σ = tan(πt/2)
        sigma = torch.tan(np.pi * t / 2.0)
        
        # Jacobian: dt/dσ = (2/π) * 1/(1+σ²)
        # So dσ/dt = π/2 * (1+σ²) = π/2 * (1+tan²(πt/2)) = π/2 * sec²(πt/2)
        jacobian = (np.pi / 2.0) * (1.0 / torch.cos(np.pi * t / 2.0)) ** 2
        
    else:
        raise ValueError("path_type must be 'linear' or 'cosine'")
    
    # σ follows log-normal distribution: σ = exp(Z) where Z ~ N(0,1)
    # Density of σ: f_σ(σ) = (1/σ) * (1/√(2π)) * exp(-(log(σ))²/2)
    # But we need the density of t, so we use the change of variables formula:
    # f_t(t) = f_σ(σ(t)) * |dσ/dt|
    
    # Handle edge cases where sigma might be very small or large
    sigma_safe = torch.clamp(sigma, 1e-8, 1e8)
    
    # Log-normal density of σ
    log_sigma = torch.log(sigma_safe)
    lognormal_density_sigma = (1.0 / sigma_safe) * (1.0 / np.sqrt(2 * np.pi)) * torch.exp(-0.5 * log_sigma ** 2)
    
    # Apply change of variables: f_t(t) = f_σ(σ(t)) * |dσ/dt|
    density_t = lognormal_density_sigma * jacobian
    
    return density_t

def calculate_lognormal_weighting(t, path_type="linear"):
    """
    Calculate the weighting w(t) = t/(1-t) * π(t) where π(t) is the lognormal density.
    
    Args:
        t: torch.Tensor, batch of timesteps in [0, 1]
        path_type: str, either "linear" or "cosine"
    
    Returns:
        torch.Tensor: weighting values
    """
    density = lognormal_timestep_density(t, path_type)
    
    if path_type == "linear":
        # For linear path: w(t) = t/(1-t) * π(t)
        t_safe = torch.clamp(t, 1e-8, 0.999999)
        weighting = (t_safe / (1.0 - t_safe)) * density
    elif path_type == "cosine":
        # For cosine path: w(t) = tan(πt/2) * π(t)
        weighting = torch.tan(np.pi * t / 2.0) * density
    else:
        raise ValueError("path_type must be 'linear' or 'cosine'")
    
    return weighting

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            weak_type="None",
            w_scale=0.0,
            weak_loss_ratio=0.2,
            weak_alpha=0.0001,
            weak_model=None,
            num_classes=1000,
            guidance_low=0.0,
            guidance_high=1.0,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.weak_type = weak_type
        self.w_scale = w_scale
        self.weak_loss_ratio = weak_loss_ratio
        self.weak_alpha = weak_alpha
        self.weak_model = weak_model
        self.num_classes = num_classes
        self.guidance_low = guidance_low
        self.guidance_high = guidance_high
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

    def __call__(self, model, images, model_kwargs=None, zs=None, training_duration=0.0):
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
            raise NotImplementedError() # TODO: add x or eps prediction

        current_w_scale = self.w_scale * (training_duration ** self.weak_alpha)
        
        if self.weak_type == "None":
            model_output, zs_tilde  = model(model_input, time_input.flatten(), **model_kwargs)
            denoising_loss = mean_flat((model_output - model_target) ** 2)
        elif self.weak_type == "LayerSkip":
            model_kwargs['double'] = True
            model_output, zs_tilde = model(model_input, time_input.flatten(), **model_kwargs)
            model_output_strong, model_output_weak = model_output.chunk(2)
            model_target = model_target + current_w_scale * (model_output_strong - model_output_weak).detach()
            denoising_loss = mean_flat((model_output_strong - model_target) ** 2)
        elif self.weak_type == "Branch":
            model_output, zs_tilde, x_mid = model(model_input, time_input.flatten(), **model_kwargs)
            denoising_loss_weak = mean_flat((x_mid - model_target) ** 2)
            model_target = model_target + current_w_scale * (model_output - x_mid).detach()
            denoising_loss = mean_flat((model_output - model_target) ** 2)
            denoising_loss = denoising_loss + self.weak_loss_ratio * denoising_loss_weak
        elif self.weak_type == "Separate":
            # Use separate weak model for prediction
            model_output, zs_tilde = model(model_input, time_input.flatten(), **model_kwargs)
            with torch.no_grad():
                weak_model_output, _ = self.weak_model(model_input, time_input.flatten(), **model_kwargs)
            model_target = model_target + current_w_scale * (model_output - weak_model_output).detach()
            denoising_loss = mean_flat((model_output - model_target) ** 2)
        elif self.weak_type == "Uncond":
            y = model_kwargs['y']
            y_null = torch.full_like(y, self.num_classes)
            model_output, zs_tilde = model(model_input, time_input.flatten(), **model_kwargs)
            
            with torch.no_grad():
                weak_model_output, _ = model(model_input, time_input.flatten(), **dict(y=y_null))
            
            # Check if timesteps are within the guidance range
            # Only add the guidance term for timesteps within the range
            guidance_term = current_w_scale * (model_output - weak_model_output).detach()
            model_target = model_target + guidance_term
            
            denoising_loss = mean_flat((model_output - model_target) ** 2)
        elif self.weak_type == "Segmented":
            y = model_kwargs['y']
            y_null = torch.full_like(y, self.num_classes)
            model_output, zs_tilde, x_mid = model(model_input, time_input.flatten(), **model_kwargs)
            
            with torch.no_grad():
                weak_model_output, _, x_mid_weak = model(model_input, time_input.flatten(), **dict(y=y_null))
            
            # Compute AG loss: model_output - x_mid
            ag_loss = mean_flat((model_output - x_mid) ** 2)
            
            # Check timestep intervals
            t_flat = time_input.flatten()  # Shape: [batch_size]
            guidance_mask = (t_flat >= self.guidance_low) & (t_flat <= self.guidance_high)  # Shape: [batch_size]
            t_low_mask = t_flat < self.guidance_low  # t < guidance_low
            
            # When t is in [guidance_low, guidance_high], use uncond loss (cond-uncond)
            guidance_term = current_w_scale * (model_output - weak_model_output).detach()
            guidance_mask_4d = guidance_mask.view(-1, 1, 1, 1)  # Expand to 4D
            model_target_guided = model_target + guidance_mask_4d * guidance_term
            
            # Compute guided loss
            denoising_loss_guided = mean_flat((model_output - model_target_guided) ** 2)
            
            # When t < guidance_low, use AG
            ag_loss_combined = mean_flat((model_output - model_target) ** 2) + self.weak_loss_ratio * ag_loss
            
            # When t > guidance_high, use pure diffusion
            diffusion_loss = mean_flat((model_output - model_target) ** 2)
            
            # Combine based on timestep conditions
            denoising_loss = torch.where(guidance_mask, denoising_loss_guided,
                            torch.where(t_low_mask, ag_loss_combined, diffusion_loss))
            
        else:
            raise NotImplementedError()
        


        # projection loss
        proj_loss = 0.
        if len(zs) > 0 and len(zs_tilde) > 0:
            bsz = zs[0].shape[0]
            for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                    z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                    z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                    proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
            proj_loss /= (len(zs) * bsz)
        
        # Ensure proj_loss is a tensor with the same shape as denoising_loss
        if not isinstance(proj_loss, torch.Tensor):
            # Create a tensor of zeros with the same shape as denoising_loss
            proj_loss = torch.zeros_like(denoising_loss)

        return denoising_loss, proj_loss
