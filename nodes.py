import torch
from tqdm import trange

from comfy.samplers import KSAMPLER


import comfy.k_diffusion.utils as utils
def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / utils.append_dims(sigma, x.ndim)


class SamplerInversedEulerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "refine": ("INT", {"default": 0, "min": 0, "max": 1}),
        }}

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, refine):
        @torch.no_grad()
        def sample_inversed_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
            """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
            extra_args = {} if extra_args is None else extra_args
            s_in = x.new_ones([x.shape[0]])
            for i in trange(1, len(sigmas), disable=disable):

                dt = sigmas[i] - sigmas[i-1]

                denoised = model(x, sigmas[i-1] * s_in, **extra_args)
                d = to_d(x, sigmas[i-1], denoised)

                for _ in range(refine):
                    # backward
                    x_ = x + d * dt
                    # forward
                    denoised = model(x_, sigmas[i] * s_in, **extra_args)
                    d_ = to_d(x_, sigmas[i], denoised)
                    d = d * 0.5 + d_ * 0.5

                x = x + d * dt

                if callback is not None:
                    callback(
                        {'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            return x

        ksampler = KSAMPLER(sample_inversed_euler)
        return (ksampler, )