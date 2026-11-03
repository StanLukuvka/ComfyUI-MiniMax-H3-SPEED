import comfy.sample
import comfy.samplers


class KSamplerNoise:
    """Same as ComfyUI's RandomNoise (comfy_extras/nodes_custom_sampler.py Noise_RandomNoise)."""

    def __init__(self, seed: int):
        self.seed = seed

    def generate_noise(self, input_latent: dict):
        batch_inds = input_latent.get("batch_index")
        return comfy.sample.prepare_noise(input_latent["samples"], self.seed, batch_inds)


def build_ksampler_inputs(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise=1.0):
    """Build what KSampler builds internally: noise, CFG guider, sampler and sigmas."""
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)
    sampler = comfy.samplers.sampler_object(sampler_name)
    # comfy.samplers.KSampler.__init__ calls set_steps(steps, denoise), which applies
    # DISCARD_PENULTIMATE_SIGMA_SAMPLERS and the denoise slice exactly as KSampler does.
    sigmas = comfy.samplers.KSampler(
        model, steps=steps, device="cpu", sampler=sampler_name, scheduler=scheduler, denoise=denoise
    ).sigmas
    return KSamplerNoise(seed), guider, sampler, sigmas
