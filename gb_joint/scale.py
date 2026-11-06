"""Optional latent-scale constraint for the linear autoencoder.

"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .clustering import encode_full_dataset, pooled_centered_variance


@torch.no_grad()
def project_latent_scale(model, loader, device):
 
    if model.architecture != "linear" or torch.device(device).type != "cpu":
        raise ValueError("Latent scale projection currently requires a linear AE on CPU.")
    target, epsilon = 1.0, 1e-8
    rng_before = torch.get_rng_state().clone()
    was_training = model.training
    try:
        model.eval()
        z_before = encode_full_dataset(model, loader, device)
        recon_before = torch.cat([
            model(batch["image_pt"].to(device)) for batch in loader
        ])
        variance_before = float(pooled_centered_variance(z_before).item())
        if not np.isfinite(variance_before) or variance_before < epsilon * epsilon:
            raise RuntimeError("Latent variance is non-finite or collapsed.")
        factor = float(np.sqrt(target / variance_before))
        encoder_output = [
            module for module in model.encoder.modules() if isinstance(module, nn.Linear)
        ][-1]
        decoder_input = [
            module for module in model.decoder.modules() if isinstance(module, nn.Linear)
        ][0]
        encoder_output.weight.mul_(factor)
        if encoder_output.bias is not None:
            encoder_output.bias.mul_(factor)
        decoder_input.weight.div_(factor)
        z_after = encode_full_dataset(model, loader, device)
        recon_after = torch.cat([
            model(batch["image_pt"].to(device)) for batch in loader
        ])
        variance_after = float(pooled_centered_variance(z_after).item())
        max_reconstruction_error = float((recon_after - recon_before).abs().max().item())
        max_latent_error = float((z_after - factor * z_before).abs().max().item())
        torch.testing.assert_close(recon_after, recon_before, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(z_after, factor * z_before, rtol=2e-5, atol=1e-5)
        if abs(variance_after - target) >= 2e-6:
            raise RuntimeError("The projected latent variance did not reach the target.")
        return {
            "variance_before": variance_before,
            "variance_after": variance_after,
            "scale_factor": factor,
            "max_abs_reconstruction_change": max_reconstruction_error,
            "max_abs_latent_scaling_error": max_latent_error,
        }, z_after
    finally:
        torch.set_rng_state(rng_before)
        model.train(was_training)
