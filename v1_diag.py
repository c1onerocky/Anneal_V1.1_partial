import torch

def count_parameters(model):
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total = sum(p.numel() for p in model.parameters())

    return {"trainable_parameters": trainable, "total_parameters": total}

def print_model_parameters(model, progress):

    stats = count_parameters(model)

    print("====================")
    print("==Model Parameters==")
    print("====================")
    print(f"trainable: {stats ['trainable_parameters']:,}")
    print(f"total: {stats['total_parameters']:,}")

def print_training_stats(model, epoch, loss, branch_variance, latent_uncertainty,latent_imag_out,latent_real_out):
    print("=" * 40)
    print(f"Epoch:{epoch}")
    print(f"loss: {loss:.6f}")
    print(f"branch_variance - distance_in_form_of_ki_mean:{branch_variance.mean().item():.6f}")
    print(f"branch_variance - distiance_in_form_of_ki_max:{branch_variance.max().item():.6f}")
    print(f"latent_magnitude_mean:{latent_uncertainty.mean().item():.6f}")
    print(f"latent_magnitude_max:{latent_uncertainty.max().item():.6f}")
    print(f"latent_imag_uncertainty_mean:{latent_imag_out.mean().item():.6f}")
    print(f"latent_imag_uncertainty_max:{latent_imag_out.max().item():.6f}")
    print(f"latent_real_magnitude_mean:{latent_real_out.mean().item():.6f}")
    print(f"latent_real_magnitude_max:{latent_real_out.max().item():.6f}")
    print(f"delta_uncertainty_mean:{model.delta_uncertainty.mean().item():.6f}")
    print(f"delta_uncertainty_max:{model.delta_uncertainty.max().item():.6f}")

