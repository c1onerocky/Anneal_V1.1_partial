import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from v1_train import train_on_synthetic



#===Visualization===#
def visualize_trajectory(model, loss, positional_prediction , pd_step_trigger ,  title = "Trajectory"):
    
    positional_prediction_visualization = positional_prediction.detach().cpu().numpy()

    positional_prediction_visualization = positional_prediction_visualization[0]

    #print("VIZ POS:", positional_prediction_visualization.shape)

    latent_uncertainty = model.latent_uncertainty

    visualization_uncertainty = latent_uncertainty.detach().cpu().numpy()

    mean_uncertainty = visualization_uncertainty.mean()

    max_uncertainty = visualization_uncertainty.max()

    threshold = mean_uncertainty + visualization_uncertainty.std()

    outlier_index = visualization_uncertainty > threshold

    fig = plt.figure(figsize=(11,8))

    ax = fig.add_subplot(111, projection='3d')

    normal_uncertainty = (visualization_uncertainty - visualization_uncertainty.min()) / (visualization_uncertainty.max() - visualization_uncertainty.min() + 1e-8)

    colors = plt.cm.viridis(normal_uncertainty)

    for branch in range(positional_prediction_visualization.shape[0]):
                        trajectory = positional_prediction_visualization[branch]

                        ax.plot(trajectory[: , 0], trajectory[: , 1], trajectory[: , 2], color = colors[branch], linewidth = 3)
                        

    if pd_step_trigger is not None:

        gt = pd_step_trigger.detach().cpu().numpy()

        ax.plot(gt[: , 0] , gt[: , 1] , gt[:  2] , color='black' , linewidth=1 , linestyle='--' , label='True Path')

    ax.scatter(positional_prediction_visualization[outlier_index , 0] , positional_prediction_visualization[outlier_index , 1] ,
                positional_prediction_visualization[outlier_index , 2] , color='red' , s=40 , label='High Uncertainty')

    metrics = f"Mean σ: {mean_uncertainty:.4f} | Max σ: {max_uncertainty:.4f}"

    if loss is not None:

        metrics = f"Loss: {loss:.4f} | " + metrics

    ax.set_title(f"{title}\n{metrics}")

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    ax.legend()

    ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.show()