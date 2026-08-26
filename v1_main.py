import torch
import torch.nn as nn
import time
import torch.nn.functional as F

from v1_data import generate_random_trajectory
from v1_config import AnnealConfig
from v1_model import ProbTrajectory5k
from v1_train import train_on_synthetic
from v1_diag import count_parameters, print_model_parameters
from v1_eval_viz import  visualize_trajectory


#===main execution===#
if __name__ == "__main__":

    
    config = AnnealConfig()
    print("MAIN-config loaded:succesfully")
    
    model = ProbTrajectory5k()
    print("MAIN-model loaded successfully")

    pos_proj = model.pos_proj

    device = model.branch_embed.device

    number_of_branches = model.number_of_branches

    step_progression = model.step_progression

    loss_fn = nn.MSELoss()

    print ("Test Run Started")

    torch.manual_seed(50)

    #===trajectory prep===#
    trajectories = []
    #print("TRAJECTORY PREP")

    for _ in range(3):

        t_traj , pos = generate_random_trajectory(num_points=20 , motion_type='circular' , noise = 0.000)

        model.branch_update = model._mlp_branch ()

        branch_update = model.branch_update

        trajectories.append((branch_update , number_of_branches, step_progression))

        progress = torch.tensor(0.5)
            
        progress = progress.unsqueeze(-1)
            
        assert progress.ndim in [1 , 2]

    #print("branch_embed:", model.branch_embed.shape)
    #print("expected mlp input:" , model.mlp1.in_features )
    
    time_feature_dimension = model.time_feature_dimension

    #sinusodal coupling over trajectory steps, might revisit just incase implicit and doesnt need to be explicit since the 
    #complexified math already has sinusodal interactions

    frequency = torch.arange(1 , time_feature_dimension // 2 + 1 , device = progress.device)
            
    phase_angle = progress * frequency * 2 * torch.pi
    
    t_feat = torch.cat([torch.sin(phase_angle) , torch.cos(phase_angle)] , dim = -1)

    P_d = torch.zeros(step_progression , 3, device = device)

    P_d.to(device = P_d.device)

    theta_d = torch.zeros(step_progression , 3, device = device)

    pd_step_trigger = P_d

    #===branch state===#

    positional_prediction = model.pos_proj

    pos_proj = model.pos_proj

    output = model(motion_type = None , update_rule = None, noise_floor = None) 

    latent_imag_out = model.latent_imag_out
        
    positional_prediction = pos_proj(latent_imag_out) 
    
    count_parameters(model)
    
    print_model_parameters(model, progress)


    #print("OUTPUT CREATED")
    #print("output shpe:" , output[0].shape)
    #print("TEST RUN COMPLETE")

    #print(type(model)) 
    #print(model)
    #===training===#
    print("BEGIN TRAIN ON SYNTHETIC")

    

    train_on_synthetic(model , positional_prediction , trajectories, step_progression , epochs = 300 , lr = 1e-2,  num_points = 20 ,  motion_type = None, update_rule = None, noise_floor = None)
    print("BEGIN EVALUATION")
    
    print ("Run complete")

    visualize_trajectory(model, positional_prediction , pd_step_trigger , title = "Trajectory")