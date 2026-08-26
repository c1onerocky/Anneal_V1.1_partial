import torch
import time
import torch.nn as nn
import torch.nn.functional as F

from v1_diag import print_training_stats


#===Training on Synthetic Data===#
def train_on_synthetic(model , loss_fn , positional_prediction , branch_variance , step_progression ,  epochs = 300 , num_points = 20 , lr = 1e-2 , motion_type = "circular", update_rule = None, noise_floor = None):

    optimizer = torch.optim.Adam(model.parameters() , lr = lr)

    t_traj = torch.linspace(0.0 , 150.0 , steps = num_points)

    trajectories = []

    for traj_idx , (trajectories , positional_prediction ,  pd_step_trigger) in enumerate(trajectories):

        last_print = time.time()

        for epoch in range(epochs):
            
            optimizer.zero_grad()

            motion_type = motion_type

            loss = loss_fn(positional_prediction , pd_step_trigger)

            positional_prediction , step_progression = model(trajectories,  motion_type = motion_type , update_rule = None, noise_floor = None)

            loss.backward()

            #print(positional_prediction.shape)
            #print(loss)

            optimizer.step()
            #print("after optimizer")

            if epoch % 50 == 0:
                print_training_stats(epoch, loss, positional_prediction , branch_variance, step_progression)

    with torch.no_grad():

        noise_floor = 0.0

        positional_prediction = model(pd_step_trigger , trajectories, motion_type = motion_type, update_rule = None, noise_floor = None)

    return   positional_prediction