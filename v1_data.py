import torch

#===Synthetic Trajectory Generator===#
def generate_random_trajectory(num_points=20 , motion_type='circular' , noise=0.000):

    t_traj = torch.linspace(0.0 , 150.0 , steps=num_points)

    if motion_type == 'linear':

        x = t_traj * 0.05

        y = t_traj * 0.02

        z = t_traj * 0.01

    elif motion_type == 'circular':

        x = torch.sin(t_traj * 0.05) * 5

        y = torch.cos(t_traj * 0.05) * 5

        z = t_traj * 0.01

    elif motion_type == 'oscillatory':

        x = torch.sin(t_traj * 0.1) * 3

        y = torch.sin(t_traj * 0.2) * 2

        z = torch.cos(t_traj * 0.15) * 2

    elif motion_type == 'random_walk':

        x = torch.cumsum(torch.randn(num_points) * 0.1 , dim=0)

        y = torch.cumsum(torch.randn(num_points) * 0.1 , dim=0)

        z = torch.cumsum(torch.randn(num_points) * 0.1 , dim=0)

    else:

        raise ValueError("Unknown motion_type")

    pos = torch.stack([x, y, z] , dim=1)

    if noise > 0.0:

        pos += torch.randn_like(pos) * noise

    return t_traj , pos