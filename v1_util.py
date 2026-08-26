import torch
import torch.nn.functional as F
import time

from v1_data import generate_random_trajectory

#===universal smoothing function===#
def smooth_tensor(x , kernel_size = 3):

    #x: tensor of shape (T) or (T , C)#
    if x.ndim == 1:
        
        #(batch=1, channel=1, length=T)#
        x= x.view(1 , 1 , -1)

    elif x.ndim == 2:

        #(1 , C , T)#
        x= x.transpose (0 , 1).unsqueeze(0)

        kernel = torch.ones (1 , 1 , kernel_size, device= x.device) / kernel_size

        x_smooth = F.conv1d(x , kernel, padding = kernel_size // 2)

    if x.ndim == 1:

        return x_smooth.view(-1)
    
    else:

        return x_smooth.squeeze(0).transpose(0 , 1)