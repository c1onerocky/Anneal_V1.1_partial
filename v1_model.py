import torch
import math
import time
import torch.nn.functional as F
import torch.nn as nn

from v1_data import generate_random_trajectory
from v1_train import train_on_synthetic
from v1_eval_viz import  visualize_trajectory
from v1_config import AnnealConfig

print("IMPORTS SUCCESSFUL")

#===ProbTrajectory5k Model===#
class ProbTrajectory5k(nn.Module):

    def __init__(self, depth = 28 ,layers = 1 ,  epsilon = 0.1 , number_of_branches = 55 , *, motion_type = None):

        super().__init__()

        config = AnnealConfig()

        self.embed_dimension = config.embed_dimension

        self.input_dimension = config.input_dimension

        self.number_of_branches = number_of_branches

        self.branch_embed_dimension = config.branch_embed_dimension

        self.state_dimension = config.state_dimension

        self.imag_dimension = config.imag_dimension

        self.hidden_dimension = config.hidden_dimension

        self.time_feature_dimension = config.time_feature_dimension

        self.step_progression = 20

        self.number_of_branches = number_of_branches

        self.layers = layers

        self.depth = depth

        self.epsilon = epsilon

        self.branch_embed = nn.Parameter(torch.randn(self.input_dimension , self.step_progression,  self.embed_dimension) * 0.1)

        branch_embed = self.branch_embed

        device = self.branch_embed.device

        self.register_buffer("freqs" , torch.exp(torch.linspace(math.log(1.0) , math.log(10.0) , self.time_feature_dimension//2)))

        #+++(REVISIT unsure if necessary at the moment, might be optional per usecase)+++#

        self.time_proj = nn.Linear(self.time_feature_dimension , self.time_feature_dimension , bias=False)

        self.pos_proj = nn.Linear(self.state_dimension, self.hidden_dimension, bias= False)

        self.input_dimension = self.embed_dimension + self.time_feature_dimension

        self.mlp1 = nn.Linear(self.input_dimension , self.step_progression , 4 * self.imag_dimension)

        self.mlp2 = nn.Linear(self.hidden_dimension , self.step_progression , 4 * self.state_dimension)

        self.delta_readout = nn.Linear(self.state_dimension , 3)

        self.branch_logits = nn.Parameter(torch.zeros(number_of_branches))

        self.res_layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(self.state_dimension) , nn.Linear (self.state_dimension , self.state_dimension) , nn.ReLU() , nn.Linear(self.state_dimension, self.state_dimension)
            ) for _ in range(layers)
        ])

        #print("branch_embed:" , self.branch_embed.shape)

    #print("MODEL CLASS LOADED")

    def _mlp_branch(self):

        embed = self.branch_embed
        print("embed:", type(embed))
        print("embed shape:", getattr(embed, "shape", None))
        print("embed device:", getattr(embed, "device", None))

        if isinstance(embed, list):

            embed = torch.as_tensor(embed , dtype = torch.float32 , device = self.mlp1.weight.device)

        if not torch.is_tensor(embed):

            raise TypeError(f"embed must be a tensor, got {type(embed)}")
        
        if embed.shape[0] != self.number_of_branches:

            #print("[ANNEAL TRACE] embed failure detected")
            #print("device:", embed.device)
            #print("dtype:", embed.dtype)
            #print("shape", embed.shape)

            raise ValueError(f"Expected{self.number_of_branches} branches, got {embed.shape[0]}")
        
        #debug prints keep#
        #print("\n[TRACE t_traj ENTRY]")
    
        #if t_feat.shape[-1] != 16:

            #print("t_feat mismatch detected")
            #print("shape:" , t_feat.shape)

            #raise ValueError("invalid t_feat dimensionality")    

        self.x = torch.cat([embed.expand(self.number_of_branches , self.step_progression , self.state_dimension)] , dim=-1)

        #print("branch_embed:", self.branch_embed.shape)
        #print("embed:", embed.shape)
        #print("t_traj:", t_traj.shape)
        #print("x before MLP:", x.shape)
        
        #print(self.mlp1)
        #print("in_features:", self.mlp1.in_features)
        #print("out_features:", self.mlp1.out_features)
        #print("bias shape:", self.mlp1.bias.shape)
        
        #assert self.x.shape[-1] == self.mlp1.in_features , f"MLP mismatch: got {self.x.shape[-1]} , expected {self.mlp1.in_features}"

        #debug prints keep#
        #print("embed" , embed.shape)
        #print("x:" , x.shape)
        #print("mlp1 expected in:" , self.mlp1.in_features)

        x = F.relu(self.mlp1(self.x))

        self.x = self.mlp2(x)

        self.B = self.x.shape
        
        #assert self.x.shape[-1] == self.mlp1.in_features , f"MLP mismatch: got {self.x.shape[-1]} , expected {self.mlp1.in_features}"

        
        return self.B, self.x

    #===forward pass===#
    def forward(self , motion_type = None , update_rule = None, noise_floor = None):

        config = AnnealConfig()

        motion_type = motion_type

        update_rule = update_rule

        self.B = self.x.shape

        #print ("forward start")

        #coupling phase generation:
        #progress is a local trajectory coordinate derived from P_d reset events
        # representaion of evolution across trajectory steps, not experiment/ training progress
        #ownership - forward: defines models internal state space coupling, represented through sinusoidal projection.
        P_map = nn.Linear(3 , self.state_dimension)

        P_d = torch.zeros(3 , device = self.mlp1.weight.device)

        pd_step_trigger = P_d

        P_lat = P_map(P_d.to(device = self.mlp1.weight.device))

        #===anchor===#
        theta_d = torch.zeros(3)
        
        Theta_map = nn.Linear(3 , self.state_dimension)
        
        Theta_lat = Theta_map(theta_d.to(device = self.mlp1.weight.device))

        #print ("[TRACE] compute t_feat id:" , id(t_feat))
        #print("t_feat shape" , t_feat.shape)
        #print(type(T), T)
        #print(type(t_traj), t_traj)

        #===branch MLP output===#
        E = self._mlp_branch()

        #branch collapse VERY important
        
        self.branch_update= self.branch_embed.view(self.input_dimension , self.step_progression, self.state_dimension)

        print("x:", self.x.shape)
        print("B:", self.B)
        #print("branch_out:", self.branch_out.shape)

        self.branch_out = self.branch_update

        self.V_real = self.branch_out[: , : , 1 , :]

        self.V_imag =self.branch_out[: , : , 2 , :]

        self.Theta_real = self.branch_out[: , : , 3 , :]

        self.Theta_imag = self.branch_out[: , : , 4 , :]

        self.denom_real = self.Theta_real

        self.denom_imag = self.Theta_imag

        self.eps_effective = self.epsilon

        self.noise_floor = 0.0

        #===complex branch aggregation===#
        self.denom_mag2 = self.denom_real**2 + self.denom_imag**2 + self.eps_effective**2

        self.num_real = P_lat.unsqueeze(0).unsqueeze(1) + self.V_real

        self.num_imag = self.Theta_lat.unsqueeze(0).unsqueeze(1) + self.V_imag

        self.latent_real_in = self.num_real

        self.latent_imag_in = self.num_imag

        self.interaction_real = (self.latent_real_in * self.denom_real + self.latent_real_in * self.denom_imag) / self.denom_mag2

        self.interaction_imag = (self.latent_imag_in * self.denom_real - self.latent_real_in * self.denom_imag) / self.denom_mag2

        self.alphas_exp = self.alphas.view(self.input_dimension, self.step_progression , -1 , self.state_dimension)

        self.latent_real_out = (self.alphas_exp * self.interaction_real).sum(dim=0)

        self.latent_imag_out = (self.alphas_exp * self.interaction_imag).sum(dim=0)

        self.positional_prediction = self.pos_proj(self.latent_real_out)

        #===compute delta and cumulative sum===#
        #DERIVED - do not feed into prediction
        #Delta is generated from latent_real_out and represents
        #change/readout of the resulting state
        delta = self.delta_readout(self.latent_real_out, self.latent_imag_out)
                        
        delta = torch.zeros_like(delta, device = self.mlp1.weight.device)

        #Debug prints keep seperate from Diagnostic prints
        #print("alphas:", self.alphas.shape)
        #print("interaction real", self.interaction_real.shape)
        #print("alphas exp", alphas_exp.shape)
        #print("latent real out", self.latent_real_out.shape)
        #print("positional_prediction", positional_prediction.shape)

        #===compute uncertainty===#
        self.latent_imag_uncertainty = self.latent_imag_out.abs().mean(dim=1)
        self.latent_real_uncertainty = self.latent_real_out.abs().mean(dim=1)
        self.pos_proj_uncertainty = self.positional_prediction.abs().mean(dim=1)

        #also done in visualization, check to see if needs removed asap
        #unc = self.latent_uncertainty.detach().cpu().numpy()
        #mean_unc = unc.mean()
        #max_unc = unc.max()
        #threshold = mean_unc + unc.std()
        #outlier_idx = unc > threshold

        #===compute per-step branch variance===#
        self.branch_variance = ((self.interaction_real - self.latent_real_in.unsqueeze(0))**2 * self.alphas.view(self.input_dimension , self.step_progression , 1)).sum(dim=0)

        #~~~optional: scale down noise contribution~~~#
        noise_floor = noise_floor

        self.branch_variance = self.branch_variance * self.noise_floor

        #===aggregate to uncertainty===#
        #mean across state dim#
        self.branch_variance = self.branch_variance.mean(dim=1)

        #===branch pruning calculation===#
        #===Changing the threshold will over right automatic alpha weighting
        self.threshold = 0.0

        self.alphas = F.softmax(self.branch_logits , dim=0)

        self.alphas[self.alphas < self.threshold] = 0

        #renormalize
        self.alphas /= self.alphas.sum()

        self.alphas_uncertainty = self.alphas.abs().mean(dim=1)

        #===anchor first latent dimension to P_d===#
        #is this a divergence anchor and is it necessary?
        #LEGACY: injects P_d anchor into latent_real_out channel 0
        #VERIFY if this is required after the current state representation clean up?
        #latent_real_out[: , 0] = P_d[0]

        latent_imag_out = self.latent_image_out

        positional_prediction = self.positional_prediction

        branch_variance = self.branch_variance

        branch_update = self.branch_update

        #debug prints keep#
        #print("denom_real:" , self.denom_real.shape)

        #legacy integrated trajectory#
        #previously used by the residual update experiment#

        #pos_super = torch.cumsum(delta , dim=0) + P_d.to(device)

        #~~~Optional residual layers~~~#
        #if update_rule == "residual":

            #for layer in self.res_layers:

                #latent_real_out = latent_real_out + layer(latent_real_out)

                #pos_super = latent_real_out

        #assert self.branch_embed.shape[-1] == config.branch_embed_dimension

        return positional_prediction , branch_update