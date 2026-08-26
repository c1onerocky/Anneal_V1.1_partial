import torch

class AnnealConfig:
    def __init__(self , time_feature_dimension = 11 , hidden_dimension = 55, imag_dimension = 3 ,  embed_dimension = 3 , state_dimension = 3, branch_embed_dimension = 55, input_dimension = 55):

        self.embed_dimension = embed_dimension

        self.input_dimension = input_dimension

        self.branch_embed_dimension = branch_embed_dimension

        self.time_feature_dimension = time_feature_dimension

        self.hidden_dimension = hidden_dimension

        self.state_dimension = state_dimension

        self.imag_dimension = imag_dimension