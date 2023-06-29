import torch
import transformers
import transformers.models.llama.modeling_llama
from einops import rearrange
import random




def replace_llama_rope_with_ntk_aware_scaled_rope():
    old_init = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__
    def ntk_scaled_init(self, dim, max_position_embeddings=2048, base=10000, device=None):

        #The method is just these three lines
        max_position_embeddings = 16384
        a = 8 #Alpha value
        base = base * a ** (dim / (dim-2)) #Base change formula

        old_init(self, dim, max_position_embeddings, base, device) 

    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__ = ntk_scaled_init