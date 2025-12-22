import math
import torch
from torch.nn import Module, Linear, Dropout
from torch.nn.functional import softmax
from torch import Tensor

class MHSAModule(Module):
    """
    Multi-Head Self-Attention Module.
    """

    def __init__(self, input_dim: int, num_heads: int, attn_drop: float = 0.0, output_drop: float = 0.0):
        """
        Constructor.
        :param input_dim: Input (and output) dimension of the module.
        :param num_heads: Number of attention heads.
        :param attn_drop: Dropout probability for the attention layer just after the softmax.
        :param output_drop: Dropout probability for the output layer.
        """

        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        assert self.input_dim % self.num_heads == 0
        self.head_dim = input_dim // self.num_heads
        self.q_proj = Linear(input_dim, input_dim)
        self.k_proj = Linear(input_dim, input_dim)
        self.v_proj = Linear(input_dim, input_dim)
        self.output_layer = Linear(input_dim, input_dim)
        self.attn_drop = Dropout(attn_drop)
        self.output_drop = Dropout(output_drop)

    def forward(self, x: Tensor) -> dict:
        """
        Forward function.
        :param x: The input tensor (batch_size, seq_len, input_dim)
        :return: Dictionary with the output tensor (batch_size, seq_len, input_dim) and the attention matrix wrt the cls token
        """

        # Projection

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Divide in heads (bsize x num_heads x seq_len x head_dimension)
        # First adding the num_heads dimension, then we swap them to end the tensor in (... seq_len x head_dim)

        q_heads_tensor = torch.reshape(q, (q.shape[0], q.shape[1], self.num_heads, self.head_dim)).transpose(1, 2)
        k_heads_tensor = torch.reshape(k, (k.shape[0], k.shape[1], self.num_heads, self.head_dim)).transpose(1, 2)
        v_heads_tensor = torch.reshape(v, (v.shape[0], v.shape[1], self.num_heads, self.head_dim)).transpose(1, 2)

        # Compute attention matrix

        att_mat = q_heads_tensor @ k_heads_tensor.transpose(-2, -1)

        # Softmax and rescale

        rescaled_att_mat = att_mat / math.sqrt(self.head_dim)
        soft_att_mat = softmax(rescaled_att_mat, dim=-1)

        # Take only first row of the attention matrix from column 1 (to exclude the cls x cls activation)

        cls_att_mat = soft_att_mat[:, :, 0, 1:]
        grid_size = int(math.sqrt(cls_att_mat.shape[-1]))
        cls_att_mat = torch.reshape(cls_att_mat, (q.shape[0], self.num_heads, grid_size, grid_size))

        soft_att_mat = self.attn_drop(soft_att_mat)

        # Multiply by values

        output = soft_att_mat @ v_heads_tensor

        # Remove head division (final output bsize x seq_len x (num_heads*heads_dim = emb_dim))

        batch, num_heads, seq, dim = output.shape
        output = output.transpose(1, 2)
        output = output.reshape(batch, seq, num_heads * dim)

        # Final output + dropout

        output = self.output_layer(output)
        output = self.output_drop(output)

        return {'out': output, 'cls_att': cls_att_mat}