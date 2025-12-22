from torch.nn import Module, Sequential, Linear, GELU, LayerNorm, Dropout
from torch import Tensor

from custom_modules.attention_module import MHSAModule


class TransformerBlock(Module):
    """
    Pre-norm Transformer Block implementation
    """

    def __init__(self, input_dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 attn_drop: float = 0.0,
                 output_drop: float = 0.0,
                 mlp_drop: float = 0.0,):
        """
        Constructor
        :param input_dim: Input (and output) dimension
        :param num_heads: Number of heads for the multi head attention layer
        :param mlp_ratio: The multiplication factor of the internal MLP
        :param attn_drop: Dropout drop probability for the MHSA
        :param output_drop: Output drop probability for the MHSA
        :param mlp_drop: Dropout drop probability for the internal MLP
        """

        super().__init__()
        self.mlp = Sequential(Linear(input_dim, int(input_dim * mlp_ratio)),
                              GELU(),
                              Dropout(mlp_drop),
                              Linear(int(input_dim * mlp_ratio), input_dim),
                              Dropout(mlp_drop),)
        self.mhsa = MHSAModule(input_dim, num_heads, attn_drop, output_drop)
        self.norm1 = LayerNorm(input_dim)
        self.norm2 = LayerNorm(input_dim)

    def forward(self, x: Tensor) -> dict:
        """
        Forward function
        :param x: Input tensor (batch_size, seq_len, input_dim)
        :return: Output tensor (batch_size, seq_len, input_dim)
        """

        normed = self.norm1(x)
        mhsa_dict = self.mhsa(normed)
        mhsa_out = mhsa_dict['out']
        cls_att = mhsa_dict['cls_att']


        # First skip

        mhsa_out += x

        normed = self.norm2(mhsa_out)
        mlp_out = self.mlp(normed)

        # Second skip

        output = mlp_out + mhsa_out

        return {'out': output, 'cls_att': cls_att}