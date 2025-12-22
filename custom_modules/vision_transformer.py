from torch import Tensor
from torch.nn import Module, ModuleList, Linear, Softmax
from torch.nn.parameter import Parameter
import torch

from custom_modules.image_encoder import ImageEncoder
from custom_modules.transformer_block import TransformerBlock

class VisionTransformer(Module):
    """
    Vision transformer implementation.
    """

    def __init__(self,
                 input_dim: int,
                 patch_size: int,
                 num_channels: int,
                 num_classes: int,
                 emb_dim: int,
                 num_heads: int,
                 n_layers: int,
                 mlp_ratio: int = 4,
                 attn_drop: float = 0.0,
                 output_drop: float = 0.0,
                 mlp_drop: float = 0.0):
        """
        Constructor
        :param input_dim: Image input size, assumed to be a square (w==h)
        :param patch_size: Patch size, assumed to be a square
        :param num_channels: Input channels
        :param num_classes: Number of output classes
        :param emb_dim: Internal embedding dimension
        :param num_heads: Number of heads in each MHSA layer
        :param n_layers: Number of transformer blocks
        :param mlp_ratio: Multiplication factor of the MLP inside the transformer blocks
        :param attn_drop: Attention dropout in each MHSA
        :param output_drop: Output dropout of each MHSA
        :param mlp_drop: Dropout of each mlp in the transformer blocks
        """

        super(VisionTransformer, self).__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.output_drop = output_drop
        self.mlp_drop = mlp_drop
        self.tokens_per_image = int((self.input_dim / self.patch_size)**2) + 1

        self.positional_embs = Parameter(torch.randn(self.tokens_per_image, self.emb_dim), requires_grad=True)
        self.cls_token = Parameter(torch.randn(1, 1, self.emb_dim,), requires_grad=True)

        self.encoder = ImageEncoder(self.input_dim, self.patch_size, self.num_channels, self.emb_dim)
        self.t_blocks = ModuleList(
            [TransformerBlock(self.emb_dim,
                              self.num_heads,
                              self.mlp_ratio,
                              self.attn_drop,
                              self.output_drop,
                              self.mlp_drop) for _ in range(self.n_layers)]
        )
        self.classification_head = Linear(self.emb_dim, self.num_classes)

    def forward(self, x: Tensor) -> dict:
        """
        Forward function
        :param x: Batches of images (batch_size, channels, height, width)
        :return: Classification logits (batch_size, num_classes)
        """

        bsize = x.shape[0]

        # Encode images

        encoded = self.encoder(x)

        # Add CLS token

        copied_cls = self.cls_token.repeat(bsize, 1, 1)
        encoded = torch.cat((copied_cls, encoded), dim=1)

        # Add positional embeddings

        encoded += self.positional_embs

        # Pass through transformer blocks

        activations = encoded

        out_dict = {}
        for i, t_block in enumerate(self.t_blocks):
            activations_dict = t_block(activations)
            out_dict[f'cls_att_tblock_{i}'] = activations_dict['cls_att']
            activations = activations_dict['out']

        # Extract CLS tokens

        cls_tokens = activations[:, 0, :]

        # Pass them through the classifier

        logits = self.classification_head(cls_tokens)
        out_dict['scores'] = logits

        return out_dict
