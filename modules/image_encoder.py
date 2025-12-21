from torch.nn import Module, Conv2d
from torch import Tensor


class ImageEncoder(Module):
    """
    Implementation for the module that patches the image and converts each patch into a token
    """

    def __init__(self, input_dim: int, patch_size: int, num_ch: int, output_dim: int):
        """
        Constructor
        :param input_dim: Size of the input image. The image is ASSUMED to be a square (w==h)
        :param patch_size: Size of the patches to use, again, assumed to be squares
        :param num_ch: Input channels
        :param output_dim: Embedding size
        """

        super().__init__()
        assert(input_dim % patch_size == 0)

        # The patch + conversion in embedding is done by using Conv2D
        # With number of output channels == desired embedding dimension

        self.patch_encoder = Conv2d(num_ch, output_dim, patch_size, stride=patch_size)
        self.input_dim = input_dim
        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function
        :param x: Input image tensor (batch_size, channels, height, width)
        :return: Token sequences (batch_size, seq_len, output_dim)
        """

        # Extract each patch and encode it
        # Assumes image in input is bsize x num_ch x w x h

        out = self.patch_encoder(x)

        # Reshape to obtain bsize x seq_len (W x H) x emb_dim

        bsize, emb_dim, w, h = out.shape
        out = out.view(bsize, emb_dim, (w * h))
        out = out.transpose(-1, -2)

        return out