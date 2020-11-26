import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

# import pytorch_fft.fft.autograd as afft

class CompactBilinearPooling(nn.Module):
    """
    Compute compact bilinear pooling over two bottom inputs.

    Args:

        output_dim: output dimension for compact bilinear pooling.

        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.

        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.

        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.

        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.

        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.
    """

    def __init__(self, input_dim1, input_dim2, output_dim,
                 sum_pool=True,
                 rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None):
        super(CompactBilinearPooling, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool
        # 产生对应的 s 和 h
        if rand_h_1 is None:
            np.random.seed(1)
            rand_h_1 = np.random.randint(output_dim, size=self.input_dim1)
        if rand_s_1 is None:
            np.random.seed(3)
            rand_s_1 = 2 * np.random.randint(2, size=self.input_dim1) - 1

        self.sketch_matrix1 = nn.Parameter(self.generate_sketch_matrix(rand_h_1, rand_s_1, self.output_dim), requires_grad=False)
        # self.register_buffer('sketch_matrix1', self.generate_sketch_matrix(
        #     rand_h_1, rand_s_1, self.output_dim))
        # self.sparse_sketch_matrix1 = self.generate_sketch_matrix(rand_h_1, rand_s_1, self.output_dim)

        if rand_h_2 is None:
            np.random.seed(5)
            rand_h_2 = np.random.randint(output_dim, size=self.input_dim2)
        if rand_s_2 is None:
            np.random.seed(7)
            rand_s_2 = 2 * np.random.randint(2, size=self.input_dim2) - 1

        # self.register_buffer('sketch_matrix2', self.generate_sketch_matrix(
        #     rand_h_2, rand_s_2, self.output_dim))
        self.sketch_matrix2 = nn.Parameter(self.generate_sketch_matrix(rand_h_2, rand_s_2, self.output_dim), requires_grad=False)
        # self.sparse_sketch_matrix2 = self.generate_sketch_matrix(rand_h_2, rand_s_2, self.output_dim)

        # if cuda:
        #     self.sparse_sketch_matrix1 = self.sparse_sketch_matrix1.cuda()
        #     self.sparse_sketch_matrix2 = self.sparse_sketch_matrix2.cuda()

    def forward(self, bottom1, bottom2):
        """
        bottom1: 1st input, 4D Tensor of shape [batch_size, input_dim1, height, width].
        bottom2: 2nd input, 4D Tensor of shape [batch_size, input_dim2, height, width].
        """
        #TODO 这里统一用的 rfft 和 irfft： https://gist.github.com/vadimkantorov/d9b56f9b85f1f4ce59ffecf893a1581a
        assert bottom1.size(1) == self.input_dim1 and \
            bottom2.size(1) == self.input_dim2

        batch_size, _, height, width = bottom1.size()
        # 确保 channel 进行一个变换, 一定要转换存储的顺序是指连续存储
        bottom1_flat = bottom1.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim1)
        bottom2_flat = bottom2.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim2)
        # 在 TF 的实现中，利用的是左乘转置的乘积转置，做测试sketch_matrix
        sketch_1 = bottom1_flat.mm(self.sketch_matrix1)
        sketch_2 = bottom2_flat.mm(self.sketch_matrix2)

        # fft1_real, fft1_imag = afft.Fft()(sketch_1, Variable(torch.zeros(sketch_1.size())).cuda())
        # fft2_real, fft2_imag = afft.Fft()(sketch_2, Variable(torch.zeros(sketch_2.size())).cuda())
        # sketch_1 实部， sketch1 是一个二维， 第一个元素为 real， 第二个为 image. stack 给后面的维度增加1
        fft1_inp = torch.stack((sketch_1, torch.zeros_like(sketch_1)), dim=-1)
        fft2_inp = torch.stack((sketch_2, torch.zeros_like(sketch_2)), dim=-1)
        # fft1_inp2 = torch.cat((
        #     torch.unsqueeze(sketch_1, dim=-1),
        #     torch.unsqueeze(torch.zeros_like(sketch_1), dim=-1)
        # ), dim=-1)
        fft1_compx = torch.rfft(fft1_inp, signal_ndim=2)
        fft2_compx = torch.rfft(fft2_inp, signal_ndim=2)
        fft1_real, fft1_imag = fft1_compx[..., 0], fft1_compx[..., 1]
        fft2_real, fft2_imag = fft2_compx[..., 0], fft2_compx[..., 1]

        # inner_dot = torch.view_as_complex(fft1_compx) * torch.view_as_complex(fft2_compx)
        fft_product_real = fft1_real.mul(fft2_real) - fft1_imag.mul(fft2_imag)
        fft_product_imag = fft1_real.mul(fft2_imag) + fft1_imag.mul(fft2_real)

        # cbp_flat = ifft(fft_product_real, fft_product_imag)[0]
        # 似乎是一个 real
        cbp_flat = torch.irfft(torch.stack((fft_product_real, fft_product_imag), dim=-1), signal_ndim=2)[..., 0]
        cbp = cbp_flat.view(batch_size, height, width, self.output_dim)

        if self.sum_pool:
            cbp = cbp.sum(dim=1).sum(dim=1)

        return cbp

    @staticmethod
    def generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
            rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert(rand_h.ndim == 1 and rand_s.ndim ==
               1 and len(rand_h) == len(rand_s))
        assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        rand_s = torch.from_numpy(rand_s)
        sparse_sketch_matrix = torch.sparse.FloatTensor(
            indices.t(), rand_s, torch.Size([input_dim, output_dim]))
        return sparse_sketch_matrix.to_dense()


class CompactBilinearPooling2(torch.nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, sum_pool = True):
        super().__init__()
        self.out_channels = out_channels
        self.sum_pool = sum_pool
        generate_tensor_sketch = lambda rand_h, rand_s, in_channels, out_channels: torch.sparse.FloatTensor(torch.stack([torch.arange(in_features), rand_h]), rand_s, [in_channels, out_channels]).to_dense()
        self.tenosr_sketch1 = torch.nn.Parameter(generate_tensor_sketch(torch.randint(out_channels, size = (in_channels1,)), 2 * torch.randint(2, size = (in_channels1,), dtype = torch.float32) - 1, in_channels1, out_channels), requires_grad = False)
        self.tensor_sketch2 = torch.nn.Parameter(generate_tensor_sketch(torch.randint(out_channels, size = (in_channels2,)), 2 * torch.randint(2, size = (in_channels2,), dtype = torch.float32) - 1, in_channels2, out_channels), requires_grad = False)

    def forward(self, x1, x2):
        fft1 = torch.rfft(x1.permute(0, 2, 3, 1).matmul(self.tensor_sketch1), signal_ndim = 1)
        fft2 = torch.rfft(x2.permute(0, 2, 3, 1).matmul(self.tensor_sketch2), signal_ndim = 1)
        # torch.rfft does not support yet torch.complex64 outputs, so we do complex product manually
        fft_complex_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim = -1)
        cbp = torch.irfft(fft_complex_product, signal_ndim = 1, signal_sizes = (self.out_channels, )) * self.out_channels
        return cbp.sum(dim = [1, 2]) if self.sum_pool else cbp.permute(0, 3, 1, 2)

if __name__ == '__main__':
    # torch.cuda.set_device('cuda:2')*
    bottom1 = Variable(torch.randn(128, 512, 14, 14))
    bottom2 = Variable(torch.randn(128, 512, 14, 14))

    layer = CompactBilinearPooling(512, 512, 8000)
    # layer.cuda()
    layer.train()

    out = layer(bottom1, bottom2)
    g = 5
