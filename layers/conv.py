import enum
from tensors import Tensor, concat
from .module import BaseModule
import numpy as np


class Conv1D(BaseModule):

    def __init__(self, kernel_size: int, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.kernel_vector = Tensor(np.random.rand(output_channels, input_channels, kernel_size))

    def forward(self, x):
        out_c, in_c, kern = self.kernel_vector.shape
        channels, seq =  x.shape
        out = []
        for c in range(out_c):
            kernel = self.kernel_vector[c, :, :]
            c_out = []
            for i in range(seq - kern + 1):
                patch = x[:, i: i+kern]
                value = (patch * kernel).sum()
                c_out.append(value.reshape((1, 1)))
            c_out = concat(c_out, axis = 1)

            out.append(c_out)
        conv_out = concat(out, axis=0)
        return conv_out

    


class Conv2D(BaseModule):
    def __init__(self, kernel_size: int, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.kernel_vector = Tensor(np.random.rand(output_channels, input_channels, kernel_size, kernel_size))

    def forward(self, x: Tensor)-> Tensor:
        cols, H_out, W_out = self.im2col(x)
        out_c, in_c, ker_h, ker_w = self.kernel_vector.shape
        conv_out = cols @ self.kernel_vector.reshape((in_c * ker_h * ker_w, out_c))
        return conv_out.T.reshape((out_c, H_out, W_out))
    
    def im2col(self, x:Tensor)-> tuple[Tensor, int, int]:
        patch_matrix = []
        _, _, ker_row, ker_col = self.kernel_vector.data.shape
        _, x_row, x_col = x.data.shape
        H_out = x_row - ker_row + 1
        W_out = x_col - ker_col + 1
        for row in range(H_out):
            for col in range(W_out):
                patch_matrix.append(x[:, row: row + ker_row, col: col + ker_col].reshape((1, -1)))

        return concat(patch_matrix, 0), H_out, W_out
    

    def col2im(self, x):
        pass
