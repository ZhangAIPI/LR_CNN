import numpy as np
import torch
import torch.nn.functional as F
import setting

device = setting.device


def get_indices(X_shape, HF, WF, stride, pad):
    """
        Returns index matrices in order to transform our input image into a matrix.
        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.
        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d.
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    # get input size
    m, n_C, n_H, n_W = X_shape

    # get output size
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1

    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = torch.arange(HF, device=device).repeat_interleave(WF)
    # Duplicate for the other channels.
    level1 = level1.repeat(n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * torch.arange(out_h, device=device).repeat_interleave(out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----

    # Slide 1 vector.
    slide1 = torch.arange(WF, device=device).repeat(HF)
    # Duplicate for the other channels.
    slide1 = slide1.repeat(n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * torch.arange(out_w, device=device).repeat(out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = torch.arange(n_C, device=device).repeat_interleave(HF * WF).reshape(-1, 1)

    return i, j, d


def im2col(X, HF, WF, stride, pad):
    """
        Transforms our input image into a matrix.
        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.
        Returns:
        -cols: output matrix.
    """
    # Padding
    X_padded = F.pad(X, [0, 0, 0, 0, pad, pad, pad, pad], mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    # Multi-dimensional arrays indexing.
    cols = X_padded[:, d, i, j]
    cols = torch.cat(tuple(cols), dim=-1)
    return cols


def col2im(dX_col, X_shape, HF, WF, stride, pad):
    """
        Transform our matrix back to the input image.
        Parameters:
        - dX_col: matrix with error.
        - X_shape: input image shape.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.
        Returns:
        -x_padded: input image with error.
    """
    # Get input size
    N, D, H, W = X_shape
    # Add padding if needed.
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = torch.zeros((N, D, H_padded, W_padded), device=device)

    # Index matrices, necessary to transform our input image into a matrix.
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    # Retrieve batch dimension by spliting dX_col N times: (X, Y) => (N, X, Y)
    m, n = dX_col.shape
    dX_col_reshaped = torch.cat(torch.split(dX_col, N, dim=1)).reshape((N, m, n // N))
    # Reshape our matrix back to image.
    # slice(None) is used to produce the [::] effect which means "for every elements".
    X_padded[slice(None), d, i, j] += dX_col_reshaped
    # Remove padding from new image if needed.
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[pad:-pad, pad:-pad, :, :]
