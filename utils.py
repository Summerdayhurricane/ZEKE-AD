import random
import numpy as np
import torch


def setup_seed(seed):
    """Fix random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(pred, threshold=0.35):
    """Normalize anomaly map before binarization."""
    if pred.max() <= 0.45:
        pred = np.where(pred < threshold, pred, (pred - pred.min()) / (0.5 - pred.min()))
    else:
        pred = np.where(pred < threshold, pred, (pred - pred.min()) / (pred.max() - pred.min()))
    return pred


def create_gaussian_kernel_3x3(sigma=1.0):
    """Create a 3x3 Gaussian kernel used for patch-token smoothing."""
    x = torch.arange(-1, 2, dtype=torch.float32)
    y = torch.arange(-1, 2, dtype=torch.float32)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    gaussian_kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))
    gaussian_kernel /= gaussian_kernel.sum()
    return gaussian_kernel.to(dtype=torch.float16)


def compute_statistics(values):
    """Compute mean, variance, and standard deviation."""
    arr = np.array(values, dtype=np.float64)
    return {
        'mean': float(np.mean(arr)),
        'var': float(np.var(arr)),
        'std': float(np.std(arr))
    }
