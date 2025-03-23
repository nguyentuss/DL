import torch
import torch.nn.functional as F

class CosineKNN:
    """
    Item-based k-Nearest Neighbors (kNN) recommender using cosine similarity.

    This model recommends items to users based on the similarity between items,
    computed via cosine similarity. It supports implicit feedback and is
    optimized for GPU acceleration using PyTorch.

    Parameters:
        k (int): Number of nearest neighbors to consider for scoring.
        device (torch.device): The device (CPU or CUDA) for all computations.

    Attributes:
        k (int): Number of nearest neighbors used in recommendations.
        device (torch.device): Device used for tensor computations.
        interaction_matrix (torch.Tensor): User-item interaction matrix on the specified device.
        item_norm (torch.Tensor): L2-normalized item vectors (used for similarity computation).
        similarity (torch.Tensor): Cosine similarity matrix between items.
    """
    def __init__(self, k = 10, device = None):
        self.k = k
        self.device = device

        self.to(device)
    def fit(self, interaction_matrix):
        """
        Fit the model to the user-item interaction matrix

        Args:
            interaction_matrix (torch.Tensor): A binary matrix (num_users x num_items)
                                               representing implicit feedback.
        """
        self.interaction_matrix = interaction_matrix.to(self.device)
        item_vectors = self.interaction_matrix.T.float()
        self.item_norm = F.normalize(item_vectors, p=2, dim=1)
        self.similarity = torch.matmul(self.item_norm, self.item_norm.T)

