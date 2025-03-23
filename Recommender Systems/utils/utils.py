import numpy as np
import torch

def transform2d_ml(users, items, ratings):
    """
    Transform the data tuples (users, items, ratings) in 2-dimension user-items and ratings value

    Parameters:
        users, items, ratings
    
    Returns:
        Matrix (torch.tensor): dtype.float
    """
    num_users = max([x for x in users])
    num_items = max([x for x in items])

    comb = (users, items, ratings)

    X = np.zeros((num_users, num_items))

    for user, item, rating in comb:
        X[user][item] = rating
    X = torch.tensor(X, dtype=torch.float)
    return X