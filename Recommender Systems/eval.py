import torch
import numpy as np

def evaluate_auc(model, test_data):
    """
    Evaluate the model using the AUC metric

    Parameters:
        model: The trained recommendation model.
        test_data: A list of tuples (user, pos_item, neg_item).
                    For each user, negative_items is a list of item indicies
                    We can use a sample negative instead of full of negative
        Returns:
        auc: The avarage AUC score across all test users
    """
    model.eval()
    aucs = []

    with torch.no_grad():
        for user, pos_item, neg_items in test_data:
            # Combine negative items with positive items
            items = neg_items + [pos_item]
            # Create a tensor
            user_tensor = torch.tensor([user] * len(items), dtype=torch.long, device=model.device)
            items_tensor = torch.tensor(items, dtype=torch.long, device=model.device)

            # Get predicted scores for all candidate items
            scores = model.forward(user_tensor, items_tensor)
            scores = scores.cpu().numpy()

            # The score for the positive item is the last element
            pos_score = scores[-1]
            
            # Count how many negatives have a lower score than the positive item
            count = sum(1 for neg_score in scores[:-1] if pos_score > neg_score)
            # Compute AUC for this user as the fraction of negatives ranked below the positive
            auc = count / len(neg_items)
            aucs.append(auc)
    # Return the average AUC across all test users
    return np.mean(aucs)
