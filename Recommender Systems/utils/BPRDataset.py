import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd

# Example DataFrame structure: columns 'user_id', 'item_id', 'rating'
# df = pd.read_csv("your_data.csv")  # Assume df is already defined

class BPRDataset(Dataset):
    def __init__(self, df, num_items, min_rating=1.0):
        """
        Initializes the BPRDataset.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing 'user_id', 'item_id', and 'rating'.
            num_items (int): Total number of items.
            min_rating (float): Minimum rating to consider an interaction positive.
        """
        self.df = df
        self.num_items = num_items
        self.min_rating = min_rating
        
        # Build a dictionary mapping user to a set of items they liked (rating >= min_rating)
        self.user_positive = {}
        for row in df.itertuples():
            if row.rating >= self.min_rating:
                self.user_positive.setdefault(row.user_id, set()).add(row.item_id)
        
        # Filter DataFrame to include only positive interactions
        self.positive_df = df[df['rating'] >= self.min_rating].reset_index(drop=True)

    def __len__(self):
        return len(self.positive_df)
    
    def __getitem__(self, idx):
        """
        For each positive interaction, sample a negative item that the user did not interact with.
        For each positive interaction, you just need one negative one.
        Over the course of many training iterations, different negatives are sampled, which effectively exposes the model to a wide variety of negatives without the computational cost of generating every single pair.
        Returns:
            user (torch.LongTensor): User index.
            pos_item (torch.LongTensor): Positive item index.
            neg_item (torch.LongTensor): Negative item index.
        """
        row = self.positive_df.iloc[idx]
        user = int(row['user_id'])
        pos_item = int(row['item_id'])
        
        # Sample a negative item that is not in the user's positive set
        while True:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in self.user_positive.get(user, set()):
                break
        
        return (torch.tensor(user, dtype=torch.long),
                torch.tensor(pos_item, dtype=torch.long),
                torch.tensor(neg_item, dtype=torch.long))

class BPREvalDataset(Dataset):
    def __init__(self, df, num_items, num_negatives, min_rating=1.0):
        self.df = df[df['rating'] >= min_rating].reset_index(drop=True)
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # Build user positive mapping
        self.user_positive = {}
        for row in df.itertuples():
            if row.rating >= min_rating:
                self.user_positive.setdefault(row.user_id, set()).add(row.item_id)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user = int(row['user_id'])
        pos_item = int(row['item_id'])
        negatives = []
        while len(negatives) < self.num_negatives:
            neg = np.random.randint(0, self.num_items)
            if neg not in self.user_positive.get(user, set()):
                negatives.append(neg)
        return (torch.tensor(user, dtype=torch.long), torch.tensor(pos_item, dtype=torch.long), torch.tensor(negatives, dtype=torch.long))
