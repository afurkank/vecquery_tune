import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineDistanceLoss(nn.Module):
    def forward(self, emb1, emb2, mask1, mask2):
        # implement the forward pass
        # emb1 = (batch_size x embed_dim)
        # emb2 = (batch_size x embed_dim)
        # mask1 = (batch_size)
        # mask2 = (batch_size)
        # mask the embeddings that are padded
        mask1 = mask1.float()
        mask2 = mask2.float()
        emb1 = emb1 * mask1.unsqueeze(-1)
        emb2 = emb2 * mask2.unsqueeze(-1)
        # normalize the embeddings
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        # calculate the cosine similarity between the embeddings
        cosine_similarities = F.cosine_similarity(emb1, emb2)
        # calculate the loss
        loss = torch.mean(1 - cosine_similarities)
        return loss

class L2DistanceLoss(nn.Module):
    def forward(self, emb1, emb2, mask1, mask2):
        # implement the forward pass
        # emb1 = (batch_size x embed_dim)
        # emb2 = (batch_size x embed_dim)
        # mask1 = (batch_size)
        # mask2 = (batch_size)
        # mask the embeddings that are padded
        mask1 = mask1.float()
        mask2 = mask2.float()
        emb1 = emb1 * mask1.unsqueeze(-1)
        emb2 = emb2 * mask2.unsqueeze(-1)
        # calculate the L2 distance between the embeddings
        l2_distances = torch.norm(emb1 - emb2, p=2, dim=1)
        # calculate the loss
        loss = torch.mean(l2_distances)
        return loss