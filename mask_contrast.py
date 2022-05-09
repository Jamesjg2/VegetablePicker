import torch 
from torch import nn
import torchvision

class MaskContrastLoss(nn.Module):
    def __init__(self):
        super(MaskContrastLoss, self).__init__()
        
    def forward(self, main_image_embedding, positive_embedding, negative_embeddings, t = .5):
        '''
        Computes the contrastive loss over all pixel embeddings for an image

        Computed by taking the sum of all individual pixel losses with pixels taken from main image 
        params:
            main_image - the image embedding we are computing the contrastive loss for
            positive_embedding - image of same shape as main image. each pixel represents pixel embedding
            negative_embeddings - list of images of same shape as main images. each pixel in each image represents pixel embedding
            t - temperature used in contrastive loss formula. Set to .5 as in paper default settings
        '''

        mean_positive = torch.mean(positive_mean_embedding)
        mean_negatives = []

        for z in negative_embeddings:
            mean_negatives.append(torch.mean(z))

        mean_negatives = torch.stack(mean_negatives)
        numerator = torch.exp(main_image_embedding * mean_positive / t)
        
        denominator = 0
        
        for mean_negative in mean_negatives:
            denominator += torch.exp(main_image_embedding * mean_negative / t)

        loss = -torch.log(numerator / denominator)
        loss = torch.sum(loss)

        return loss