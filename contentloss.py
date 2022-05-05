import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

class ContentLoss(nn.Module):
    def __init__(self) -> None:
        super(ContentLoss, self).__init__()
        feature_model_extractor_node = "features.35"
        feature_model_normalize_mean = [0.485, 0.456, 0.406]
        feature_model_normalize_std = [0.229, 0.224, 0.225]
        
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_node = feature_model_extractor_node
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(True)
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data. This is the VGG model preprocessing method of the ImageNet dataset.
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: torch.Tensor, hr_tensor: torch.Tensor) -> torch.Tensor:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        hr_tensor = self.normalize(hr_tensor)

        sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
        hr_feature = self.feature_extractor(hr_tensor)[self.feature_model_extractor_node]

        # Find the feature map difference between the two images
        content_loss = F.mse_loss(sr_feature, hr_feature)

        return content_loss