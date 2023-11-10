from torchvision.models._api import WeightsEnum
from torch import nn
import torchvision
import torch


# def get_state_dict(self, *args, **kwargs):
#     kwargs.pop("check_hash")
#     return torch.hub.load_state_dict_from_url(self.url, *args, **kwargs)
# WeightsEnum.get_state_dict = get_state_dict


def create_effinetb2_model(num_classes:int=101,
                           seed:int=1126):
    torch.manual_seed(seed)

    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights)
    
    ## freeze parameters for extracters
    for parameter in model.parameters():
        parameter.requires_grad = False

    ## redefine clf
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes))

    effnet2_transforms = weights.transforms()

    return model, effnet2_transforms


def create_vit_model(num_classes:int=101, 
                     seed:int=1126):
    torch.manual_seed(seed)

    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.vit_b_16(weights=weights)

    # Freeze all layers in model
    for param in model.parameters():
        param.requires_grad = False

    ## redefine clf
    model.heads = nn.Sequential(nn.Linear(in_features=768, 
                                          out_features=num_classes))
    
    return model, transforms


class TinyVGG(nn.Module):

  def __init__(self, input_shape: int, hidden_units: int, output_shape: int = 101) -> None:
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=3, 
                    stride=1, 
                    padding=0),
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2) 
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
      )

  def forward(self, x: torch.Tensor):
      # x = self.conv_block_1(x)
      # x = self.conv_block_2(x)
      # x = self.classifier(x)
      # return x
      return self.classifier(self.conv_block_2(self.conv_block_1(x)))
