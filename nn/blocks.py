import torch
import torch.nn as nn

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_channels: int, out_channels: int, stride: int=1) -> None:
    super().__init__()
    self.conv1 = nn.Conv2d(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=3,
      stride=stride,
      padding=1,
      bias=False
    )
    self.bn1 = nn.BatchNorm2d(num_features=out_channels)
    self.relu1 = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(
      in_channels=out_channels,
      out_channels=out_channels,
      kernel_size=3,
      stride=1,
      padding=1,
      bias=False
    )
    self.bn2 = nn.BatchNorm2d(num_features=out_channels)
    self.relu2 = nn.ReLU(inplace=True)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != self.expansion*out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(
          in_channels=in_channels,
          out_channels=self.expansion*out_channels,
          kernel_size=1,
          stride=stride,
          bias=False
        ),
        nn.BatchNorm2d(num_features=out_channels)
      )

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    out = self.conv1(input)
    out = self.bn1(out)
    out = self.relu1(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out += self.shortcut(input)
    out = self.relu2(out)
    return out

class BottleneckBlock(nn.Module):
  expansion = 4

  def __init__(self, in_channels: int, out_channels: int, stride: int=1) -> None:
    super().__init__()
    self.conv1 = nn.Conv2d(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=1,
      stride=1,
      padding=0,
      bias=False
    )
    self.bn1 = nn.BatchNorm2d(num_features=out_channels)
    self.relu1 = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(
      in_channels=out_channels,
      out_channels=out_channels,
      kernel_size=3,
      stride=stride,
      padding=1,
      bias=False
    )
    self.bn2 = nn.BatchNorm2d(num_features=out_channels)
    self.relu2 = nn.ReLU(inplace=True)

    self.conv3 = nn.Conv2d(
      in_channels=out_channels,
      out_channels=self.expansion*out_channels,
      kernel_size=1,
      stride=1,
      padding=0,
      bias=False
    )
    self.bn3 = nn.BatchNorm2d(num_features=self.expansion*out_channels)
    self.relu3 = nn.ReLU(inplace=True)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != self.expansion*out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(
          in_channels=in_channels,
          out_channels=self.expansion*out_channels,
          kernel_size=1,
          stride=stride,
          bias=False
        ),
        nn.BatchNorm2d(num_features=self.expansion*out_channels)
      )

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    out = self.conv1(input)
    out = self.bn1(out)
    out = self.relu1(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu2(out)
    out = self.conv3(out)
    out = self.bn3(out)
    out += self.shortcut(input)
    out = self.relu3(out)
    return out
