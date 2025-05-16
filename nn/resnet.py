import torch
import torch.nn as nn

class ResNet(nn.Module):
  def __init__(self, block: nn.Module, num_blocks: list[int], num_classes: int) -> None:
    super().__init__()
    self.in_channels = 64
    self.conv = nn.Conv2d(
      in_channels=3,
      out_channels=64,
      kernel_size=3,
      stride=1,
      padding=1,
      bias=False
    )
    self.bn = nn.BatchNorm2d(num_features=64)
    self.relu = nn.ReLU(inplace=True)

    self.layer1 = self.build_resnet_layer(block=block, num_blocks=num_blocks[0], out_channels=64, stride=1)
    self.layer2 = self.build_resnet_layer(block=block, num_blocks=num_blocks[1], out_channels=128, stride=2)
    self.layer3 = self.build_resnet_layer(block=block, num_blocks=num_blocks[2], out_channels=256, stride=2)
    self.layer4 = self.build_resnet_layer(block=block, num_blocks=num_blocks[3], out_channels=512, stride=2)

    self.avgpool = nn.AvgPool2d(kernel_size=4)

    self.fc = nn.Linear(in_features=512*block.expansion, out_features=num_classes, bias=True)

  def build_resnet_layer(self, block: nn.Module, num_blocks: int, out_channels: int, stride: int) -> nn.Sequential:
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(in_channels=self.in_channels, out_channels=out_channels, stride=stride))
      self.in_channels = out_channels * block.expansion
    return nn.Sequential(*layers)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    out = self.conv(input)
    out = self.bn(out)
    out = self.relu(out)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.avgpool(out)
    out = out.view(out.size(0), -1)
    out = self.fc(out)
    return out
