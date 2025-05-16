from .blocks import BasicBlock
from .resnet import ResNet

class ResNet34(ResNet):
  def __init__(self, num_classes: int) -> None:
    super().__init__(block=BasicBlock, num_blocks=[3, 4, 6, 3], num_classes=num_classes)
