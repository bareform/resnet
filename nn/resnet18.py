from .blocks import BasicBlock
from .resnet import ResNet

class ResNet18(ResNet):
  def __init__(self, num_classes: int) -> None:
    super().__init__(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=num_classes)
