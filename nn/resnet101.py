from .blocks import BottleneckBlock
from .resnet import ResNet

class ResNet101(ResNet):
  def __init__(self, num_classes: int) -> None:
    super().__init__(block=BottleneckBlock, num_blocks=[3, 4, 23, 3], num_classes=num_classes)
