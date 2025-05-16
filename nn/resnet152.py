from .blocks import BottleneckBlock
from .resnet import ResNet

class ResNet152(ResNet):
  def __init__(self, num_classes: int) -> None:
    super().__init__(block=BottleneckBlock, num_blocks=[3, 8, 36, 3], num_classes=num_classes)
