import os

import torchvision

def main():
  cifar10_destination = os.path.join("data", "datasets")
  if not (os.path.exists(cifar10_destination) and os.path.isdir(cifar10_destination)):
    os.makedirs(cifar10_destination, exist_ok=True)
  torchvision.datasets.CIFAR10(root=cifar10_destination, train=True, download=True)
  torchvision.datasets.CIFAR10(root=cifar10_destination, train=False, download=True)
  print("Finished downloading CIFAR10 dataset!")

if __name__ == "__main__":
  main()
