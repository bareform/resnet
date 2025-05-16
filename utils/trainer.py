from nn import (
  ResNet18,
  ResNet34,
  ResNet50,
  ResNet101,
  ResNet152
)

import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def get_argparser():
  parser = argparse.ArgumentParser(prog="trainer",
                      description="training loop for ResNet")
  parser.add_argument("--root", type=str, default=os.path.join("data", "datasets"))
  parser.add_argument("--transform", action="store_true", default=False)
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--num_workers", type=int, default=1)
  parser.add_argument("--pin_memory", action="store_true", default=True)
  parser.add_argument("--num_epochs", type=int, default=10)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--optimizer", type=str, default="AdamW",
                      choices=["AdamW"])
  parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR",
                      choices=["CosineAnnealingLR"])
  parser.add_argument("--t_max", type=int, default=10)
  parser.add_argument("--model_name", type=str, default="ResNet18",
                      choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"])
  parser.add_argument("--random_seed", type=int, default=0)
  parser.add_argument("--compile", action="store_true", default=False)
  parser.add_argument("--ckpt_dir", type=str, default=os.path.join("checkpoints"))
  return parser

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
  model.train()
  running_loss = 0.0
  correct_preds = 0
  total_preds = 0
  with tqdm(dataloader, desc="Training", unit="batch") as pbar:
    for images, labels in dataloader:
      images = images.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      _, predicted = torch.max(outputs, dim=1)
      total_preds += labels.size(0)
      correct_preds += (predicted == labels).sum().item()
      running_loss += loss.item()

      pbar.update(1)
      pbar.set_postfix({
        "loss": f"{loss.item():.2f}",
        "accuracy": f"{correct_preds / total_preds:.2f}"
      })
    scheduler.step()
  loss = running_loss / len(dataloader)
  acc = correct_preds / total_preds
  return loss, acc

def test_epoch(model, dataloader, criterion, device):
  model.eval()
  running_loss = 0.0
  correct_preds = 0
  total_preds = 0
  with torch.no_grad():
    with tqdm(dataloader, desc="Testing", unit="batch") as pbar:
      for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, dim=1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()
        running_loss += loss.item()

        pbar.update(1)
        pbar.set_postfix({
          "loss": f"{loss.item():.2f}",
          "accuracy": f"{correct_preds / total_preds:.2f}"
        })
  loss = running_loss / len(dataloader)
  acc = correct_preds / total_preds
  return loss, acc

def main():
  model_mapping = {
    "ResNet18": ResNet18(num_classes=10),
    "ResNet34": ResNet34(num_classes=10),
    "ResNet50": ResNet50(num_classes=10),
    "ResNet101": ResNet101(num_classes=10),
    "ResNet152": ResNet152(num_classes=10)
  }
  args = get_argparser().parse_args()

  torch.manual_seed(args.random_seed)
  np.random.seed(args.random_seed)
  random.seed(args.random_seed)

  if not (os.path.exists(args.ckpt_dir) and os.path.isdir(args.ckpt_dir)):
    os.makedirs(args.ckpt_dir, exist_ok=True)

  if args.transform:
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
  else:
    transform = transforms.Compose([
      transforms.ToTensor()
    ])

  train_set = torchvision.datasets.CIFAR10(root=args.root, train=True, transform=transform)
  train_loader = data.DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=args.pin_memory
  )
  test_set = torchvision.datasets.CIFAR10(root=args.root, train=False, transform=transform)
  test_loader = data.DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=args.pin_memory
  )

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")

  model = model_mapping.get(args.model_name, None)
  model = model.to(device)
  print(f"Training: {args.model_name}")
  if model is None:
    raise RuntimeError("something went wrong: model was not set!")
  if args.compile:
    print("Compiling with Just-In-Time compilation")
    model = torch.compile(model)
  criterion = torch.nn.CrossEntropyLoss()
  if args.optimizer == "AdamW":
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
  else:
    raise RuntimeError("something went wrong: optimizer was not set!")
  if args.lr_scheduler == "CosineAnnealingLR":
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)
  else:
    raise RuntimeError("something went wrong: learning rate scheduler was not set!")
  best_test_accuracy = 0.0
  for epoch in range(args.num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
    test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
    print(f"Epoch: {epoch + 1}/{args.num_epochs}")
    print(f"Train Loss: {train_loss:.2f}, Train Accuracy: {train_acc:.2f}")
    print(f"Test Loss: {test_loss:.2f}, Test Accuracy: {test_acc:.2f}")
    if best_test_accuracy < test_acc:
      best_test_accuracy = test_acc
      print("Saving new best model")
      torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"best_cifar10_{args.model_name}.pth"))

if __name__ == "__main__":
  main()
