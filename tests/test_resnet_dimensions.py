from nn import (
  ResNet18,
  ResNet34,
  ResNet50,
  ResNet101,
  ResNet152
)

import torch

def test_dimensions(model: torch.nn.Module) -> bool:
  input = torch.randn(1, 3, 32, 32)
  try:
    model(input)
  except:
    return False
  return True

def main() -> bool:
  test_cases = {
    "DIMENSION ALIGNMENT": test_dimensions
  }
  resnets = {
    "18": ResNet18(num_classes=10),
    "34": ResNet34(num_classes=10),
    "50": ResNet50(num_classes=10),
    "101": ResNet101(num_classes=10),
    "152": ResNet152(num_classes=10)
  }
  results = []
  print("Testing ResNet implementations...")
  print("DETAILED REPORT")
  print("---------------")
  for test_case_name, test_case_func in test_cases.items():
    print(f"Problem: {test_case_name}")
    print("Feedback")
    for model_name, model in resnets.items():
      test_case_status = test_case_func(model)
      if test_case_status:
        print(f"Dimensions of ResNet{model_name} are aligned")
      else:
        print(f"\033[91mDimensions of ResNet{model_name} are misaligned\033[0m")
      results.append(test_case_status)
  if all(results):
    print("All five ResNet implementations are dimension aligned!")
  return all(results)

if __name__ == "__main__":
  main()
