from .test_resnet_dimensions import (
  main as test_resnet_dimensions_main
)

def main() -> None:
  runner = {
    "test_resnet_dimensions": test_resnet_dimensions_main,
  }
  results = []
  for test_case_file_main_fn in runner.values():
    test_case_file_main_result = test_case_file_main_fn()
    results.append(test_case_file_main_result)
    print()
  if all(results):
    print("Passed all test cases!")
  else:
    print("FAILED")
    print("Failed the following test cases:")
    for test_case_idx, test_case_file in enumerate(runner.keys()):
      if not results[test_case_idx]:
        print(f"  - {test_case_file}")

if __name__ == "__main__":
  main()
