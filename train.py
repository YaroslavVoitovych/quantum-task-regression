import argparse
from estimator import Estimator

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--dataset_path', type=str, help='Enter absolute dataset path')
parser.add_argument('--test_size', type=float, help='Enter test size from 0 to 1 for validation')


args = parser.parse_args()

if __name__ == '__main__':
    new_estimator = Estimator()
    new_estimator.fit(dataset_path=args.dataset_path, test_size=float(args.test_size))
    new_estimator.save_model()

