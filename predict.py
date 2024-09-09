import argparse
from estimator import Estimator

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--dataset_path', type=str, help='Enter absolute dataset path')
parser.add_argument('--model_path', type=str, help='Enter absolute model path')

args = parser.parse_args()

if __name__ == '__main__':
    new_estimator = Estimator(model_path=args.model_path, inference_mode=True)
    new_estimator.predict(feature_dataset_path=args.dataset_path, save_preds_to_csv=True)
