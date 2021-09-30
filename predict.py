import argparse as ap
import warnings
import sys
import json
from classifier import Classifier

warnings.filterwarnings("ignore")

parser = ap.ArgumentParser(description='Polus')

parser.add_argument('-d', '--data', type=str, help='data file')
parser.add_argument('-m', '--model', type=str, help='model (either [\'cnn\',\'cnn_new\',\'papei\',\'palo\'])')
parser.add_argument('-c', '--config', type=str, help='configuration file containing model parameters')
parser.add_argument('-f', '--file', help='Model file to save when training or to load when testing')
parser.add_argument('-v', '--verbose', help='Print prediction results', dest='verbose', action='store_true')

# Parse input arguments
args = parser.parse_args()
# print(f"Mode: Predict") ###

c = Classifier(model_name=args.model, config_file=args.config)
res = c.predict(args.data, args.file)
if args.verbose:
  for pred in res:
    try:
      print(json.dumps({"id": str(pred[0]), "channel": pred[0].split('_')[1], "sentiment": str(pred[1]), "proba": 0.75}, sort_keys=True))
    except IndexError:
      print(json.dumps({"id": str(pred[0]), "channel": "unknown", "sentiment": str(pred[1]), "proba": 0.75}, sort_keys=True))
    sys.stdout.flush()
