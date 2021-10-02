import os, sys
import argparse as ap
from classifier import Classifier
import datetime
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

parser = ap.ArgumentParser(description='Polus')

parser.add_argument('-d', '--data', type=str, help='data file')
parser.add_argument('-m', '--model', type=str, help='model (either [\'palo_text\',\'polus_te\',)')
parser.add_argument('-c', '--config', type=str, help='configuration file containing model parameters')
parser.add_argument('-f', '--file', help='Model file to save when training or to load when testing')
parser.add_argument('-i', '--infile', help='Model file to load when re-training')
parser.add_argument('-ddp', '--dedup', help='Dedup data during loading', action='store_true')

subparsers = parser.add_subparsers(help='Mode', dest='mode')

train = subparsers.add_parser('train', help='Train a model')
train.add_argument('--epochs', type=int, help='Number of epochs', default=30)
train.add_argument('--batch', type=int, help='Batch Size', default=16)
train.add_argument('-ddp', '--dedup', help='Dedup data during loading', action='store_true')

retrain = subparsers.add_parser('retrain', help='Re-train a model')
retrain.add_argument('--epochs', type=int, help='Number of epochs', default=30)
retrain.add_argument('--batch', type=int, help='Batch Size', default=16)
retrain.add_argument('-ddp', '--dedup', help='Dedup data during loading', action='store_true')

test = subparsers.add_parser('test', help='Test a model')
test.add_argument('--fig', type=str, help='Output confusion matrix to an image')

predict = subparsers.add_parser('predict', help='Predict sentiment')

# Parse input arguments
args = parser.parse_args()
args.file = 'model_registry/' + args.file

print('*' * 88)
print(f"All args: {args}")
print('*' * 88)
print(f"Mode: {args.mode}")

logger.warning("For data deduplication use arg: -ddp")
logger.warning("If mode == 'test' then ddp is TRUE")

if args.mode == 'train':
  c = Classifier(model_name=args.model, config_file=args.config, epochs=args.epochs, batch_size=args.batch)
  c.train(args.data, args.dedup, args.file)

elif args.mode == 'retrain':
  c = Classifier(model_name=args.model, config_file=args.config, epochs=args.epochs, batch_size=args.batch)
  c.retrain(args.data, args.dedup, args.infile, args.file)

elif args.mode == 'test':
  c = Classifier(model_name=args.model, config_file=args.config)
  labels, cm, cr, acc, f1_macro, f1_micro, f1_weighted, roc_auc, auprc, macro_precision, micro_precision, \
  weighted_precision, macro_recall, micro_recall, weighted_recall = c.test(
    args.data, args.file)

  with open(
    f'{args.file.replace("model_registry/", "")}_{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")}_scores.txt', 'w'
  ) as f:
    f.write('ROC AUC: {0:0.2f}%\n'.format(100 * roc_auc))
    f.write('AUPRC: {0:0.2f}%\n'.format(100 * auprc))
    f.write('Accuracy: {0:0.2f}%\n'.format(100 * acc))
    f.write('F1 Macro: {0:0.2f}%\n'.format(100 * f1_macro))
    f.write('F1 Micro: {0:0.2f}%\n'.format(100 * f1_micro))
    f.write('F1 Weighted: : {0:0.2f}%\n'.format(100 * f1_weighted))
    f.write('Macro Precision: {0:0.2f}%\n'.format(100 * macro_precision))
    f.write('Macro Recall: {0:0.2f}%\n'.format(100 * macro_recall))
    f.write('Micro Precision: {0:0.2f}%\n'.format(100 * micro_precision))
    f.write('Micro Recall: {0:0.2f}%\n'.format(100 * micro_recall))
    f.write('Weighted Precision: {0:0.2f}%\n'.format(100 * weighted_precision))
    f.write('Weighted Recall: {0:0.2f}%\n'.format(100 * weighted_recall))

elif args.mode == 'predict':
  c = Classifier(model_name=args.model, config_file=args.config)
  res = c.predict(args.data, args.file)
  if res:
    print('... done!')
else:
  parser.print_usage()
