import os
import sys
import warnings
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import configparser as cp
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, \
  classification_report, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from utilities.preprocessor import PreProcessor
from utilities.embed_extractor import EmbeddingsExtractor
from utilities.data_loader import get_embeddings, load_data, onehot_to_categories
from models.models import polus_te, polus_to


class Classifier:

  def __init__(self, model_name, config_file, epochs=30, batch_size=16):
    self.epochs = epochs
    self.batch_size = batch_size

    config = cp.ConfigParser()
    config.read(config_file)

    self.embedding_dim = config.getint(model_name, 'embeddings', fallback=300)
    self.max_length = config.getint(model_name, 'max_length', fallback=50)
    self.use_lsh = config.getboolean(model_name, 'use_lsh', fallback=False)
    self.sim_threshold = config.getfloat(model_name, 'sim_threshold', fallback=1)
    self.max_query_id = config.getint(model_name, 'max_query_id', fallback=3000)
    self.emoji_count = config.getint(model_name, 'emoji_count', fallback=1018)
    self.hashtag_count = config.getint(model_name, 'hashtag_count', fallback=0)
    self.corpus = config.get(model_name, 'corpus', fallback='fasttext')
    self.model_name = model_name

    self.embeddings, self.word_indices = get_embeddings(corpus=self.corpus, dim=self.embedding_dim)

    if self.model_name == 'polus_to':
      self.model = polus_to(
        self.embeddings, self.max_length, config)
    elif self.model_name == 'polus_te':
      self.model = polus_te(
        self.embeddings, self.max_length, config)
    else:
      raise NotImplementedError(
        'Unrecognized model ' + self.model_name + '. It should be one of [\'polus_to\', \'polus_te\']')

    self.pipeline = Pipeline([('preprocess', PreProcessor(TextPreProcessor(
      normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
      include_tags={'hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored'},
      fix_html=True,
      segmenter="twitter",
      corrector="twitter",
      unpack_hashtags=True,  # original hashtags are not included in embeddings dictionary, so unpack
      unpack_contractions=True,
      spell_correct_elong=False,
      tokenizer=SocialTokenizer(lowercase=True).tokenize,
      dicts=[emoticons]))), ('ext', EmbeddingsExtractor(word_indices=self.word_indices,
                                                        max_lengths=self.max_length,
                                                        add_tokens=True,
                                                        unk_policy="random"))])

  def train(self, datafile, dedup, out_file):
    """
    Train model
    :param datafile:
    :param dedup:
    :param out_file:
    :return:
    """

    print(self.model.summary())

    # text, sentiment, encoder_mapping
    data = load_data(datafile, self.pipeline, split=True, dedup=dedup, test_size=0.2)
    trainTextX, trainEmojisX, trainY, valTextX, valEmojisX, testY, class_weights, encoder_mapping = data
    print("Train on size:{}".format(len(trainY)))

    if self.model_name == 'polus_to':
      trainX = [trainTextX]
      valX = [valTextX]
    elif self.model_name == 'polus_te':
      trainX = [trainTextX, trainEmojisX]
      valX = [valTextX, valEmojisX]
    else:
      trainX = None
      valX = None
      print("No model name: ", self.model_name)

    callbacks = []
    checkpoint = ModelCheckpoint(out_file+'_min_val_loss.h5', save_best_only=True, monitor='val_loss', mode='min')
    callbacks.append(checkpoint)
    checkpoint = ModelCheckpoint(out_file+'_max_val_acc.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    callbacks.append(checkpoint)

    history = self.model.fit(
      x=trainX, y=trainY,
      validation_data=(valX, testY),
      class_weight=class_weights,
      epochs=self.epochs,
      batch_size=self.batch_size,
      callbacks=callbacks
    )

    hist_df = pd.DataFrame(history.history, columns=[['epochs', 'loss', 'accuracy', 'val_loss', 'val_accuracy']])
    with open(f"model_history/{out_file.rsplit('/',1)[1].rsplit('.', 1)[0]}.csv", mode='w') as f:
      hist_df.to_csv(f)

  def retrain(self, datafile, dedup, in_file, out_file):
    """
    Train model
    :param datafile:
    :param dedup:
    :param in_file:
    :param out_file:
    :return:
    """

    print(self.model.summary())

    # text, sentiment, encoder_mapping
    data = load_data(datafile, self.pipeline, split=True, dedup=dedup, test_size=0.2)
    trainTextX, trainEmojisX, trainY, valTextX, valEmojisX, testY, class_weights, encoder_mapping = data
    print("Re-Train on size:{}".format(len(trainY)))

    if self.model_name == 'polus_to':
      trainX = [trainTextX]
      valX = [valTextX]
    elif self.model_name == 'polus_te':
      trainX = [trainTextX, trainEmojisX]
      valX = [valTextX, valEmojisX]
    else:
      trainX = None
      valX = None
      print("No model name: ", self.model_name)

    callbacks = []
    checkpoint = ModelCheckpoint(out_file+'_min_val_loss.h5', save_best_only=True, monitor='val_loss', mode='min')
    callbacks.append(checkpoint)
    checkpoint = ModelCheckpoint(out_file+'_max_val_acc.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    callbacks.append(checkpoint)

    print(f"Loading model ...")
    loaded_model = load_model(in_file)

    history = loaded_model.fit(
      x=trainX, y=trainY,
      validation_data=(valX, testY),
      class_weight=class_weights,
      epochs=self.epochs,
      batch_size=self.batch_size,
      callbacks=callbacks
    )

    hist_df = pd.DataFrame(history.history)
    with open(f"model_history/{in_file.rsplit('/',1)[1].rsplit('.', 1)[0]}.csv", mode='w') as f:
      hist_df.to_csv(f)

  def test(self, datafile, infile):
    """
    Test Model
    :param datafile:
    :param infile:
    :return:
    """

    # text, sentiment, encoder_mapping
    data = load_data(datafile, self.pipeline, dedup=True)
    TextX, EmojisX, y, encoder_mapping = data

    if self.model_name == 'polus_to':
      X = [TextX]
    elif self.model_name == 'polus_te':
      X = [TextX, EmojisX]
    else:
      X = None
      print("No model name: ", self.model_name)

    encoded_labels = onehot_to_categories(y)
    encoded_classes = set(list(encoded_labels))
    print(f"Encoded Labels:{encoded_classes}")
    print(f"Labels:{set([encoder_mapping[l] for l in encoded_labels])}")
    labels = encoded_labels

    test_loss = []
    test_accs = []

    print(f"Loading model ...")
    model_test = load_model(infile)
    print(f"Testing ...")
    score = model_test.evaluate(X, y, verbose=1, )

    test_loss.append(score[0])
    test_accs.append(score[1])

    # Predicting the Test set results
    y_pred = model_test.predict(X)

    with open("pred_prob_distr.pickle", 'wb') as save:
      pickle.dump(y_pred, save)

    predictions = tf.argmax(y_pred, axis=1)

    # Creating the Confusion Matrix (non-norm)
    cm = confusion_matrix(labels, predictions)

    # # Classification Report
    cr = classification_report(labels, predictions, target_names=encoder_mapping.values())

    # Scores
    macro_precision = precision_score(labels, predictions, average='macro')
    micro_precision = precision_score(labels, predictions, average='micro')
    weighted_precision = precision_score(labels, predictions, average='weighted')
    macro_recall = recall_score(labels, predictions, average='macro')
    micro_recall = recall_score(labels, predictions, average='micro')
    weighted_recall = recall_score(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions, )
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    one_vs_all_labels = label_binarize(labels, classes=[0, 1, 2])
    one_vs_all_predictions = label_binarize(predictions, classes=[0, 1, 2])
    roc_auc = roc_auc_score(one_vs_all_labels, one_vs_all_predictions)
    auprc = average_precision_score(one_vs_all_labels, one_vs_all_predictions)

    print('')
    print('Testing samples: %i' % len(X[0]))
    print('')
    print('Metrics')
    print('-' * 40)
    print('F1 Macro: {0:0.2f}%'.format(100 * f1_macro))
    print('ROC AUC: {0:0.2f}%'.format(100 * roc_auc))
    print('AUPRC: {0:0.2f}%'.format(100 * auprc))
    print('Accuracy: {0:0.2f}%'.format(100 * acc))
    print('-' * 40)
    print('F1 Micro: {0:0.2f}%'.format(100 * f1_micro))
    print('F1 Weighted: : {0:0.2f}%'.format(100 * f1_weighted))
    print('Macro Precision: {0:0.2f}%'.format(100 * macro_precision))
    print('Micro Precision: {0:0.2f}%'.format(100 * micro_precision))
    print('Weighted Precision: {0:0.2f}%'.format(100 * weighted_precision))
    print('Macro Recall: {0:0.2f}%'.format(100 * macro_recall))
    print('Micro Recall: {0:0.2f}%'.format(100 * micro_recall))
    print('Weighted Recall: {0:0.2f}%'.format(100 * weighted_recall))
    print('\n')
    print('Confusion Matrix')
    print('-' * 20)
    print(cm)
    print('\n')
    print('Sums Matrix')
    print('-' * 20)
    print(cm.sum(axis=1))
    print('\n')
    print('Normalized Confusion Matrix')
    print('-' * 40)
    print(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    print('\n')
    print('Classification Report')
    print('-' * 60)
    print(cr)

    return labels, cm, cr, acc, f1_macro, f1_micro, f1_weighted, roc_auc, auprc, macro_precision, micro_precision, weighted_precision, macro_recall, micro_recall, weighted_recall

  def predict(self, filename, infile):
    """
    Predict on new data from dataset
    :param filename:
    :param infile:
    :return:
    """

    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    sys.stderr = stderr
    warnings.filterwarnings("ignore")

    data = load_data(filename, self.pipeline, predict=True)
    if not data:
      print("Nothing to predict !")
      return [[]]
    TextX, EmojisX, sid, encoder_mapping = data

    if self.model_name == 'polus_to':
      X = [TextX]
    elif self.model_name == 'polus_te':
      X = [TextX, EmojisX]
    else:
      X = None
      print("No model name: ", self.model_name)

    model_test = load_model(infile)

    # Predicting results
    y_pred = model_test.predict(X)
    predictions = tf.argmax(y_pred, axis=1)

    res = []
    for i in range(len(sid)):
      res.append([
        str(sid[i]),
        encoder_mapping[predictions[i].numpy()],
      ])

    df = pd.DataFrame(data=res, columns=['id', 'predicted_sentiment'])
    df.to_csv(filename+'.res.csv', encoding='utf-8', index=False)
    print(f"Predictions saved at {filename+'.res.csv'}")

    return res
