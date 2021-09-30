import re
import gzip
import traceback
import jsonlines
import emoji
import pandas as pd
import numpy as np
from collections import Counter

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from embeddings.WordVectorsManager import WordVectorsManager


def prepare_labels(y):
  """
  Labels to categories
  :param y:
  :return:
  """
  return labels_to_categories(y)  # y_cat, encoder_mapping


def prepare_text_inputs(TextX, split, pipeline):
  """
  Transform text input to features using sklearn pipeline and embeddings
  :param TextX:
  :param split:
  :param pipeline:
  :return:
  """
  if split:
    return pipeline.fit_transform(TextX[0]), pipeline.fit_transform(TextX[1])
  else:
    return pipeline.fit_transform(TextX[0])


def prepare_emojis_inputs(EmojisX, split, emojis_vocab="emojis/emojis_dump_latest.txt"):
  """
  Transform Emojis to one-hot matrix
  :param EmojisX:
  :param split:
  :param emojis_vocab:
  :return:
  """

  emojis = []
  with open(emojis_vocab, encoding="utf-8") as f:
    for line in f:
      emojis.append(line.rstrip())
  mlb = MultiLabelBinarizer(classes=emojis, sparse_output=False)

  onehot = mlb.fit(emojis)
  if split:
    return onehot.transform(EmojisX[0]), onehot.transform(EmojisX[1])
  else:
    return onehot.transform(EmojisX[0])


def onehot_to_categories(y):
  """
  Transform one-hot labels to category label ([0,0,0,1,0] - '2') for human readability.
  :param y:
  :return:
  """
  return np.asarray(y).argmax(axis=-1)


def get_class_weights2(y, smooth_factor=0):
  """
  Estimate class weights for unbalanced datasets.
  :param y: labels
  :param smooth_factor: max(counter.values()) * smooth_factor
  :return: {class: weight}
  """
  counter = Counter(y)

  if smooth_factor > 0:
    p = max(counter.values()) * smooth_factor
    for k in counter.keys():
      counter[k] += p

  majority = max(counter.values())

  return {cls: float(majority / count) for cls, count in counter.items()}


def labels_to_categories(y):
  """
  Label encoder. ("neutral" -> 1)
  :param y:
  :return: encoded label, encoder mapping
  """

  encoder = LabelEncoder()
  encoder.fit(y)
  encoder_mapping = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
  y_num = encoder.transform(y)
  return y_num, encoder_mapping


def categories_to_onehot(y):
  """
  Transform categories to one-hot ready for training.
  :param y:
  :return:
  """
  return to_categorical(y)


def validate_string(text):
  """
  Validate that sentiment is str
  :param text:
  :return: text or np.nan
  """
  if isinstance(text, str) and text.strip() != '':
    return text
  elif text == np.nan:
    return np.nan
  elif text == 'nan':
    print(f" '{text}' is invalid string")
  else:
    print(f" '{text}' is invalid string")
    return np.nan


def all_lower(string):
  """
  Lower test and remove indentation
  :param string:
  :return: string
  """
  if not isinstance(string, str):
    return string
  string = string.lower()
  string = re.sub("ά", "α", string)
  string = re.sub("ό", "ο", string)
  string = re.sub("έ", "ε", string)
  string = re.sub("ί", "ι", string)
  string = re.sub("ύ", "υ", string)
  string = re.sub("ή", "η", string)
  string = re.sub("ώ", "ω", string)
  string = re.sub("ϊ", "ι", string)
  string = re.sub("ΐ", "ι", string)
  string = re.sub("ϋ", "υ", string)
  return string


def read_jsonl(filename):
  """
  Read jsonlines data
  :param filename:
  :return: data
  """
  data = []
  try:
    with gzip.open(filename, 'rb') as fp:
      j_reader = jsonlines.Reader(fp)
      for obj in j_reader:
        data.append(obj)
  except OSError:
    with open(filename, 'rb') as fp:
      j_reader = jsonlines.Reader(fp)
      for obj in j_reader:
        data.append(obj)
  return data


def extract_emojis(text):
  """
  Extract emojis and occurrences from text
  :param text: text str
  :return: {':poop:', 3}
  """
  l1st = []
  try:
    emjs = emoji.emoji_lis(text)
    if emjs:
      for e in emjs:
        l1st.append(emoji.demojize(e.get('emoji')))
    c = Counter(l1st)
    return {x[0]: x[1] for x in c.most_common()}
  except TypeError:
    pass
    return {}
  except Exception:
    traceback.print_exc()
    return {}


def get_embeddings(corpus, dim):
  """
  Get embeddings for dataset
  :param corpus:
  :param dim:
  :return: emb_matrix, wv_map
  """
  vectors = WordVectorsManager(corpus, dim).read()
  vocab_size = len(vectors)
  print('Loaded %s word vectors.' % vocab_size)
  wv_map = {}
  pos = 0
  # +1 for zero padding token and +1 for unk
  emb_matrix = np.ndarray((vocab_size + 2, dim), dtype='float32')
  for i, (word, vector) in enumerate(vectors.items()):
    if len(vector) > 199:
      pos = i + 1
      wv_map[word] = pos
      emb_matrix[pos] = vector

  # add unknown token
  pos += 1
  wv_map["<unk>"] = pos
  emb_matrix[pos] = np.random.uniform(low=-0.05, high=0.05, size=dim)

  return emb_matrix, wv_map


def parse_input_file(filename, dedup=False, predict=False):
  """
  Read input raw dataset file
  :param filename:
  :param dedup:
  :param predict:
  :return: dataset
  """

  print(f"Loading data, deduplication: {dedup} ...")
  if re.search(r'\.csv\b', filename):
    df = pd.read_csv(filename, encoding='utf-8')
  else:
    df = pd.DataFrame(read_jsonl(filename))

  # missing_labels = []
  df['text'] = df['text'].apply(lambda x: validate_string(x))
  df = df.dropna(subset=['text'])
  df.drop_duplicates(subset=['id'], inplace=True)
  if 'emojis' not in df.columns:
    df['emojis'] = df['text'].apply(lambda x: extract_emojis(x))
  if predict:
    df = df[['id', 'text', 'emojis']]
  else:
    if 'channel' in df.columns:
      df = df[df['channel'].isin(['facebook', 'instagram', 'twitter', 'youtube'])]
    if dedup:
      df.drop_duplicates(subset=['sentiment', 'text'], inplace=True)
    df = df[['sentiment', 'text', 'emojis']]
  # missing_labels = list({'negative', 'positive', 'neutral'} - set(df.sentiment.unique().tolist()))
  #
  # data = df.values.tolist()
  # if missing_labels:c
  #   for label in missing_labels:
  #     data.append([label, 'dummy', 1, 0, {}, []])

  return df


def load_data(filename, pipeline, split=False, predict=False, dedup=False, test_size=0.1):
  """
  Load data
  :param filename:
  :param pipeline:
  :param split:
  :param predict:
  :param dedup:
  :param test_size:
  :return:
  """
  if predict and dedup:
    raise ValueError("Error: When predict==True then dedup==False")
  df = parse_input_file(filename, dedup, predict)
  if df.empty:
    return None
  if predict:
    split = False
    encoder_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}  # hardcoded as set before
    TextX = prepare_text_inputs([df['text']], split=split, pipeline=pipeline)
    EmojisX = prepare_emojis_inputs([df['emojis']], split=split)
    sid = df['id'].values.tolist()
    return TextX, EmojisX, sid, encoder_mapping
  else:
    if split:
      class_weights = get_class_weights2(labels_to_categories(df['sentiment'])[0], smooth_factor=0)
      trainTextX, testTextX, trainEmojisX, testEmojisX = train_test_split(
        df[['text', 'sentiment']], df[['emojis', 'sentiment']], test_size=test_size, random_state=101
      )
      trainY, encoder_mapping = labels_to_categories(trainTextX["sentiment"])
      trainY = categories_to_onehot(trainY)
      testY, encoder_mapping = labels_to_categories(testTextX["sentiment"])
      testY = categories_to_onehot(testY)
      trainTextX, testTextX = prepare_text_inputs([trainTextX['text'], testTextX['text']], split=split, pipeline=pipeline)
      trainEmojisX, testEmojisX = prepare_emojis_inputs([trainEmojisX['emojis'], testEmojisX['emojis']], split=True)
      return trainTextX, trainEmojisX, trainY, testTextX, testEmojisX, testY, class_weights, encoder_mapping
    else:
      TextX = prepare_text_inputs([df['text']], split=split, pipeline=pipeline)
      EmojisX = prepare_emojis_inputs([df['emojis']], split=split)
      Y, encoder_mapping = labels_to_categories(df["sentiment"])
      Y = categories_to_onehot(Y)
      return TextX, EmojisX, Y, encoder_mapping
