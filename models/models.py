from tensorflow.keras.layers import Bidirectional, concatenate, Input
from tensorflow.keras.layers import Dropout, Dense, LSTM, Embedding, GaussianNoise
from tensorflow.keras.layers import MaxPooling1D, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn import preprocessing

from utilities.layers import Attention


def embeddings_layer(max_length, embeddings, trainable=False, masking=False, scale=False, normalize=False):
  """
  Embeddings Layer
  :param max_length:
  :param embeddings:
  :param trainable:
  :param masking:
  :param scale:
  :param normalize:
  :return:
  """

  if scale:
    print("Scaling embedding weights...")
    embeddings = preprocessing.scale(embeddings)
  if normalize:
    print("Normalizing embedding weights...")
    embeddings = preprocessing.normalize(embeddings)

  vocab_size = embeddings.shape[0]
  embedding_size = embeddings.shape[1]

  _embedding = Embedding(
    input_dim=vocab_size,
    output_dim=embedding_size,
    input_length=max_length if max_length > 0 else None,
    trainable=trainable,
    # mask_zero=masking if max_length > 0 else False,
    weights=[embeddings]
  )
  return _embedding


def palo_text(emb, max_length, hparams):
  """
  Palo model only text input
  :param emb:
  :param max_length:
  :param hparams:
  :return:
  """

  classes = 3
  noise = hparams.getfloat('palo_text', 'noise', fallback=0.0)
  dropout_words = hparams.getfloat('palo_text', 'dropout_words', fallback=0.3)
  dropout_rnn = hparams.getfloat('palo_text', 'dropout_rnn', fallback=0.3)
  dropout_rnn_U = hparams.getfloat('palo_text', 'dropout_rnn_U', fallback=0.3)
  dropout_dense = hparams.getfloat('palo_text', 'dropout_dense', fallback=0.3)
  cells = hparams.getint('palo_text', 'cells', fallback=150)
  l2_reg = hparams.getfloat('palo_text', 'l2_reg', fallback=0.01)
  loss_l2 = hparams.getfloat('palo_text', 'loss_l2', fallback=0.0001)
  dropout_attention = hparams.getfloat('palo_text', 'dropout_attention', fallback=0.5)
  lr = hparams.getfloat('palo_text', 'lr', fallback=0.001)
  clipnorm = hparams.getint('palo_text', 'clipnorm', fallback=1)
  loss_fn = hparams.getint('palo_text', 'loss', fallback='binary_crossentropy')

  inp = Input(shape=(max_length,), name='text_input')
  x = embeddings_layer(max_length=max_length, embeddings=emb,
                       trainable=False, masking=True, scale=False,
                       normalize=False)(inp)
  x = GaussianNoise(noise)(x)
  x = (Dropout(dropout_words))(x)
  x = Bidirectional(LSTM(cells, return_sequences=True, dropout=dropout_rnn_U, kernel_regularizer=l2(l2_reg)))(x)
  x = Dropout(dropout_rnn)(x)
  x = Bidirectional(LSTM(cells, return_sequences=True, dropout=dropout_rnn_U, kernel_regularizer=l2(l2_reg)))(x)
  x = Dropout(dropout_rnn)(x)
  x = Conv1D(64, kernel_size=5, padding="valid", kernel_initializer="he_uniform")(x)
  x = MaxPooling1D(5)(x)
  x = Attention()(x)
  x = Dropout(dropout_attention)(x)

  dense = Dense(1024, activity_regularizer=l2(loss_l2), activation='relu')(x)
  dense = Dropout(dropout_dense)(dense)
  dense = Dense(128, activity_regularizer=l2(loss_l2), activation='relu')(dense)
  dense = Dropout(dropout_dense)(dense)
  outp = Dense(classes, activation='softmax')(dense)

  model = Model(inputs=inp, outputs=outp)
  adam = Adam(clipnorm=clipnorm, lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(optimizer=adam, loss=loss_fn, metrics=['accuracy'])
  return model


def palo_ti(emb, max_length, hparams):
  """
  Palo model text and emojis
  :param emb:
  :param max_length:
  :param hparams:
  :return:
  """

  classes = 3
  noise = hparams.getfloat('palo_ti', 'noise', fallback=0.0)
  dropout_words = hparams.getfloat('palo_ti', 'dropout_words', fallback=0.3)
  dropout_rnn = hparams.getfloat('palo_ti', 'dropout_rnn', fallback=0.3)
  dropout_rnn_U = hparams.getfloat('palo_ti', 'dropout_rnn_U', fallback=0.3)
  dropout_dense = hparams.getfloat('palo_ti', 'dropout_dense', fallback=0.3)
  cells = hparams.getint('palo_ti', 'cells', fallback=150)
  l2_reg = hparams.getfloat('palo_ti', 'l2_reg', fallback=0.01)
  loss_l2 = hparams.getfloat('palo_ti', 'loss_l2', fallback=0.0001)
  dropout_attention = hparams.getfloat('palo_ti', 'dropout_attention', fallback=0.5)
  lr = hparams.getfloat('palo_ti', 'lr', fallback=0.001)
  emoji_count = hparams.getint('palo_ti', 'emoji_count', fallback=1018)
  clipnorm = hparams.getint('palo_ti', 'clipnorm', fallback=1)
  loss_fn = hparams.getint('palo_ti', 'loss', fallback='binary_crossentropy')

  inp1 = Input(shape=(max_length,), name='text_input')
  inp2 = Input(shape=(emoji_count,), name='emoji_input')
  x = embeddings_layer(max_length=max_length, embeddings=emb,
                       trainable=False, masking=True, scale=False,
                       normalize=False)(inp1)
  x = GaussianNoise(noise)(x)
  x = (Dropout(dropout_words))(x)
  x = Bidirectional(LSTM(cells, return_sequences=True, dropout=dropout_rnn_U, kernel_regularizer=l2(l2_reg)))(x)
  x = Dropout(dropout_rnn)(x)
  x = Bidirectional(LSTM(cells, return_sequences=True, dropout=dropout_rnn_U, kernel_regularizer=l2(l2_reg)))(x)
  x = Dropout(dropout_rnn)(x)
  x = Conv1D(64, kernel_size=5, padding="valid", kernel_initializer="he_uniform")(x)
  x = MaxPooling1D(5)(x)
  x = Attention()(x)
  x = Dropout(dropout_attention)(x)

  conc = concatenate([x, inp2])
  dense = Dense(1024, activity_regularizer=l2(loss_l2), activation='relu')(conc)
  dense = Dropout(dropout_dense)(dense)
  dense = Dense(128, activity_regularizer=l2(loss_l2), activation='relu')(dense)
  dense = Dropout(dropout_dense)(dense)
  outp = Dense(classes, activation='softmax')(dense)

  model = Model(inputs=[inp1, inp2], outputs=outp)
  adam = Adam(clipnorm=clipnorm, lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(optimizer=adam, loss=loss_fn, metrics=['accuracy'])
  return model


