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


def polus_to(emb, max_length, hparams):
  """
  Palo model only text input
  :param emb:
  :param max_length:
  :param hparams:
  :return:
  """

  classes = 3
  noise = hparams.getfloat('polus_to', 'noise', fallback=0.0)
  dropout_words = hparams.getfloat('polus_to', 'dropout_words', fallback=0.3)
  dropout_rnn = hparams.getfloat('polus_to', 'dropout_rnn', fallback=0.3)
  dropout_rnn_U = hparams.getfloat('polus_to', 'dropout_rnn_U', fallback=0.3)
  dropout_dense = hparams.getfloat('polus_to', 'dropout_dense', fallback=0.3)
  cells = hparams.getint('polus_to', 'cells', fallback=150)
  l2_reg = hparams.getfloat('polus_to', 'l2_reg', fallback=0.01)
  loss_l2 = hparams.getfloat('polus_to', 'loss_l2', fallback=0.0001)
  dropout_attention = hparams.getfloat('polus_to', 'dropout_attention', fallback=0.5)
  lr = hparams.getfloat('polus_to', 'lr', fallback=0.001)
  clipnorm = hparams.getint('polus_to', 'clipnorm', fallback=1)
  loss_fn = hparams.getint('polus_to', 'loss', fallback='binary_crossentropy')

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


def polus_te(emb, max_length, hparams):
  """
  Polus model with text and emojis input
  :param emb:
  :param max_length:
  :param hparams:
  :return:
  """

  classes = 3
  noise = hparams.getfloat('polus_te', 'noise', fallback=0.0)
  dropout_words = hparams.getfloat('polus_te', 'dropout_words', fallback=0.3)
  dropout_rnn = hparams.getfloat('polus_te', 'dropout_rnn', fallback=0.3)
  dropout_rnn_U = hparams.getfloat('polus_te', 'dropout_rnn_U', fallback=0.3)
  dropout_dense = hparams.getfloat('polus_te', 'dropout_dense', fallback=0.3)
  cells = hparams.getint('polus_te', 'cells', fallback=150)
  l2_reg = hparams.getfloat('polus_te', 'l2_reg', fallback=0.01)
  loss_l2 = hparams.getfloat('polus_te', 'loss_l2', fallback=0.0001)
  dropout_attention = hparams.getfloat('polus_te', 'dropout_attention', fallback=0.5)
  lr = hparams.getfloat('polus_te', 'lr', fallback=0.001)
  emoji_count = hparams.getint('polus_te', 'emoji_count', fallback=1018)
  clipnorm = hparams.getint('polus_te', 'clipnorm', fallback=1)
  loss_fn = hparams.getint('polus_te', 'loss', fallback='binary_crossentropy')

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


