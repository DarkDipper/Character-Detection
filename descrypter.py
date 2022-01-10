from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np

class Descrypter:
  def __init__(self):
    self.input_labels = [
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
      'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
      'U', 'V', 'W', 'X', 'Y', 'Z','PAD'
    ]
    self.input_map={c:idx for idx, c in enumerate(self.input_labels)}
    self.output_labels =[
      ' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
      'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
      'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'PAD'
    ]
    self.output_map={c:idx for idx, c in enumerate(self.output_labels)}
    self.model=None

  def build_model(self):
    input_layer = Input(shape=[16])
    embedding_layer = Embedding(input_dim=len(self.input_labels),output_dim=512)(input_layer)
    bi_layer_1 = Bidirectional(LSTM(256,return_sequences=True),merge_mode='ave')(embedding_layer)
    bi_layer_2 = Bidirectional(LSTM(128,return_sequences=True),merge_mode='ave')(bi_layer_1)
    bi_layer_3 = Bidirectional(LSTM(64,return_sequences=True),merge_mode='ave')(bi_layer_2)
    bi_layer_4 = Bidirectional(LSTM(32,return_sequences=True),merge_mode='ave')(bi_layer_3)
    dense_layer = Dense(len(self.output_labels),activation='softmax')(bi_layer_4)

    loss = SparseCategoricalCrossentropy()
    optimizer = Adam(learning_rate=1e-4)
    metric = SparseCategoricalAccuracy()
    
    self.model = Model(input_layer, dense_layer)
    
    self.model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    # self.model.summary()

  def read_data(self,csv_filename):
    df = pd.read_csv(csv_filename)
    list_input = []
    list_output = []
    for idx, row in df.iterrows():
      enc_msg = [self.input_map[c] for c in row['enc']]
      np_enc_msg = self.input_map['PAD']*np.ones(16)
      np_enc_msg[:len(enc_msg)]=enc_msg

      raw_msg = [self.output_map[c] for c in row['msg']]
      np_raw_msg = self.output_map['PAD']*np.ones(16)
      np_raw_msg[:len(raw_msg)]=raw_msg

      list_input.append(np_enc_msg)
      list_output.append(np_raw_msg)
    return np.array(list_input), np.array(list_output)

  def train(self):
    x_train, y_train = self.read_data('/content/data.v3.train.csv')
    x_valid, y_valid = self.read_data('/content/data.v3.valid.csv')

    checkpoint_callback = ModelCheckpoint(
      filepath='/content/drive/MyDrive/ds101/ds101f21project-group07/models/dec_weight.hdf5',
      save_weights_only=True,
      verbose = 1,
      save_best_only=True)

    self.model.fit(
      x_train,y_train,
      validation_data=(x_valid, y_valid),
      epochs=100,
      batch_size=32,
      callbacks=[checkpoint_callback]
    )

  def predict(self,enc_text):
    result = ""
    result_label_map = {idx:c for idx,c in enumerate(self.output_labels)}
    enc_msg = [self.input_map[c] for c in enc_text]
    np_enc_msg = self.input_map['PAD']*np.ones(16)
    np_enc_msg[:len(enc_msg)]=enc_msg
    np_enc_msg = np.array(np_enc_msg)
    temp = self.model.predict(np.array([np_enc_msg]))
    for n in temp[0]:
      c = result_label_map[np.argmax(n)]
      if c !='PAD':
        result =result+c
    return result
    
