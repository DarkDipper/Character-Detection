import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class Decrypter:
    def __init__(self):
        self.input_labels = [
          '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z', 'PAD'
        ]
        self.input_map = {c:idx for idx, c in enumerate(self.input_labels)}
        self.output_labels = self.labels = [
          ' ','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K', 'L', 
          'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V', 'W', 'X', 'Y', 'Z','PAD'
        ]
        self.output_map = {c:idx for idx, c in enumerate(self.output_labels)}
        self.model = None

    def build_model(self):
        input_layer = Input(shape = [16])
        emb_layer = Embedding(input_dim = len(self.input_labels), output_dim = 16)(input_layer)
        bid_layer_1 = Bidirectional(LSTM(128, return_sequences = True), merge_mode = 'ave')(emb_layer)
        bid_layer_2 = Bidirectional(LSTM(64, return_sequences = True), merge_mode = 'ave')(bid_layer_1)
        #bid_layer_3 = Bidirectional(LSTM(32, return_sequences = True), merge_mode = 'ave')(bid_layer_1)
        dense_layer_1 = Dense(32, activation = 'relu')(bid_layer_2)
        dense_layer_2 = Dense(len(self.output_labels), activation = 'softmax')(dense_layer_1)

        model = Model(input_layer, dense_layer_2)
        model.summary()
        
        loss_func = SparseCategoricalCrossentropy()
        opt_func = Adam(learning_rate = 1e-3)
        metric = SparseCategoricalAccuracy()

        model.compile(loss = loss_func, optimizer = opt_func, metrics = [metric])
        self.model = model
    
    def read_data(self, csv_filename):
        df = pd.read_csv(csv_filename)
        list_inputs = []
        list_outputs = []
        for idx, row in df.iterrows():
            enc_msg = [self.input_map[c] for c in row['enc']]
            np_enc_msg = self.input_map['PAD'] * np.ones(16)
            np_enc_msg[:len(enc_msg)] = enc_msg

            raw_msg = [self.output_map[c] for c in row['msg']]
            np_raw_msg = self.output_map['PAD'] * np.ones(16)
            np_raw_msg[:len(raw_msg)] = raw_msg

            list_inputs.append(np_enc_msg)
            list_outputs.append(np_raw_msg)
        
        return np.array(list_inputs), np.array(list_outputs)

    def save_model(self):
        self.model.save('/models/decr_final-model.h5')

    def load_model(self):
        self.model = keras.models.load_model('models/derc_best_2.h5')

    def train(self):
        x_train, y_train = self.read_data('/content/data.v3.train.csv')
        x_valid, y_valid = self.read_data('/content/data.v3.valid.csv')

        decr_path = '/content/drive/MyDrive/ds101f21project-group07/models/derc_best_2.h5'
        decr_ckpt = ModelCheckpoint(
          decr_path,
          monitor = 'val_loss',
          verbose = 1,
          save_best_only=True,
          save_weights_only=False,
        )

        e_callback = EarlyStopping(verbose=1, mode='min', patience = 10)

        self.model.fit(
          x_train, y_train,
          validation_data = (x_valid, y_valid),
          epochs = 500,
          batch_size = 32,
          callbacks = [e_callback, decr_ckpt],
        )

    def predict(self, msg):
        enc_msg = [self.input_map[c] for c in msg]
        np_enc_msg = self.input_map['PAD'] * np.ones(16)
        np_enc_msg[:len(enc_msg)] = enc_msg

        input_data = np.array([np_enc_msg])
        output_data = self.model.predict(input_data) 
        y_predict = np.argmax(output_data[0], axis = 1)
        list_outputs_char = [self.output_labels[idx] for idx in y_predict]
        return ''.join(list_outputs_char)

