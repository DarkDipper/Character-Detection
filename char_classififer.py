import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_generator import parse_annotation_xml
from char_generator import CharGenerator


class CharClassifier:
    def __init__(self):
        self.labels = [
          '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z', '?'
        ]
        self.label_map = {c:idx for idx, c in enumerate(self.labels)}
        self.model = None

    def build_model(self):
        input_layer = Input(shape=[160, 160, 3])
        backend = MobileNetV2(input_shape = [160, 160, 3], input_tensor = input_layer, include_top = False)
        backend_output = backend(input_layer)

        conv_layer_1 = Conv2D(256, kernel_size = 1, activation = 'relu')(backend_output)
        conv_layer_2 = Conv2D(128, kernel_size = 2, activation = 'relu')(conv_layer_1)
        pooling_layer_1 = MaxPooling2D(pool_size = 2)(conv_layer_2)
        conv_layer_3 = Conv2D(64, kernel_size = 1, activation = 'relu')(pooling_layer_1)
        flat_layer_1 = Flatten()(conv_layer_3)
        dense_layer_1 = Dense(64, activation = 'relu')(flat_layer_1)
        output_layer = Dense(len(self.labels), activation = 'softmax')(dense_layer_1)
        
        model = Model(input_layer, output_layer)
        model.summary()

        loss_func = SparseCategoricalCrossentropy()
        opt_func = Adam(learning_rate = 1e-4)
        metric = SparseCategoricalAccuracy()

        model.compile(loss = loss_func, optimizer = opt_func, metrics = [metric])
        self.model = model
    
    def save_model(self):
        save_model(self.model, '/models/char_classifier_model.hdf5')

    def load_model(self):
        self.model = keras.models.load_model('models/char_classifier_best.h5')


    def train(self):
      train_images, _= parse_annotation_xml('/content/train/anns', '/content/train/images')
      valid_images, _= parse_annotation_xml('/content/valid/anns', '/content/valid/images')
      train_generator = CharGenerator(32, train_images, self.label_map)
      valid_generator = CharGenerator(32, valid_images, self.label_map)

      model_filepath = "/content/drive/MyDrive/DataScience_Fall2021/ds101f21project-group07/models/char_classifier_ckpt.hdf5"
      model_ckpt = ModelCheckpoint(
          model_filepath,
          monitor="val_loss",
          verbose = 1,
          save_best_only=True,
          save_weights_only=True,
      )

      e_callback = EarlyStopping(verbose=1, mode='min', patience = 15)

      self.model.fit(
        train_generator,
        validation_data = valid_generator,
        epochs = 50,
        callbacks = [e_callback, model_ckpt]
      )


    def predict(self, image):
        """
        :param image: a PIL Image object
        :return: a string of the encrypted message
        """
        predicted_string = ""
        np_image = np.array(image)
        
        for i in range(4):
          for j in range(4):
            char_image = np.array([np_image[i*160: (i+1)* 160, j*160: (j+1)*160, :]])
            y_predict = self.model.predict(char_image)
            y_prob = np.argmax(y_predict, axis = 1)
            predicted_string += self.labels[y_prob[0]] if self.labels[y_prob[0]] != '?' else ''
        return predicted_string 
  