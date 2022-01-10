from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from data_generator import parse_annotation_xml
from char_generator import CharGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

class CharClassifier:
  def __init__(self):
    self.labels = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
        'U', 'V', 'W', 'X', 'Y', 'Z', '?']
    self.label_map = {c:idx for idx,c in enumerate(self.labels)}
    self.model = None
  def build_model(self):
    input_layer = Input(shape=[160,160,3])
    backend = MobileNetV2(input_shape=[160,160,3],input_tensor=input_layer, include_top=False)
    backend_output = backend(input_layer)
    conv_layer_1 = Conv2D(256,kernel_size=1,activation='relu')(backend_output)
    conv_layer_2 = Conv2D(128,kernel_size=2,activation='relu')(conv_layer_1)
    pool_layer_1 = MaxPooling2D(pool_size=2)(conv_layer_2)
    conv_layer_3 = Conv2D(64,kernel_size=1,activation='relu')(pool_layer_1)
    flatten_layer = Flatten()(conv_layer_3)
    dense_layer_1 = Dense(64,activation='relu')(flatten_layer)
    output_layer = Dense(len(self.labels),activation='softmax')(dense_layer_1)

    loss = SparseCategoricalCrossentropy()
    optimizer = Adam(learning_rate=1e-3)
    metric = SparseCategoricalAccuracy()

    self.model = Model(input_layer,output_layer)
    self.model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    # self.model.summary()
  def train(self):
    train_images, _ = parse_annotation_xml('/content/train/anns','/content/train/images')
    valid_images, _ = parse_annotation_xml('/content/valid/anns','/content/valid/images')
    train_generator = CharGenerator(32,train_images, self.label_map)
    valid_generator = CharGenerator(32,valid_images, self.label_map)
    
    checkpoint_callback = ModelCheckpoint(
      filepath='/content/drive/MyDrive/ds101/ds101f21project-group07/models/char_weight.hdf5',
      save_weights_only=True,
      verbose = 1,
      save_best_only=True)

    self.model.fit(
      train_generator,
      validation_data=valid_generator,
      epochs=100,
      callbacks=[checkpoint_callback]
    )
  def predict(self,image):
    output_label_map={idx:c for idx,c in enumerate(self.labels)}
    result = ""
    np_image = np.array(image)
    for idx in range(16):
      row=int(idx/4)
      column= idx%4
      char_image = np.array([np_image[row*160:(row+1)*160,column*160:(column+1)*160,:]])
      # print(char_image.shape)
      temp = output_label_map[np.argmax(self.model.predict(char_image))]
      # print(temp)
      if temp != '?':
        result=result+temp
    return result
    

      