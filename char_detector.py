import cv2
import numpy as np
from utils import decode_netout, merge_file
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.inception_v3 import preprocess_input
from losses import YoloLoss
from callbacks import MapEvaluation
from data_generator import BatchGenerator, parse_annotation_xml

class CharDetector:
    def __init__(self, intial_epoch=0):
        self.model = None
        self.anchors = [4.78485,6.07404, 8.18252,10.30598, 13.21220,13.30323, 14.06538,19.72716, 21.47912,23.24997]
        self.num_anchors = 5
        self.labels = [
          '0', '1', '2', '3', '4', '5', '6', '7', '8', 
          '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
          'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
          'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.num_classes = len(self.labels)
        self.initial_epoch = intial_epoch
        self.batch_size=32

    def build_model(self):
        backend = InceptionV3(
          include_top = False,
          weights='pretrained/inception_backend.h5',
          input_shape=[400,400,3]
        )
        output_layer = Conv2D(
          self.num_anchors*(5+self.num_classes),
          kernel_size=(1,1),
          padding='same'
        )(backend.output)
        reshape_layer = Reshape([11,11,self.num_anchors,5+self.num_classes])(output_layer)

        model = Model(backend.input,reshape_layer)
        model.summary()

        loss = YoloLoss(
          anchors=self.anchors,grid_size=[11,11],
          batch_size=self.batch_size,
          lambda_obj=5.0
        )

        optimizer = Adam(learning_rate=1e-4,decay=0.0)
        model.compile(loss=loss,optimizer=optimizer)
        self.model=model

    def save_model(self):
        pass  # delete this line and replace yours

    def load_model(self):
        self.model = load_model(
          file_path,
          custom_objects=('yolo_loss':YoloLoss(
            anchors=self.anchors,
            grid_size=[15,15],
            batch_size=self.batch_size
            )
          )
        )

    def train(self, **kwargs):
      generator_config = {
        'IMAGE_H':500, 'IMAGE_W':500, 'IMAGE_C':3, 'GRID_H':15, 'GRID_W':15,
        'BOX':self.num_anchors, 'LABELS':['cat'], 
        'CLASS':1, 'ANCHORS':self.anchors, 'BATCH_SIZE':self.batch_size
      }
      train_images,_ = parse_annotation_xml('/content/Cat-Data/train/anns','/content/Cat-Data/train/images')
      train_generator = BatchGenerator(
        train_images,generator_config
      )

      valid_images,_ = parse_annotation_xml('/content/Cat-Data/valid/anns','/content/Cat-Data/valid/images')
      valid_generator = BatchGenerator(
        train_images,generator_config
      )

      map_callback = MapEvaluation(
        self, valid_generator, save_best='models/TPhong_best_map.h5'
      )
      checkpoint=ModelCheckpoint(
        'models/checkpoint.h5',
        save_best_only=True,
      )

      self.model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=100,
        steps_per_epoch=int(len(train_generator)*0.1),
        validation_steps=int(len(valid_generator)*0.1),
        callbacks=[map_callback,checkpoint],
        initial_epoch=self.initial_epoch
      )
    def predict(self, image):
        """
        Autotest will call this function
        :param image: a PIL Image object
        :return: a list of boxes, each item is a tuple of (x_min, y_min, x_max, y_max)
        """
        resized_image = image.resize([500,500])
        np_resized_image = np.numpy(resized_image)
        batch_images = np.array([np_resized_image])

        netout = self.model.predict(batch_images)[0]

        boxes = decode_netout(netout,self.anchors,self.num_classes,0.5,0.7)
        
        list_results = []
        for box in boxes:
          xmin=box.xmin * image.width
          xmax = box.xmax*image.width
          ymin=box.ymin*image.height
          ymax=box.ymax*image.height
          list_results.append((xmin,ymin,xmax,ymax))
        return list_results

    def preprocess_input(self, image):
        return image

    def infer(self, image, iou_threshold=0.5, score_threshold=0.5):
        image = cv2.resize(image, (500, 500))
        # make it RGB (it is important for normalization of some backends)
        image = image[..., ::-1]

        image = self.preprocess_input(image)
        if len(image.shape) == 3:
            input_image = image[np.newaxis, :]
        else:
            input_image = image[np.newaxis, ..., np.newaxis]

        netout = self.model.predict(input_image)[0]

        boxes = decode_netout(netout, self.anchors,
                              self.num_classes, score_threshold, iou_threshold)

        return boxes
