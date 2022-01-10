from descrypter import Descrypter
from char_classifier import CharClassifier
import tensorflow as tf
class Decoder:
    def __init__(self):
        self.classifier = None
        self.descrypter = None

    def load_models(self):
        self.classifier = CharClassifier()
        # self.classifier.build_model()
        self.classifier.model = tf.keras.models.load_model('models/char_model.hdf5')

        self.descrypter = Descrypter()
        # self.descrypter.build_model()
        self.descrypter.model = tf.keras.models.load_model('models/dec_model.hdf5')

    def read_encrypted_message(self, image):
        """
        :param image: a PIL Image object
        :return: a string of the encrypted message
        """
        result = self.classifier.predict(image)
        return result  # delete this line and replace yours

    def decrypt_message(self, encrypted_message):
        """
        :param encrypted_message: a string of the encrypted message
        :return: a string of the decrypted message
        """
        result = self.descrypter.predict(encrypted_message)
        return result
    def predict(self,image):
        encrypted_message = self.read_encrypted_message(image)
        descrypted_message = self.decrypt_message(encrypted_message)
        return (encrypted_message, descrypted_message)
