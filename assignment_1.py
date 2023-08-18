import tensorflow as tf
import csv, datetime, pickle
from keras.preprocessing.image import load_img, img_to_array

from simple import SimpleModel
from resnet import SEBlock, ResidualBlock, BottleneckBlock, ResNetBlock, ResNetBackbone, ResNet

from sklearn.metrics import confusion_matrix
import numpy as np


EPOCH_NUM = 50
TRAINED_MODELS_PATH = './trained_models'


class SaveHistoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Capture and save training history data (accuracy and loss)
        if epoch == 0:
            self.history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
        self.history['accuracy'].append(logs.get('accuracy'))
        self.history['loss'].append(logs.get('loss'))
        self.history['val_accuracy'].append(logs.get('val_accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))


class LocalDataset():
    def __init__(self, name = './Sketch_EITZ') -> None:
        self.dataset_folder = name
        self.mapping = {}
        self.number_of_classes = 0
        
        self.train_images = []
        self.train_labels = []
        self.train_dataset = None

        self.test_images = []
        self.test_labels = []
        self.test_dataset = None

        self.img_height = 128 # originally 256
        self.img_width = 128 # originally 256
        self.batch_size = 64 # originally 32


    def load_mapping(self):
        print("loading mapping")
        with open(self.dataset_folder + "/mapping.txt") as csv_file:
            mapping = csv.reader(csv_file, delimiter='\t')
        
            for row in mapping:
                self.mapping[row[0]] = row[1]

        self.number_of_classes = len(self.mapping)


    def image_analizis(self, image_path, label):
        loaded_image = load_img(image_path)
        image_array = img_to_array(loaded_image)
        print(image_array.shape)
        print(image_array.dtype)
        print(label)


    def load_train_images(self):
        print("loading train images")
        with open(self.dataset_folder + "/train.txt") as csv_file:
            train_images = csv.reader(csv_file, delimiter='\t')

            for image in train_images:
                self.image_analizis((self.dataset_folder + '/' + image[0]), image[1])
                break
        
            for image in train_images:
                image_path = self.dataset_folder + '/' + image[0]
                self.train_images.append(image_path)
                self.train_labels.append(int(image[1]))
        
        self.train_dataset = self.create_dataset(self.train_images, self.train_labels)


    def load_test_images(self):
        print("loading test images")
        with open(self.dataset_folder + "/test.txt") as csv_file:
            test_images = csv.reader(csv_file, delimiter='\t')
        
            for image in test_images:
                image_path = self.dataset_folder + '/' + image[0]
                self.test_images.append(image_path)
                self.test_labels.append(int(image[1]))
        
        self.test_dataset = self.create_dataset(self.test_images, self.test_labels)


    def create_dataset(self, images_path, labels):
        dataset = tf.data.Dataset.from_tensor_slices((images_path, labels))
        dataset = dataset.map(self.parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset


    def parse_image(self, image_path, label):
        image_decoded = tf.image.decode_png(tf.io.read_file(image_path), channels=3)
        image_decoded = tf.image.resize(image_decoded, (self.img_height, self.img_width))
        #image_decoded = self.datagen.random_transform(image_decoded)
        image_decoded = tf.cast(image_decoded, tf.float32) / 255.0
        label = tf.cast(label, tf.int32)

        return image_decoded, label


def main(dataset_model):
    input_shape = (dataset_model.img_height, dataset_model.img_width, 3)
    print(input_shape)

    simple_model = SimpleModel(number_of_classes=dataset_model.number_of_classes)
    simple_model = simple_model.model(input_shape)
    simple_model.summary()

    simple_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            'accuracy'
        ]
    )

    callbacks = [SaveHistoryCallback()]

    history = simple_model.fit(
        dataset_model.train_dataset,
        validation_data=dataset_model.test_dataset,
        epochs=EPOCH_NUM,
        callbacks=callbacks
    )

    timestamp = datetime.datetime.now()

    # save trainning data
    with open(f'{TRAINED_MODELS_PATH}/{timestamp}_training_history.pkl', 'wb') as f:
        pickle.dump(callbacks[0].history, f)

    # save trained model
    simple_model.save(f'{TRAINED_MODELS_PATH}/{timestamp}_simple_model.h5')

    # After training, evaluate the model on a test dataset
    test_loss, test_accuracy = simple_model.evaluate(dataset_model.test_dataset)

    print(f'Test accuracy: {test_accuracy}')
    print(f'Test loss: {test_loss}')

    true_labels = []
    predicted_labels = []

    for images, labels in dataset_model.test_dataset:
        predictions = simple_model.predict(images)
        predicted_labels.extend(np.argmax(predictions, axis=1))
        true_labels.extend(labels.numpy())

    confusion = confusion_matrix(true_labels, predicted_labels)

    with open(f'{TRAINED_MODELS_PATH}/{timestamp}_confusion_matrix.pkl', 'wb') as f:
        pickle.dump(confusion, f)


if __name__ == '__main__':
    dataset = LocalDataset()
    dataset.load_mapping()
    dataset.load_train_images()
    dataset.load_test_images()

    main(dataset)

