import tensorflow as tf
import csv, datetime
from keras.preprocessing.image import load_img, img_to_array

from simple import SimpleModel
from resnet import ResNet

from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


EPOCH_NUM = 50
TRAINED_MODELS_PATH = r'./trained_models'


class SaveHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, filename):
        super(SaveHistoryCallback, self).__init__()
        self.filename = filename
        self.history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}

    def on_epoch_end(self, epoch, logs=None):
        self.history['accuracy'].append(logs.get('accuracy'))
        self.history['loss'].append(logs.get('loss'))
        self.history['val_accuracy'].append(logs.get('val_accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))

        # Convert history to DataFrame and save to CSV
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.filename, index=False)


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


    def get_mapping_as_label_list(self):
        label_list = []
        for key in self.mapping:
            label_list.append(key)
        return label_list


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


def train_model_save_data(model, dataset, name = 'default'):
    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime("%Y_%m_%d_%H_%M_%S")

    base_file_path = TRAINED_MODELS_PATH + '/' + name + '_' + timestamp

    # save epoch results
    callbacks = [SaveHistoryCallback(base_file_path + '_history.csv')]

    # Train the model
    model.fit(
        dataset.train_dataset,
        epochs=EPOCH_NUM,
        callbacks=callbacks,
        validation_data=dataset.test_dataset
    )

    # save trained model
    model.save(base_file_path + '_model.h5')

    # save confusion matrix
    generate_confusion_matrix(model, dataset, base_file_path)


def generate_confusion_matrix(model, dataset, base_file_path):
    true_labels = []
    predicted_labels = []

    for images, labels in dataset.test_dataset:
        predictions = model.predict(images)
        predicted_labels.extend(np.argmax(predictions, axis=1))
        true_labels.extend(labels.numpy())

    # Generate the confusion matrix
    confusion = confusion_matrix(true_labels, predicted_labels)

    class_labels = dataset.get_mapping_as_label_list()

    # Create a DataFrame from the confusion matrix
    confusion_df = pd.DataFrame(confusion, index=class_labels, columns=class_labels)

    confusion_df.to_csv(base_file_path + '_confusion_matrix.csv', index=False)


def main(dataset_model):
    input_shape = (dataset_model.img_height, dataset_model.img_width, 3)
    print(input_shape)

    # make date with format year-month-day_hour:minute:second
    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime("%Y_%m_%d_%H_%M_%S")

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

    resnet_model = ResNet(number_of_classes=dataset_model.number_of_classes)
    resnet_model = resnet_model.model(input_shape)
    resnet_model.summary()

    resnet_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            'accuracy'
        ]
    )

    #train_model_save_data(simple_model, dataset_model, 'simple')
    train_model_save_data(resnet_model, dataset_model, 'resnet')


if __name__ == '__main__':
    dataset = LocalDataset()
    dataset.load_mapping()
    dataset.load_train_images()
    dataset.load_test_images()

    main(dataset)

