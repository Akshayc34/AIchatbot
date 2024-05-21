# Import libraries
import os
import cv2
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define directory paths, image size, learning rate, and model name
TRAIN_DIR = "/Users/akshaychudasama/Desktop/Ai assingment/Datasets/train"
TEST_DIR = "/Users/akshaychudasama/Desktop/Ai assingment/Datasets/test"
IMG_SIZE = 200
LR = 1e-3
MODEL_NAME = 'messivsronaldo-{}-{}.model'.format(LR, '6conv-basic')

# Labelling the dataset
def label_img(img_path):
    label = 0 if 'messi' in img_path.lower() else 1
    return label

# Data augmentation and preprocessing
def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize pixel values to range [0, 1]
    return img

def create_train_data():
    training_data = []

    for img_file in tqdm(os.listdir(TRAIN_DIR)):
        try:
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            path = os.path.join(TRAIN_DIR, img_file)

            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img_data is None:
                print(f"Unable to load image: {img_file}")
                continue

            img_data = preprocess_image(img_data)

            label = label_img(img_file)

            training_data.append([img_data, label])

        except Exception as e:
            print(f"Error processing image: {img_file}, Error: {str(e)}")

    # Shuffle the training data
    shuffle(training_data)

    return training_data

def process_test_data():
    testing_data = []
    for img_file in tqdm(os.listdir(TEST_DIR)):
        try:
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            path = os.path.join(TEST_DIR, img_file)

            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img_data is None:
                print(f"Unable to load image: {img_file}")
                continue

            img_data = preprocess_image(img_data)

            label = label_img(img_file)

            testing_data.append([img_data, label])
        except Exception as e:
            print(f"Error processing image: {img_file}, Error: {str(e)}")

    shuffle(testing_data)

    return testing_data

# Running the training and the testing in the dataset for our model
train_data = create_train_data()
test_data = process_test_data()

if not test_data:
    print("Test data is empty. Exiting.")
else:
    import tensorflow as tf

    # Adjusting the model architecture
    tf.compat.v1.reset_default_graph()
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 256, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 512, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    # Splitting the testing data and training data
    X_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    Y_train = np.array([i[1] for i in train_data])
    X_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    Y_test = np.array([i[1] for i in test_data])

    # Fitting the data into our model with increased epochs
    model.fit({'input': X_train}, {'targets': np.eye(2)[Y_train]}, n_epoch=20,
              validation_set=({'input': X_test}, {'targets': np.eye(2)[Y_test]}),
              snapshot_step=30, show_metric=True, run_id=MODEL_NAME)
    model.save(MODEL_NAME)

    # Testing the data
    fig = plt.figure()

    for num, data in enumerate(test_data[:20]):
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

        # Make prediction
        model_out = model.predict([data])[0]

        # Get the predicted label
        predicted_label_index = np.argmax(model_out)
        if predicted_label_index == 0:
            str_label = 'Messi'
        else:
            str_label = 'Ronaldo'

        # Display the image and label
        y = fig.add_subplot(4, 5, num + 1)
        y.imshow(orig, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)

    plt.show()
