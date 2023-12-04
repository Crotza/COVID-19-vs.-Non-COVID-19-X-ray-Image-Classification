from google.colab import drive
drive.mount('contentdrive')

# Suppress TensorFlow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import time as tm

# Paths and parameters
train_path = 'contentdriveMyDrivedataset-comparison-2000train'
test_path = 'contentdriveMyDrivedataset-comparison-2000test'
valid_path = 'contentdriveMyDrivedataset-comparison-2000validation'
BATCH_SIZE = 16
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Data generators
image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1.255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.255)

train = image_gen.flow_from_directory(
    train_path,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    color_mode = rgb,
    class_mode = 'binary',
    batch_size = BATCH_SIZE
)

test = test_data_gen.flow_from_directory(
    test_path,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    color_mode = rgb,
    class_mode = 'binary',
    batch_size = BATCH_SIZE,
    shuffle = False
)

valid = test_data_gen.flow_from_directory(
    valid_path,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    color_mode = rgb,
    class_mode = 'binary',
    batch_size = BATCH_SIZE
)

# Function to create ResNet50 model
def create_resnet50_model()
    base_model = tf.keras.applications.ResNet50(weights = 'imagenet', include_top = False, input_shape = (IMG_WIDTH, IMG_HEIGHT, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

# Function to create VGG16 model
def create_vgg16_model()
    base_model = tf.keras.applications.VGG16(weights = 'imagenet', include_top = False, input_shape = (IMG_WIDTH, IMG_HEIGHT, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

# Function to create NASNetMobile model
def create_nasnetmobile_model()
    base_model = tf.keras.applications.NASNetMobile(weights='imagenet', include_top = False, input_shape = (IMG_WIDTH, IMG_HEIGHT, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

# Function to create DenseNet121 model
def create_densenet_model()
    base_model = tf.keras.applications.DenseNet121(weights = 'imagenet', include_top = False, input_shape = (IMG_WIDTH, IMG_HEIGHT, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

# Function to create MobileNet model
def create_mobilenet_model()
    base_model = tf.keras.applications.MobileNet(weights = 'imagenet', include_top = False, input_shape = (IMG_WIDTH, IMG_HEIGHT, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

# Function to create Custom CNN model
def create_custom_cnn(input_shape = (IMG_WIDTH, IMG_HEIGHT, 3))
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

# Create models
resnet50_model = create_resnet50_model()
vgg16_model = create_vgg16_model()
nasnetmobile_model = create_nasnetmobile_model()
densenet_model = create_densenet_model()
mobilenet_model = create_mobilenet_model()
custom_cnn_model = create_custom_cnn()

# Callbacks
early = tf.keras.callbacks.EarlyStopping(monitor = val_loss, mode = min, patience = 3)
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 2, verbose = 1, factor = 0.3, min_lr = 0.000001)
callbacks_list = [early, learning_rate_reduction]

# Class weights
class_weights = sk.utils.class_weight.compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(train.classes),
    y = train.classes
)
cw = dict(zip(np.unique(train.classes), class_weights))

# Training models
print(Training ResNet50...);
start_time = tm.time()
resnet50_history = resnet50_model.fit(train, epochs = 25, validation_data = valid, class_weight = cw, callbacks = callbacks_list)
end_time = tm.time()
totaltime_resnet50 = end_time - start_time
print(fTotal training time for ResNet50 {totaltime_resnet50.2f} seconds)

print(Training VGG16...);
start_time = tm.time()
vgg16_history = vgg16_model.fit(train, epochs = 25, validation_data = valid, class_weight = cw, callbacks = callbacks_list)
end_time = tm.time()
totaltime_vgg16 = end_time - start_time
print(fTotal training time for VGG16 {totaltime_vgg16.2f} seconds)

print(Training NASNetMobile...);
start_time = tm.time()
nasnetmobile_history = nasnetmobile_model.fit(train, epochs = 25, validation_data = valid, class_weight = cw, callbacks = callbacks_list)
end_time = tm.time()
totaltime_nasnetmobile = end_time - start_time
print(fTotal training time for NASNetMobile {totaltime_nasnetmobile.2f} seconds)

print(Training DenseNet121...);
start_time = tm.time()
densenet_history = densenet_model.fit(train, epochs = 25, validation_data = valid, class_weight = cw, callbacks = callbacks_list)
end_time = tm.time()
totaltime_densenet121 = end_time - start_time
print(fTotal training time for DenseNet121 {totaltime_densenet121.2f} seconds)


print(Training MobileNet...);
start_time = tm.time()
mobilenet_history = mobilenet_model.fit(train, epochs = 25, validation_data = valid, class_weight = cw, callbacks = callbacks_list)
end_time = tm.time()
totaltime_mobilenet = end_time - start_time
print(fTotal training time for MobileNet {totaltime_mobilenet.2f} seconds)


print(Training Custom CNN...);
start_time = tm.time()
custom_cnn_history = custom_cnn_model.fit(train, epochs = 25, validation_data = valid, class_weight = cw, callbacks = callbacks_list)
end_time = tm.time()
totaltime_customcnn = end_time - start_time
print(fTotal training time for Custom CNN {totaltime_customcnn.2f} seconds)


def plot_metrics(history, model_name)
    epochs = range(1, len(history.history['accuracy']) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot training and validation accuracy
    ax1.plot(epochs, history.history['accuracy'], 'C0', label='Training Accuracy')
    ax1.plot(epochs, history.history['val_accuracy'], 'C0', linestyle='dashed', label='Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='y')

    # Plot training and validation loss
    ax2 = ax1.twinx()
    ax2.plot(epochs, history.history['loss'], 'C1', label='Training Loss')
    ax2.plot(epochs, history.history['val_loss'], 'C1', linestyle='dashed', label='Validation Loss')
    ax2.set_ylabel('Loss')
    ax2.tick_params(axis='y')

    # If learning rate is recorded in history, plot learning rate on a second y-axis
    if 'lr' in history.history
        ax3 = ax1.twinx()
        ax3.plot(epochs, history.history['lr'], 'C2', label='Learning Rate')
        ax3.set_ylabel('Learning Rate')
        ax3.tick_params(axis='y', rotation=0)
        ax3.spines['right'].set_position(('outward', 60))

    # Combine all the legends into one
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    if 'lr' in history.history
        lines_3, labels_3 = ax3.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3, loc='upper left')
    else
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title(f'Metrics for {model_name}')
    plt.show()

# Plot the history for each model
plot_metrics(resnet50_history, 'ResNet50')
plot_metrics(vgg16_history, 'VGG16')
plot_metrics(nasnetmobile_history, 'NASNetMobile')
plot_metrics(densenet_history, 'DenseNet121')
plot_metrics(mobilenet_history, 'MobileNet')
plot_metrics(custom_cnn_history, 'Custom CNN')

# Evaluate the models on test data
resnet50_test_accu = resnet50_model.evaluate(test)
vgg16_test_accu = vgg16_model.evaluate(test)
nasnetmobile_test_accu = nasnetmobile_model.evaluate(test)
densenet_test_accu = densenet_model.evaluate(test)
mobilenet_test_accu = mobilenet_model.evaluate(test)
customcnn_test_accu = custom_cnn_model.evaluate(test)
print(f'ResNet50 Testing Accuracy {resnet50_test_accu[1]100}%')
print(f'VGG16 Testing Accuracy {vgg16_test_accu[1]100}%')
print(f'NASNetMobile Testing Accuracy {nasnetmobile_test_accu[1]100}%')
print(f'DenseNet121 Testing Accuracy {densenet_test_accu[1]100}%')
print(f'MobileNet Testing Accuracy {mobilenet_test_accu[1]100}%')
print(f'Custom CNN Testing Accuracy {customcnn_test_accu[1]100}%')

# Function to make predictions and display results
def make_predictions_and_display(model, generator, model_name)
    predictions = model.predict(generator, verbose = 1)
    predictions = predictions  0.5

    cm = confusion_matrix(generator.classes, predictions)
    cr = classification_report(generator.classes, predictions, target_names = ['NORMAL', 'COVID'])
    print(fConfusion Matrix for {model_name}n, cm)
    print(fClassification Report for {model_name}n, cr)

    sns.heatmap(cm, annot = True, fmt = d)
    plt.title(fConfusion Matrix for {model_name})
    plt.show()

# Make predictions and display results for each model
make_predictions_and_display(resnet50_model, test, ResNet50)
make_predictions_and_display(vgg16_model, test, VGG16)
make_predictions_and_display(nasnetmobile_model, test, NASNetMobile)
make_predictions_and_display(densenet_model, test, DenseNet121)
make_predictions_and_display(mobilenet_model, test, MobileNet)
make_predictions_and_display(custom_cnn_model, test, Custom CNN)