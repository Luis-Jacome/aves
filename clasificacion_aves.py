import os
import numpy             as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow        as tf
from matplotlib.pyplot import imshow
from tensorflow        import keras
from tensorflow.keras import layers
from keras.models      import Sequential, Model
from keras.utils       import image_dataset_from_directory
from keras.layers      import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.utils import plot_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

import time
import datetime
#import zoneinfo
#zona_hermosillo = zoneinfo.ZoneInfo("America/Hermosillo")
#t = time.time()

#!gdown '1tvQqvds-LMeb5acCrG-FnIi_ROQ5WeMH'

#!unzip Dataset_Aves.zip
df_clases = pd.read_csv( "Dataset_Aves/birds.csv"   )

#df_clases.shape

#df_clases.head()

#df_clases[df_clases['data set'] == 'train']['class id'].value_counts()

training_directory   = 'Dataset_Aves/train'
test_directory       = 'Dataset_Aves/test'
validation_directory = 'Dataset_Aves/valid'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
IMG_SHAPE = IMG_SIZE + (3,)

train_data_generator = ImageDataGenerator(rescale=1/255.)
val_datagenerator = ImageDataGenerator(rescale=1/255.)
test_datagenerator = ImageDataGenerator(rescale=1/255.)

train_dataset = train_data_generator.flow_from_directory(training_directory,
                                                         target_size = IMG_SIZE,
                                                         batch_size = 32,
                                                         shuffle = True,
                                                         class_mode = "categorical")

validation_dataset = val_datagenerator.flow_from_directory(validation_directory,
                                                         target_size = IMG_SIZE,
                                                         batch_size = 32,
                                                         shuffle = False,
                                                         class_mode = "categorical")

test_dataset = test_datagenerator.flow_from_directory(test_directory,
                                                         target_size = IMG_SIZE,
                                                         batch_size = 32,
                                                         shuffle = False,
                                                         class_mode = "categorical")

IMG_SHAPE = IMG_SIZE + (3,)
inputs = tf.keras.Input(shape=IMG_SHAPE)
base = MobileNetV2(include_top = False, input_tensor = inputs)
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(525, activation = "softmax")(x)
model = Model(inputs = inputs, outputs= x)

#model.summary()

base_learning_rate = 0.0001
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss      = tf.keras.losses.CategoricalCrossentropy( from_logits = False ),
    metrics   = ['accuracy']
)

#hora_hermosillo = datetime.datetime.now(zona_hermosillo)
#print(f'En Hermosillo, son las {hora_hermosillo.hour:2}:{hora_hermosillo.minute:2}:{hora_hermosillo.second:2}')

historial= model.fit( train_dataset, validation_data=validation_dataset,  epochs = 5, verbose=1 )

#Gráficas

acc =  [0.] + historial.history['accuracy']
val_acc = [0.] + historial.history['val_accuracy']

loss = [ np.max(historial.history['loss']) ] + historial.history['loss']
val_loss = [ np.max(historial.history['val_loss']) ] + historial.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.xlim([1,5])
plt.ylim([0,1.1])
plt.grid(True)
plt.title('Training Accuracy\nMobileNetV2')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.xlim([1,5])
max1 = np.max(loss)
max2 = np.max(val_loss)
plt.ylim([0, np.max([max1,max2])+0.1 ])
plt.grid(True)
plt.title('Training Loss')
plt.xlabel('epoch')
plt.savefig('Metrics.png')


#Loss, Accuracy = model.evaluate(train_dataset)

#print("    Train Loss: {:.5f}".format(Loss))
#print("Train Accuracy: {:.2f}%".format(Accuracy * 100))

#Loss, Accuracy = model.evaluate(validation_dataset)

#print("    Validation Loss: {:.5f}".format(Loss))
#print("Validation Accuracy: {:.2f}%".format(Accuracy * 100))

#Loss, Accuracy = model.evaluate(test_dataset)

#print("Test Loss: {:.5f}".format(Loss))
#print("Test Accuracy: {:.2f}%".format(Accuracy * 100))

model.save('model.h5')
