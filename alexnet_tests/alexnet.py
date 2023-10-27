import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
#used to get flops
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
#used to test tf model
import numpy as np
print('TensorFlow:', tf.__version__)
#used to save model
import os
import datetime
#used to convert to tflite binary array
import pathlib


(x_train_labels_labels,train_labels_labels),(test_images_images,test_labels_labels) = datasets.mnist.load_data()

x_train_labels_labels = tf.pad(x_train_labels_labels, [[0, 0], [2,2], [2,2]])/255
test_images_images = tf.pad(test_images_images, [[0, 0], [2,2], [2,2]])/255
x_train_labels_labels = tf.expand_dims(x_train_labels_labels, axis=3, name=None)
test_images_images = tf.expand_dims(test_images_images, axis=3, name=None)
x_train_labels_labels = tf.repeat(x_train_labels_labels, 3, axis=3)
test_images_images = tf.repeat(test_images_images, 3, axis=3)
x_val = x_train_labels_labels[-2000:,:,:,:]
y_val = train_labels_labels[-2000:]
x_train_labels_labels = x_train_labels_labels[:-2000,:,:,:]
train_labels_labels = train_labels_labels[:-2000]

'''
var_batchsize = 64
var_epoch = 1
model = models.Sequential()
model.add(layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=x_train_labels_labels.shape[1:]))
model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
model.add(layers.Lambda(tf.nn.local_response_normalization))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))
model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
model.add(layers.Lambda(tf.nn.local_response_normalization))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))
model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
'''


var_batchsize = 100
var_epoch = 1
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=x_train_labels_labels.shape[1:]))
#Layers Here
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=x_train_labels_labels.shape[1:]))
#End Layers Here
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))
model.summary()


model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
history = model.fit(x_train_labels_labels, train_labels_labels, batch_size=var_batchsize, epochs=1, validation_data=(x_val, y_val))

# Get Flops
forward_pass = tf.function(
    model.call,
    input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

graph_info = profile(forward_pass.get_concrete_function().graph,
                        options=ProfileOptionBuilder.float_operation())

# The //2 is necessary since `profile` counts multiply and accumulate
# as two flops, here we report the total number of multiply accumulate ops
flops = graph_info.total_float_ops // 2
print('Flops: {:,}'.format(flops))

print("Here")

print("Evaluate Model")
model.evaluate(test_images_images, test_labels_labels)

#Lets now save data how we want

model_name = "alex_net"

# Save the entire keras model
model_name_keras = model_name + ".keras"
model.save(model_name_keras)

# Log and print model size
print("Model Size:",os.stat(model_name_keras).st_size, "Bytes")

##TODO what is this again
mnist_sampleset = tf.data.Dataset.from_tensor_slices((test_images_images)).batch(1)
def representative_dataset_gen():
    for input_value in mnist_sampleset.take(100):
        yield [input_value]
        
#Create Directory for files
model_name_tflite = model_name + ".tflite"
tflite_models_dir = pathlib.Path(__file__).parent.absolute()
tflite_models_dir = tflite_models_dir.joinpath('tflite_models')
tflite_models_dir.mkdir(exist_ok=True, parents=True)
print(tflite_models_dir)
tflite_model_file = tflite_models_dir.joinpath(model_name_tflite)
print(tflite_model_file)



converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# NOTE: The current version of TensorFlow appears to break the model when using optimizations
#    You can try uncommenting the following if you would like to generate a smaller size .tflite model
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# converter.representative_dataset = representative_dataset_gen
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#50% flash usage decrease
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
tflite_model = converter.convert()
tflite_model_file.write_bytes(tflite_model)

# Print and Log tflite model size
print("tflite model size:",os.stat("tflite_models/"+model_name_tflite).st_size, "Bytes")

#Load Into Interpreter and test

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Run predictions on every image in the "test" dataset.
prediction_digits = []
for test_image in test_images_images:
  # Pre-processing: add batch dimension and convert to float32 to match with
  # the model's input data format.
  test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
  interpreter.set_tensor(input_index, test_image)

  # Run inference.
  interpreter.invoke()

  # Post-processing: remove batch dimension and find the digit with highest probability.
  output = interpreter.tensor(output_index)
  digit = np.argmax(output()[0])
  prediction_digits.append(digit)

# Compare prediction results with ground truth labels to calculate accuracy.
accurate_count = 0
for index in range(len(prediction_digits)):
  if prediction_digits[index] == test_labels_labels[index]:
    accurate_count += 1
accuracy = accurate_count * 1.0 / len(prediction_digits)

# Log and print accuracy
print("TF Lite Model Accuracy: ", accuracy)

command = 'cmd /c "Wsl xxd -i tflite_models/'+ "alex_net" +'.tflite > tflite_models/'+"alex_net"+'.cc"'
print(command)
os.system(command)