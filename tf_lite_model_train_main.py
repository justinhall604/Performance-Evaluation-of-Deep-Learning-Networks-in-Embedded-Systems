# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 23:01:22 2023

@author: justi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May  1 20:10:26 2022

@author: justi
"""
# disable debugging logs (specificially optimization warnings)
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import datetime

#this is to call script to convert to array
import subprocess
#import pandas as pd
# modules for tensorflow
import tensorflow as tf
from tensorflow import keras

#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense, Flatten, Conv1D, Reshape, LSTM, Conv2D, Dropout, MaxPooling2D

# modules for plotting 
import numpy as np
import matplotlib.pyplot as plt
import random

# module for converting to c array
import binascii

import csv
import pathlib

#used to get flops
from tensorflow.python.profiler import model_analyzer, option_builder
from tensorflow.keras import datasets, layers, models, losses

#Run tf on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
#device_name = tf.test.gpu_device_name()
#if not device_name:
  #raise SystemError('GPU device not found')
#print('Found GPU at: {}'.format(device_name))

print("TensorFlow version: ", tf.__version__)

from tabulate import tabulate

#tf.compat.v1.disable_eager_execution()
#sess = tf.compat.v1.Session()

# Global Vars
model_name = ''
is_optimize_int16_enabled = False
is_optimize_dynrng_enabled = False 

#### Uncomment following for 2d cnn
  # If required reshape input data to include channels (only for 2D conv)



def main():

  global train_images
  global test_images
  # Load In Dataset
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

  # Normalize pixel values to be between 0 and 1
  train_images, test_images = train_images / 255.0, test_images / 255.0

  #This is for alex net(DELETE)
  '''  (train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()
  train_images = tf.pad(train_images, [[0, 0], [2,2], [2,2]])/255
  test_images = tf.pad(test_images, [[0, 0], [2,2], [2,2]])/255
  train_images = tf.expand_dims(train_images, axis=3, name=None)
  test_images = tf.expand_dims(test_images, axis=3, name=None)
  train_images = tf.repeat(train_images, 3, axis=3)
  test_images = tf.repeat(test_images, 3, axis=3)
  x_val = train_images[-2000:,:,:,:]
  y_val = train_labels[-2000:]
  train_images = train_images[:-2000,:,:,:]
  train_labels = train_labels[:-2000]'''

  ## Setup Logging
  class Transcript(object):

      def __init__(self, filename):
          self.terminal = sys.stdout
          self.logfile = open(filename, "a")

      def write(self, message):
          self.terminal.write(message)
          self.logfile.write(message)

      def flush(self):
          # this flush method is needed for python 3 compatibility.
          # this handles the flush command by doing nothing.
          # you might want to specify some extra behavior here.
          pass

  def start(filename):
      """Start transcript, appending print output to given filename"""
      sys.stdout = Transcript(filename)

  def stop():
      """Stop transcript and return print functionality to normal"""
      sys.stdout.logfile.close()
      sys.stdout = sys.stdout.terminal 


  '''
  include_channels_for_2D_CNN()
  train_images = train_images.reshape(train_images.shape[0], 
                          train_images.shape[1], 
                          train_images.shape[2], 1)

  test_images = test_images.reshape(test_images.shape[0], 
                        test_images.shape[1], 
                        test_images.shape[2], 1)
  '''
  #####################################################################################
  ################################### Load in Model ###################################
  #####################################################################################
  model = create_model_srnn_f_d64r_d10s()

  '''
  # Log everything to text file
  import transcript
  transcript.start('logfile.log')
  print("inside file")
  transcript.stop()
  print("outside file")
  '''

  csv_logger = tf.keras.callbacks.CSVLogger(model_name + '_metrics.csv')

  from contextlib import redirect_stdout

  # Write model name to text file
  with open("training_metrics/" + model_name + '_metrics.txt', 'w') as f:
      with redirect_stdout(f):
          print(model_name)
  # write model to text file
  with open("training_metrics/" + model_name + '_metrics.txt', 'a') as f:
      with redirect_stdout(f):
          model.summary()
  #also log to terminal
  print(model.summary())



  #train_images = train_images.reshape(60000, 28*28)
  #test_images = test_images.reshape(10000, 28*28)

  model.compile(optimizer='adam',
                #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  ##history = model.fit(train_images, train_labels, epochs=10, 
  ##                    validation_data=(test_images, test_labels), callbacks=[csv_logger])
  '''
  with open("training_metrics/" + model_name + '_metrics.txt', 'a') as f:
      with redirect_stdout(f):
          history = model.fit(train_images, train_labels, epochs=10, 
                      validation_data=(test_images, test_labels))
          '''
          
  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,profile_batch='10, 15')     

  history = model.fit(train_images, train_labels,steps_per_epoch=1, epochs=1, 
              validation_data=(test_images, test_labels), 
              callbacks=[tensorboard_callback])

#Calculate Flops

  input_signature = [
      tf.TensorSpec(
          shape=(1, *params.shape[1:]), 
          dtype=params.dtype, 
          name=params.name
      ) for params in model.inputs
  ]
  
  forward_graph = tf.function(model, input_signature).get_concrete_function().graph
  options = option_builder.ProfileOptionBuilder.float_operation()
  print("get model analysis")
  graph_info = model_analyzer.profile(forward_graph, options=options,cmd='scope')
  #Converte to multipleir accumulates (TF does two ops per multiplier add)
  flops = graph_info.total_float_ops // 2

  compare_table = [(i, getattr(graph_info,i)) for i in graph_info.DESCRIPTOR.fields_by_name.keys()]
  #print(tabulate.tabulate(compare_table, headers=["Name","Value"]))

  #Write all the associated flops info
  with open("training_metrics/" + model_name + '_metrics.txt', 'a') as f:
    with redirect_stdout(f):
        print(tabulate(compare_table, headers=["Name","Value"]))
        
  #Write flops to text file
  with open("training_metrics/" + model_name + '_metrics.txt', 'a') as f:
    with redirect_stdout(f):
        print("Multiply–accumulate operation: ", flops)


  # convert the history.history dict to a pandas DataFrame:     
  #hist_df = pd.DataFrame(history.history) 
  
  # Log and print model size
  with open("training_metrics/" + model_name + '_metrics.txt', 'a') as f:
      with redirect_stdout(f):
          for key, value in history.history.items():
              print(key, value)
              
  scores = model.evaluate(test_images, test_labels, verbose=2)
  print("Model Loss:", scores[0])
  print("Model Accuracy:", scores[1])
  with open("training_metrics/" + model_name + '_metrics.txt', 'a') as f:
      with redirect_stdout(f):
          print(model.get_compile_config())
          print("Results")
          print("Model Loss:", scores[0])
          print("Model Accuracy:", scores[1])

  # Save the entire keras model
  model_name_keras = "keras_model/" + model_name + ".keras"
  model.save(model_name_keras)

  # Save a second copy as h5 just for netron until it updates 
  model_name_keras = "netron/" + model_name + ".h5"
  model.save(model_name_keras)  

  # Log and print model size
  print("Model Size:",os.stat(model_name_keras).st_size, "Bytes")
  with open("training_metrics/" + model_name + '_metrics.txt', 'a') as f:
      with redirect_stdout(f):
          print("Model Size:",os.stat(model_name_keras).st_size, "Bytes")

  ##TODO what is this again
  mnist_sampleset = tf.data.Dataset.from_tensor_slices((test_images)).batch(1)
  def representative_dataset_gen():
      for input_value in mnist_sampleset.take(100):
          yield [input_value]

  ####DO if LSTM####
  run_model = tf.function(lambda x: model(x))
  # This is important, let's fix the input size.
  BATCH_SIZE = 1
  STEPS = 28
  INPUT_SIZE = 28
  concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))
  
  # model directory.
  #MODEL_DIR = "keras_lstm"
  #model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

  #converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
  #tflite_model = converter.convert()
  ##end todo ltsm
          
  #Create Directory for files
  model_name_tflite = model_name + ".tflite"
  tflite_models_dir = pathlib.Path(__file__).parent.absolute()
  tflite_models_dir = tflite_models_dir.joinpath('tflite_models')
  tflite_models_dir.mkdir(exist_ok=True, parents=True)
  #print(tflite_models_dir)
  tflite_model_file = tflite_models_dir.joinpath(model_name_tflite)
  #print(tflite_model_file)

  

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  #tflite_model = converter.convert()
  # NOTE: The current version of TensorFlow appears to break the model when using optimizations
  #    You can try uncommenting the following if you would like to generate a smaller size .tflite model
  # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
  # converter.representative_dataset = representative_dataset_gen
  #converter.optimizations = [tf.lite.Optimize.DEFAULT]
  if is_optimize_int16_enabled is True:
    with open("training_metrics/" + model_name + '_metrics.txt', 'a') as f:
      with redirect_stdout(f):
          print("Optimization: Post-training float16 quantization")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops
    ]
  elif is_optimize_dynrng_enabled is True:
    with open("training_metrics/" + model_name + '_metrics.txt', 'a') as f:
      with redirect_stdout(f):
          print("Optimization: Post-training dynamic range quantization")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
  else:
    with open("training_metrics/" + model_name + '_metrics.txt', 'a') as f:
      with redirect_stdout(f):
          print("Optimization: None")

  MODEL_DIR = "tf_models"
  model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)
  converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
  tflite_model = converter.convert()
  #MODEL_DIR = "keras_lstm"

  print("######Made It########")
  
  #tflite_model = converter.convert()
  #tflite_model_file.write_bytes(tflite_model)
  #save another copy in main directory for easy copy paste to mcu
# Save the model.
  with open('tf_model.tflite', 'wb') as f:
    f.write(tflite_model)

  # Print and Log tflite model size
  print("tflite model size:",os.stat("tflite_models/"+model_name_tflite).st_size, "Bytes")
  with open("training_metrics/" + model_name + '_metrics.txt', 'a') as f:
      with redirect_stdout(f):
          print("tflite model size:",os.stat("tflite_models/"+model_name_tflite).st_size, "Bytes")

  #Load Into Interpreter and test

  interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
  interpreter.allocate_tensors()

  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for test_image in test_images:
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)
    #interpreter.set_tensor(input_index[0]["index"], test_images[i:i+1, :, :])

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

    # Please note: TfLite fused Lstm kernel is stateful, so we need to reset
    # the states.
    # Clean up internal states.
    interpreter.reset_all_variables()

  # Compare prediction results with ground truth labels to calculate accuracy.
  accurate_count = 0
  for index in range(len(prediction_digits)):
    if prediction_digits[index] == test_labels[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction_digits)

  # Log and print accuracy
  print("TF Lite Model Accuracy: ", accuracy)
  with open("training_metrics/" + model_name + '_metrics.txt', 'a') as f:
      with redirect_stdout(f):
          print("tflite Model Accuracy: ", accuracy)

  #save a perminent copy in directory
  command = 'cmd /c "Wsl xxd -i tflite_models/'+model_name +'.tflite > tflite_models/'+model_name+'.cc"'
  print(command)
  os.system(command)

  #Save a generic copy for copy paste in main directory
  command = 'cmd /c "Wsl xxd -i model.tflite > tf_model.cc"'
  print(command)
  os.system(command)

  #replace first line with correct code
  with open("tf_model.cc") as f:
    lines = f.readlines()

  lines # ['This is the first line.\n', 'This is the second line.\n']

  lines[0] = "alignas(8) const unsigned char tf_model[] = {\n"

  lines # ["This is the line that's replaced.\n", 'This is the second line.\n']

  with open("tf_model.cc", "w") as f:
      f.writelines(lines)

  '''
  tflite_model = converter.convert()
  open(model_name_tflite, "wb").write(tflite_model)
  interpreter = tf.lite.Interpreter(model_path=model_name_tflite)
  '''
  '''
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Adjust the model interpreter to take 10,000 inputs at once instead of just 1
  interpreter.resize_tensor_input(input_details[0]["index"], (10000, 28*28))
  interpreter.resize_tensor_input(output_details[0]["index"], (10000, 10))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Set the test input and run
  interpreter.set_tensor(input_details[0]["index"], train_images)
  interpreter.invoke()

  # Get the result and check its accuracy
  output_data = interpreter.get_tensor(output_details[0]["index"])

  a = [np.argmax(y, axis=None, out=None) for y in output_data]
  b = [np.argmax(y, axis=None, out=None) for y in test_labels]

  accuracy = (np.array(a) == np.array(b)).mean()
  print("TFLite Accuracy:", accuracy)

  def convert_to_c_array(bytes) -> str:
    hexstr = binascii.hexlify(bytes).decode("UTF-8")
    hexstr = hexstr.upper()
    array = ["0x" + hexstr[i:i + 2] for i in range(0, len(hexstr), 2)]
    array = [array[i:i+10] for i in range(0, len(array), 10)]
    return ",\n  ".join([", ".join(e) for e in array])

  tflite_binary = open("tflite_models/"+model_name_tflite, 'rb').read()
  ascii_bytes = convert_to_c_array(tflite_binary)
  c_file = "const unsigned char tf_model[] = {\n  " + ascii_bytes + "\n};\nunsigned int tf_model_len = " + str(len(tflite_binary)) + ";"
  # print(c_file)
  open("tflite_binary/" + model_name + ".h", "w").write(c_file)
  '''
  '''
  result = subprocess.run(["wsl","xxd","-i","model_f_d10.tflite > model_data.cc"], shell=True, capture_output=True, text=True)
  print(result)
  '''
  #sys.stdout.close()


def include_channels_for_2D_CNN():
  return

  

#put all models at end
#don't forget we can reshape       
#keras.layers.Reshape(target_shape=(28, 28, 1)),

# Create a model with a 28x28 pixel input vector
#    -> 1 hidden layer of 64 nodes
#    -> 10 categories of outputs (digits 0-9)
def create_model_f_d10s():   
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model
#done

def create_model_f_d10r():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d10r'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation="relu"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d16r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(16, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d16r__d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r__d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(16, activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d20r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d10s_d20r'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(20, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d32r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d10s_d32r'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d128r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d10s_d128r'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d256r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d10s_d256r'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(256, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d384r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d10s_d384r'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(384, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d512r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d10s_d512r'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(512, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model
#Flash overflow

def create_model_f_d64r_d16r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d10s_d64r_d16r'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(16, activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d16r_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(name="L1_Flatten"),
      keras.layers.Dense(16, activation="relu",name="L2_Dense16"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d16r_f_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(name="L1_Flatten"),
      keras.layers.Dense(16, activation="relu",name="L2_Dense16"),
      keras.layers.Flatten(name="L3_Flatten"),
      keras.layers.Dense(64, activation="relu",name="L4_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L5_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d16r_d16r_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_d16r_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(name="L1_Flatten"),
      keras.layers.Dense(16, activation="relu",name="L2_Dense16"),
      keras.layers.Dense(16, activation="relu",name="L3_Dense16"),
      keras.layers.Dense(64, activation="relu",name="L4_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L5_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(name="L2_Flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d16r_f_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(name="L0.5_Flatten"),
      keras.layers.Dense(16, activation="relu",name="C_L1_Dense16r"),
      keras.layers.Flatten(name="L2_Flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_f_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(name="L1_Flatten"),
      keras.layers.Flatten(name="L2_Flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d16s_f_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16s_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(name="L0.5_Flatten"),
      keras.layers.Dense(16, activation="softmax",name="C_L1_Dense16r"),
      keras.layers.Flatten(name="L2_Flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d16r_d256r_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_d256r_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(name="L1_Flatten"),
      keras.layers.Dense(16, activation="relu",name="L2_Dense16"),
      keras.layers.Dense(256, activation="relu",name="C_L3_Dense64"),
      keras.layers.Dense(64, activation="relu",name="L4_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L5_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d16r_d64r_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_d64r_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(name="L1_Flatten"),
      keras.layers.Dense(16, activation="relu",name="L2_Dense16"),
      keras.layers.Dense(64, activation="relu",name="C_L3_Dense64"),
      keras.layers.Dense(64, activation="relu",name="L4_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L5_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d16r_drop05_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_drop05_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(name="L1_Flatten"),
      keras.layers.Dense(16, activation="relu",name="L2_Dense16"),
      keras.layers.Dropout(0.5, name="drop"),
      keras.layers.Dense(64, activation="relu",name="L4_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L5_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d16r_norm_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_norm_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(name="L1_Flatten"),
      keras.layers.Dense(16, activation="relu",name="L2_Dense16"),
      #keras.layers.Dense(16, activation="relu",name="L3_Dense16"),
      keras.layers.experimental.preprocessing.Normalization(name = "Normalize"),
      keras.layers.Dense(64, activation="relu",name="L4_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L5_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model


def create_model_f_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.experimental.preprocessing.Normalization(name = "C_L1_Normalize"),
      keras.layers.Flatten(name="L2_Flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d64_f_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d64_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(name="L0.5_Flatten"),
      keras.layers.Dense(64,activation="relu",name = "C_L1_dense64"),
      keras.layers.Flatten(name="L2_Flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d256_f_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d256_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(name="L0.5_Flatten"),
      keras.layers.Dense(256,activation="relu",name = "C_L1_dense256"),
      keras.layers.Flatten(name="L2_Flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_d05_f_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_d05_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Dropout(0.5,name = "C_L1_Dropout"),
      keras.layers.Flatten(name="L2_Flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d16r_r_2p4_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)

    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_r_2p4_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      keras.layers.Flatten(name="L1_flatten"),
      keras.layers.Dense(16, activation="relu",name="L2_Dense16"),
      keras.layers.Reshape((4,4,1),name="L3_Reshape"),
      keras.layers.AveragePooling2D((2,2),strides=4,padding='valid',name="L4_2DAvgPool"),
      keras.layers.Flatten(name="L5_flatten"),
      keras.layers.Dense(64, activation="relu",name="L6_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L7_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])

    return model

def create_model_2p4_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)

    # Define the model architecture
    global model_name 
    model_name = 'model_2p4_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      #keras.layers.Reshape((4,4,1),name="L3_Reshape"),
      keras.layers.AveragePooling2D((2,2),strides=4,padding='valid',name="C_L1_2DAvgPool"),
      keras.layers.Flatten(name="L2_flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])
    return model

def create_model_2p2_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)

    # Define the model architecture
    global model_name 
    model_name = 'model_2p2_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      #keras.layers.Reshape((4,4,1),name="L3_Reshape"),
      keras.layers.AveragePooling2D((2,2),strides=2,padding='valid',name="C_L1_2DAvgPool"),
      keras.layers.Flatten(name="L2_flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])
    return model

def create_model_2p1_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)

    # Define the model architecture
    global model_name 
    model_name = 'model_2p1_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      #keras.layers.Reshape((4,4,1),name="L3_Reshape"),
      keras.layers.AveragePooling2D((2,2),strides=1,padding='valid',name="C_L1_2DAvgPool"),
      keras.layers.Flatten(name="L2_flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])
    return model

def create_model_2mp4_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)

    # Define the model architecture
    global model_name 
    model_name = 'model_2mp4_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      #keras.layers.Reshape((4,4,1),name="L3_Reshape"),
      keras.layers.MaxPooling2D((2,2),strides=4,padding='valid',name="C_L1_2DMaxPool"),
      keras.layers.Flatten(name="L2_flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])
    return model

def create_model_2mp1_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)

    # Define the model architecture
    global model_name 
    model_name = 'model_2mp1_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      #keras.layers.Reshape((4,4,1),name="L3_Reshape"),
      keras.layers.MaxPooling2D((2,2),strides=1,padding='valid',name="C_L1_2DMaxPool"),
      keras.layers.Flatten(name="L2_flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])
    return model

def create_model_f_d16r_r_2p2_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)

    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_r_2p2_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      keras.layers.Flatten(name="L1_flatten"),
      keras.layers.Dense(16, activation="relu",name="L2_Dense16"),
      keras.layers.Reshape((4,4,1),name="L3_Reshape"),
      keras.layers.AveragePooling2D((2,2),strides=2,padding='valid',name="L4_2DAvgPool"),
      keras.layers.Flatten(name="L5_flatten"),
      keras.layers.Dense(64, activation="relu",name="L6_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L7_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])

    return model

def create_model_f_d16r_r_2p1_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)

    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_r_2p1_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      keras.layers.Flatten(name="L1_flatten"),
      keras.layers.Dense(16, activation="relu",name="L2_Dense16"),
      keras.layers.Reshape((4,4,1),name="L3_Reshape"),
      keras.layers.AveragePooling2D((2,2),strides=1,padding='valid',name="L4_2DAvgPool"),
      keras.layers.Flatten(name="L5_flatten"),
      keras.layers.Dense(64, activation="relu",name="L6_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L7_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])

    return model

def create_model_f_d16r_r_2mp1_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)

    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_r_2mp1_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      keras.layers.Flatten(name="L1_flatten"),
      keras.layers.Dense(16, activation="relu",name="L2_Dense16"),
      keras.layers.Reshape((4,4,1),name="L3_Reshape"),
      keras.layers.MaxPooling2D((2,2),strides=1,padding='valid',name="L4_2DMaxPool"),
      keras.layers.Flatten(name="L5_flatten"),
      keras.layers.Dense(64, activation="relu",name="L6_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L7_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])

    return model

def create_model_f_d16r_r_2mp4_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)

    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_r_2mp4_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      keras.layers.Flatten(name="L1_flatten"),
      keras.layers.Dense(16, activation="relu",name="L2_Dense16"),
      keras.layers.Reshape((4,4,1),name="L3_Reshape"),
      keras.layers.MaxPooling2D((2,2),strides=4,padding='valid',name="L4_2DMaxPool4"),
      keras.layers.Flatten(name="L5_flatten"),
      keras.layers.Dense(64, activation="relu",name="L6_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L7_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])

    return model

def create_model_srnn_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    '''    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)'''

    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_r_2mp4_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      keras.layers.LSTM(20,time_major=False, return_sequences=True, name="C_L1_LSTM"),
      keras.layers.Flatten(name="L2_flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          #loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

    return model

def create_model_2cnn4_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)

    # Define the model architecture
    global model_name 
    model_name = 'model_2cnn4_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      keras.layers.Conv2D(4, (2,2), activation='relu',name="C_L1_2cnn4"),
      keras.layers.Flatten(name="L2_flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])

    return model

def create_model_2cnn2_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)
    # Define the model architecture
    global model_name 
    model_name = 'model_2cnn2_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      keras.layers.Conv2D(2, (2,2), activation='relu',name="C_L1_2cnn2"),
      keras.layers.Flatten(name="L2_flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])
    return model

def create_model_2cnn1_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)
    # Define the model architecture
    global model_name 
    model_name = 'model_2cnn1_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      keras.layers.Conv2D(1, (2,2), activation='relu',name="C_L1_2cnn1"),
      keras.layers.Flatten(name="L2_flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])
    return model

def create_model_2cnn1_44_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)
    # Define the model architecture
    global model_name 
    model_name = 'model_2cnn1_44_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      keras.layers.Conv2D(1, (4,4), activation='relu',name="C_L1_2cnn1"),
      keras.layers.Flatten(name="L2_flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])
    return model

def create_model_2cnn3_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)
    # Define the model architecture
    global model_name 
    model_name = 'model_2cnn3_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      keras.layers.Conv2D(3, (2,2), activation='relu',name="C_L1_2cnn3"),
      keras.layers.Flatten(name="L2_flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])
    return model

def create_model_f_d16r_r_2cnn_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)

    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_r_2cnn_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      keras.layers.Flatten(name="L1_flatten"),
      keras.layers.Dense(16, activation="relu",name="L2_Dense16"),
      keras.layers.Reshape((4,4,1),name="L3_Reshape"),
      keras.layers.Conv2D(64, (2,2), activation='relu',name="L4_2Dcnn"),
      keras.layers.Flatten(name="L5_flatten"),
      keras.layers.Dense(64, activation="relu",name="L6_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L7_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])
    return model

def create_model_f_d16r_r_2cnn16_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)

    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_r_2cnn16_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      keras.layers.Flatten(name="L1_flatten"),
      keras.layers.Dense(16, activation="relu",name="L2_Dense16"),
      keras.layers.Reshape((4,4,1),name="L3_Reshape"),
      keras.layers.Conv2D(16, (2,2), activation='relu',name="L4_2Dcnn"),
      keras.layers.Flatten(name="L5_flatten"),
      keras.layers.Dense(64, activation="relu",name="L6_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L7_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])

    return model

def create_model_f_d16r_r_2depthr_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)

    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_r_2cdepthr_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      keras.layers.Flatten(name="L1_flatten"),
      keras.layers.Dense(16, activation="relu",name="L2_Dense16"),
      keras.layers.Reshape((4,4,1),name="L3_Reshape"),
      keras.layers.DepthwiseConv2D(2, (2,2), activation='relu',name="L4_2Ddepthr"),
      keras.layers.Flatten(name="L5_flatten"),
      keras.layers.Dense(64, activation="relu",name="L6_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L7_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])

    return model

def create_model_2depthr_f_d64r_d10s():
    #Lets add in 2 more channels
    global train_images
    train_images = train_images.reshape(train_images.shape[0], 
                        train_images.shape[1], 
                        train_images.shape[2], 1)
    global test_images
    test_images = test_images.reshape(test_images.shape[0], 
                          test_images.shape[1], 
                          test_images.shape[2], 1)

    # Define the model architecture
    global model_name 
    model_name = 'model_2cdepthr_f_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=train_images.shape[1:]),
      keras.layers.DepthwiseConv2D(2, (2,2), activation='relu',name="C_L1_2Ddepthr"),
      keras.layers.Flatten(name="L2_flatten"),
      keras.layers.Dense(64, activation="relu",name="L3_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L4_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          #loss=keras.losses.MeanSquaredError(),
          metrics=['accuracy'])
    return model
    


def create_model_f_d16r_d16s_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d16r_d16s_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(name="L1_Flatten"),
      keras.layers.Dense(16, activation="relu",name="L2_Dense16"),
      keras.layers.Dense(16, activation="softmax",name="L3_Dense16"),
      keras.layers.Dense(64, activation="relu",name="L4_Dense64"),
      keras.layers.Dense(10, activation="softmax",name="L5_Dense10"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d64r_d32r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d10s_d64r_d32r'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d64r_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d10s_d64r_d64r'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d64r_d128r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d10s_d64r_d128r'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d64r_d256r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d10s_d64r_d256r'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(256, activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d64r_d384r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d10s_d64r_d384r'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(384, activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

###This use new titling system
### Initial...Hidden1......Hidden2.....ect.......Output
def create_model_f_d64r_d16r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d64r_d16r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(16, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d64r_d16r_d10s_optimized_fp16():
    # Define the model architecture
    global is_optimize_int16_enabled
    is_optimize_int16_enabled = True
    global model_name 
    model_name = 'model_f_d64r_d16r_d10s_optimized_fp16'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(16, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d64r_d32r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d64r_d32r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d64r_d128r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d64r_d128r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d64r_dr50_d128r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d64r_dr50_d128r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dropout(0.50),
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d64r_d256r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d64r_d256r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(256, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d64r_d384r_d10s_optimized_fp16():
    # Define the model architecture
    global is_optimize_int16_enabled
    is_optimize_int16_enabled = True
    global model_name 
    model_name = 'f_d64r_d384r_d10s_optimized_fp16'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(384, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d256r_d16r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d256r_d16r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(256, activation="relu"),
      keras.layers.Dense(16, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d256r_d32r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d256r_d32r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(256, activation="relu"),
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d256r_d64r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d256r_d64r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(256, activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d256r_d128r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d256r_d128r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(256, activation="relu"),
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d256r_d256r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d256r_d256r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(256, activation="relu"),
      keras.layers.Dense(256, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_f_d256r_d384r_d10s():
    # Define the model architecture
    global model_name 
    model_name = 'model_f_d256r_d384r_d10s'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Flatten(),
      keras.layers.Dense(256, activation="relu"),
      keras.layers.Dense(384, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model



def create_model_2c32_mp_2c64_mp_2c64_f_d10():
    # Define the model architecture
    global model_name 
    model_name = 'model_2c32_mp_2c64_mp_2c64_f_d10'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28,1)),
      keras.layers.Conv2D(32, (3,3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(64, (3,3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(64, (3,3), activation='relu'),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_2c16_mp_2c64_mp_2c64_f_d10():
    # Define the model architecture
    global model_name 
    model_name = 'model_2c16_mp_2c64_mp_2c64_f_d10'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28,1)),
      keras.layers.Conv2D(16, (3,3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(64, (3,3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(64, (3,3), activation='relu'),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_2c64_mp_2c64_mp_2c64_f_d10():
    # Define the model architecture
    global model_name 
    model_name = 'model_2c64_mp_2c64_mp_2c64_f_d10'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28,1)),
      keras.layers.Conv2D(64, (3,3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(64, (3,3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(64, (3,3), activation='relu'),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_2c128_mp_2c64_mp_2c64_f_d10():
    # Define the model architecture
    global model_name 
    model_name = 'model_2c128_mp_2c64_mp_2c64_f_d10'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28,1)),
      keras.layers.Conv2D(128, (3,3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(64, (3,3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(64, (3,3), activation='relu'),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model



def create_model_2c256_mp_2c64_mp_2c64_f_d10():
    # Define the model architecture
    global model_name 
    model_name = 'model_2c256_mp_2c64_mp_2c64_f_d10'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28,1)),
      keras.layers.Conv2D(256, (3,3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(64, (3,3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(64, (3,3), activation='relu'),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_1c16_f_d10():
    # Define the model architecture
    global model_name 
    model_name = 'model_1c16_f_d10'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28,)),
      keras.layers.Conv1D(1,1, activation='relu'),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_1c32_f_d10():
    # Define the model architecture
    global model_name 
    model_name = 'model_1c32_f_d10'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28,)),
      keras.layers.Conv1D(32,1, activation='relu'),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_1c64_f_d10():
    # Define the model architecture
    global model_name 
    model_name = 'model_1c64_f_d10'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28,)),
      keras.layers.Conv1D(64,1, activation='relu'),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_1c128_f_d10():
    # Define the model architecture
    global model_name 
    model_name = 'model_1c128_f_d10'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28,)),
      keras.layers.Conv1D(128,1, activation='relu'),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_1c256_f_d10():
    # Define the model architecture
    global model_name 
    model_name = 'model_1c256_f_d10'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28,)),
      keras.layers.Conv1D(256,1, activation='relu'),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_1c384_f_d10():
    # Define the model architecture
    global model_name 
    model_name = 'model_1c384_f_d10'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28,)),
      keras.layers.Conv1D(256,1, activation='relu'),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_alexnet():
    # Define the model architecture
    global model_name 
    model_name = 'model_alexnet'
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28,1)),
      keras.layers.Resizing(224, 224),
      keras.layers.Conv2D(96, (3,3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(256, (3,3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(384, (3,3), activation='relu'),
      keras.layers.Conv2D(384, (3,3), activation='relu'),
      keras.layers.Conv2D(256, (3,3), activation='relu'),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation="softmax"),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

def create_model_alexnet2():
    # Define the model architecture
    global model_name 
    model_name = 'model_alexnet2'
    model = keras.Sequential([
      #keras.layers.InputLayer(input_shape=(28, 28,1)),
      keras.layers.Resizing(224, 224, interpolation='bilinear',crop_to_aspect_ratio=False,input_shape=train_images.shape[1:]),
      keras.layers.Conv2D(96, 11, strides=4, padding='same') ,
      keras.layers.Lambda(tf.nn.local_response_normalization),
      keras.layers.Activation('relu'),
      keras.layers.MaxPooling2D(3, strides=2),
      keras.layers.Conv2D(256, 5, strides=4, padding='same'),
      keras.layers.Lambda(tf.nn.local_response_normalization),
      keras.layers.Activation('relu'),
      keras.layers.MaxPooling2D(3, strides=2),
      keras.layers.Conv2D(384, 3, strides=4, padding='same'),
      keras.layers.Activation('relu'),
      keras.layers.Conv2D(384, 3, strides=4, padding='same') ,
      keras.layers.Activation('relu') ,
      keras.layers.Conv2D(256, 3, strides=4, padding='same') ,
      keras.layers.Activation('relu') ,
      keras.layers.Flatten(),
      keras.layers.Dense(4096, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(4096, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(10, activation='softmax'),
    ])
    # Train the digit classification model
    model.compile(optimizer='adam',
          loss=keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
    return model

main()