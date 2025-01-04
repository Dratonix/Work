# Copyright 2021 Adam Byerly & Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf
import os
import cv2
import librosa
tf2 = tf.compat.v2

# constants
MNIST_IMG_SIZE = 28
MNIST_TRAIN_IMAGE_COUNT = 60000
PARALLEL_INPUT_CALLS = 16

# normalize dataset
def pre_process(image, label):
    return (image / 256)[...,None].astype('float32'), tf.keras.utils.to_categorical(label, num_classes=2)
def dataload(target_names, data_dir):

  for i, target in enumerate(target_names):
      target_count.append(0)
      path = data_dir + target + '/'  # path to each target directory
      
      # Walk through the directory and its subdirectories
      for root, dirs, files in os.walk(path):
          for filename in files:
              name, ext = os.path.splitext(filename)
              if ext.lower() == '.wav':  # Ensure case-insensitive check for .wav files
                  # Stop if we have reached the maximum number of samples
                  if j >= MAX_SAMPLES:
                      print(f"Reached maximum sample limit: {MAX_SAMPLES} samples.")
                      break
                  
                  name = os.path.join(root, filename)
                  y.append(i)
                  X_names.append(name)
                  target_count[i] += 1
                  j += 1
          
          # Break the outer loop if max samples are reached
          if j >= MAX_SAMPLES:
              break
      
      print(f'{target} #recs = {target_count[i]}')
  
    # Print total number of records collected
      print(f'total #recs = {len(y)}')
      
      # Shuffle the data before splitting
      X_names, y = shuffle(X_names, y, random_state=42)
      
      # Split the data into training and testing sets
      X_train, X_test, y_train, y_test = train_test_split(X_names, y, stratify=y, test_size=0.20, random_state=42)
      
      # Print the number of records in the train and test sets
      print(f'train #recs = {len(X_train)}')
      print(f'test #recs = {len(X_test)}')

def generator(image, label):
    return (image, label), (label, image)

def generate_tf_data(X_train, y_train, X_test, y_test, batch_size):
	dataset_train = datagen_mfcc(X_train,y_train,batch_size=16)
  dataset_test=datagen_mfcc(X_test,y_test,batch_size=16)
    
	return dataset_train, dataset_test


def datagen_mfcc(X_train, y_train, batch_size, target_names, SR=16000, n_mfcc=28, duration=28):
    """
    Data generator for training models with MFCC features.
    
    Args:
        X_train: List of file paths to audio files.
        y_train: List of labels corresponding to X_train.
        batch_size: Number of samples per batch.
        target_names: List of class names for one-hot encoding.
        SR: Sampling rate for audio files.
        n_mfcc: Number of MFCC coefficients to compute.
        duration: Fixed duration (in seconds) for each audio clip.
        
    Yields:
        A tuple (x_batch, y_batch, sample_weight) for training.
    """
    
    # Calculate class weights based on the frequency of the labels
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    while True:
        # Shuffle data at the start of each epoch
        idx = np.arange(len(X_train))
        np.random.shuffle(idx)
        X_train = [X_train[i] for i in idx]
        y_train = [y_train[i] for i in idx]
        
        for start in range(0, len(X_train), batch_size):
            x_batch = []
            y_batch = []
            sample_weight_batch = []
            
            end = min(start + batch_size, len(X_train))
            train_batch = X_train[start:end]
            labels_batch = y_train[start:end]
            
            for i in range(len(train_batch)):
                # Load audio
                data, rate = librosa.load(train_batch[i], sr=SR)
                
                # Random circular shift for data augmentation
                data = np.roll(data, random.randint(0, len(data)))
                
                # Pad or truncate audio to ensure consistent length
                target_length = duration * rate
                data = librosa.util.fix_length(data, size=target_length)
                
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=data, sr=rate, n_mfcc=n_mfcc)
                mfcc = mfcc.T  # Transpose to have time-steps first
                
                x_batch.append(mfcc)
                y_batch.append(labels_batch[i])
                
                # Assign sample weight based on class weight
                sample_weight_batch.append(class_weight_dict[labels_batch[i]])
            
            # Convert to numpy arrays
            x_batch = np.array(x_batch, dtype=np.float32)
            y_batch = np.array(y_batch, dtype=np.float32)
            
            # Normalize MFCC features
            x_batch = (x_batch - np.mean(x_batch, axis=0)) / np.std(x_batch, axis=0)
            x_batch = np.expand_dims(x_batch, axis=-1)  # Add the channel dimension
            
            # One-hot encode labels
            y_batch = to_categorical(y_batch, num_classes=len(target_names))
            
            # Convert sample weights to numpy array
            sample_weight_batch = np.array(sample_weight_batch, dtype=np.float32)
            
            yield x_batch, y_batch, sample_weight_batch

