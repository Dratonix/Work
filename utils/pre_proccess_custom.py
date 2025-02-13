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
import os
import numpy as np
import tensorflow as tf
import cv2
import librosa
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random
from scipy import signal  # Corrected import for signal
from keras import Model
from keras.utils import to_categorical
from skimage.transform import resize

# TensorFlow 2.x compatibility
tf2 = tf.compat.v2

# constants
MNIST_IMG_SIZE = 28
MNIST_TRAIN_IMAGE_COUNT = 60000
PARALLEL_INPUT_CALLS = 16

# normalize dataset
def pre_process(image, label):
    return (image / 256)[..., None].astype('float32'), tf.keras.utils.to_categorical(label, num_classes=2)

def dataload(target_names, data_dir):
    target_count = []
    X_names = []
    y = []
    MAX_SAMPLES = 10000
    j = 0

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

    return X_train, y_train, X_test, y_test
def generator(image, label):
    return (image, label), (label, image)

def generate_tf_data(target_names, data_dir, batch_size=16):
    X_train,y_train,X_test,y_test=dataload(target_names=target_names, data_dir=data_dir)
    dataset_train = datagen_mfcc(X_train, y_train, batch_size=16, target_names=target_names)
    dataset_test = datagen_mfcc(X_test, y_test, batch_size=16, target_names=target_names)
    
    return dataset_train, dataset_test

            # Convert sample weight
from skimage.transform import resize

def datagen_mfcc(X_train, y_train, batch_size, target_names, SR=16000, n_mfcc=28, duration=28):
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
                
                # Resize MFCC to change height from 876 to 28
                mfcc_resized = resize(mfcc, (28, n_mfcc), mode='reflect')  # Resize to (28, n_mfcc)
                
                x_batch.append(mfcc_resized)
                y_batch.append(labels_batch[i])
                
            # Convert to numpy arrays
            x_batch = np.array(x_batch, dtype=np.float32)
            y_batch = np.array(y_batch, dtype=np.float32)
            
            # Normalize MFCC features
            x_batch = (x_batch - np.mean(x_batch, axis=0)) / np.std(x_batch, axis=0)
            x_batch = np.expand_dims(x_batch, axis=-1)  # Add the channel dimension
            
            # One-hot encode labels
            y_batch = to_categorical(y_batch, num_classes=len(target_names))
            
            # Yield the batch
            yield x_batch, y_batch
