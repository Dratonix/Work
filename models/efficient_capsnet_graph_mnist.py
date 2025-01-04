# Copyright 2021 Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
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
from utils.layers import PrimaryCaps, FCCaps, Length, Mask

import tensorflow as tf

def efficient_capsnet_graph(input_shape):
    inputs = tf.keras.Input(input_shape)
    
    # Conv layers
    x = tf.keras.layers.Conv2D(32, 5, activation="relu", padding='valid', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, 2, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # LSTM Layers
    x = tf.keras.layers.Reshape((-1, 128))(x)  # Flatten spatial dimensions
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)  # LSTM
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)  # Another LSTM
    x = tf.keras.layers.BatchNormalization()(x)

    # Now we need to reshape the output of the LSTM to pass it to PrimaryCaps
    print("Shape before PrimaryCaps:", x.shape)  # Print shape for debugging
    
    # Reshaping to match the expected shape for PrimaryCaps
    # We will treat the second axis as the height (timesteps) and reshape accordingly
    # Suppose PrimaryCaps expects (batch_size, height, width, num_capsules, capsule_dim)
    
    # Reshaping the output of LSTM into capsules shape
    # Example: Let's assume you want 32 capsules, each of size 8
    num_capsules = 32
    capsule_dim = 8
    
    # Calculate the new height and width after LSTM processing
    batch_size = tf.shape(x)[0]
    timesteps = tf.shape(x)[1]  # This is the "height" of the capsules in the next layer
    
    # We assume we can split the LSTM output channels into the number of capsules you want
    x = tf.reshape(x, [batch_size, timesteps, num_capsules, capsule_dim])
    
    print("Shape after reshaping for PrimaryCaps:", x.shape)
    
    # Pass into PrimaryCaps layer (if PrimaryCaps is a custom layer)
    # x = PrimaryCaps(num_capsules=num_capsules, capsule_dim=capsule_dim)(x)
    
    return x  # You can return or continue processing through your Capsule layers

# Example usage:
model = efficient_capsnet_graph((28, 28, 1))  # Input shape for MNIST-like data


def generator_graph(input_shape):
    """
    Generator graph architecture.

    Parameters
    ----------   
    input_shape: list
        network input shape
    """
    inputs = tf.keras.Input(16*10)
    
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid', kernel_initializer='glorot_normal')(x)
    x = tf.keras.layers.Reshape(target_shape=input_shape, name='out_generator')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x, name='Generator')


def build_graph(input_shape, mode, verbose):
    """
    Efficient-CapsNet graph architecture with reconstruction regularizer. The network can be initialize with different modalities.

    Parameters
    ----------   
    input_shape: list
        network input shape
    mode: str
        working mode ('train', 'test' & 'play')
    verbose: bool
    """
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.layers.Input(shape=(10,))
    noise = tf.keras.layers.Input(shape=(10, 16))

    efficient_capsnet = efficient_capsnet_graph(input_shape)

    if verbose:
        efficient_capsnet.summary()
        print("\n\n")
    
    digit_caps, digit_caps_len = efficient_capsnet(inputs)
    noised_digitcaps = tf.keras.layers.Add()([digit_caps, noise]) # only if mode is play
    
    masked_by_y = Mask()([digit_caps, y_true])  
    masked = Mask()(digit_caps)
    masked_noised_y = Mask()([noised_digitcaps, y_true])
    
    generator = generator_graph(input_shape)

    if verbose:
        generator.summary()
        print("\n\n")

    x_gen_train = generator(masked_by_y)
    x_gen_eval = generator(masked)
    x_gen_play = generator(masked_noised_y)

    if mode == 'train':   
        return tf.keras.models.Model([inputs, y_true], [digit_caps_len, x_gen_train], name='Efficinet_CapsNet_Generator')
    elif mode == 'test':
        return tf.keras.models.Model(inputs, [digit_caps_len, x_gen_eval], name='Efficinet_CapsNet_Generator')
    elif mode == 'play':
        return tf.keras.models.Model([inputs, y_true, noise], [digit_caps_len, x_gen_play], name='Efficinet_CapsNet_Generator')
    else:
        raise RuntimeError('mode not recognized')
