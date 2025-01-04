import tensorflow as tf
import numpy as np
from utils.layers import PrimaryCaps, FCCaps, Length, Mask

def efficient_capsnet_graph(input_shape):
    
    inputs = tf.keras.Input(input_shape)

    # Conv layers
    x = tf.keras.layers.Conv2D(32, 5, activation="relu", padding='same', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)   
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(x)   
    x = tf.keras.layers.BatchNormalization()(x)

    # PrimaryCaps Layer (Ensure the Conv2D inside PrimaryCaps layer uses padding='same' and strides=1)
    x = PrimaryCaps(128, 9, 16, 8)(x)
    
    # Fully Connected Capsule Layer (FCCaps)
    digit_caps = FCCaps(10, 16)(x)

    # Length Layer
    x = tf.keras.layers.Reshape((-1, 16))(digit_caps)  # Reshape to (batch_size, time_steps, feature_dim)
    
    # LSTM layer
    x = tf.keras.layers.LSTM(64, return_sequences=False)(x)  # You can adjust the number of units (64 here)
    
    # Dense output layer for classification
    x = tf.keras.layers.Dense(5, activation='softmax')(x)  # 5 classes (woodboring pests)

    return tf.keras.Model(inputs=inputs, outputs=x, name='Efficient_CapsNet_LSTM')



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
