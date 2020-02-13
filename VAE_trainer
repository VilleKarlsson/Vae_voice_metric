import numpy as np


from keras.layers import Lambda, Input, Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Deconvolution2D, Conv2DTranspose, MaxPooling2D, UpSampling2D




def sampling(args):


    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon



data =  np.load('sound_arr.npy')
length,x_dim,y_dim = np.shape(data)
data = data.reshape(length,x_dim,y_dim,1)
x_train = data.astype('float32') / 255

input_shape = (x_dim,y_dim,1)

intermediate_dim = 32*11
latent_dim = 16
batch_size = 128
epochs = 100
original_dim = 128*44

#Our VAE model is made out of encoder and decoder
#First we inialize the encoder model
inputs = Input(shape=input_shape, name='encoder_input')

x = Conv2D(32, kernel_size=3, activation='relu')(inputs)
x = MaxPooling2D()(x)
x = Conv2D(64, kernel_size=3, activation='relu')(x)
x = MaxPooling2D()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Flatten()(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])


encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
y = Dense(intermediate_dim, activation='relu')(latent_inputs)
y = Reshape((32,11,1))(y)
y = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(y)
y = BatchNormalization()(y)
y = Conv2DTranspose(32,(3, 3), strides=2, activation='relu', padding='same')(y)
y = BatchNormalization()(y)
outputs = Conv2DTranspose(1,(3, 3), activation='sigmoid', padding='same')(y)



decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()


outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

#Here we define the variational autoencoder custom loss
def my_vae_loss(y_true, y_pred):
    xent_loss = original_dim* binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    return vae_loss


# train the autoencoder
vae.compile(optimizer='adam',loss=my_vae_loss)
vae.fit(x_train,x_train,
        batch_size=batch_size,
        epochs=epochs)
vae.save_weights('vae_mlp_mnist.h5')

vae.save('variational_encoder.h5')
decoder.save('decoder_model.h5')
