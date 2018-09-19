import sys
import matplotlib

import os
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import misc
from PIL import Image

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.callbacks import Callback
from keras import backend as K
from keras import metrics

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# input image dimensions
img_rows, img_cols, img_chns = 64, 64, 3

# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

batch_size = 16

original_img_size = (img_rows, img_cols, img_chns)
latent_dim = 1024
epsilon_std = 1.0
epochs = 30

def read_data(filepath):
	print('=== read_data ===')

	data = []
	file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
	file_list.sort()
	
	for i, filename in enumerate(file_list):
		image = scipy.misc.imread(os.path.join(filepath, filename))
		data.append(image)


	data = np.array(data)
	data = data/255
	print('=== done ===')
	return data

def read_label(filepath):
    print('=== read_label ===')

    data = pd.read_csv(filepath)
    data = np.array(data)
    data = data[:,1:].astype('float')

    print('=== done ===')
    return data

def main():

	x_test = read_data(sys.argv[1]+'/test')
	y_label = read_label(sys.argv[1]+'/test.csv')
	x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

	"""
	Bulid Model
	"""

	#####################
	#		Encoder		#
	#####################

	x = Input(shape=original_img_size)
	conv_1 = Conv2D(img_chns,
					kernel_size=(2, 2),
					padding='same', activation='relu')(x)
	conv_2 = Conv2D(filters,
					kernel_size=(2, 2),
					padding='same', activation='relu',
					strides=(2, 2))(conv_1)
	conv_3 = Conv2D(filters,
					kernel_size=num_conv,
					padding='same', activation='relu',
					strides=1)(conv_2)
	conv_4 = Conv2D(filters,
					kernel_size=num_conv,
					padding='same', activation='relu',
					strides=1)(conv_3)
	
	flat = Flatten()(conv_4)

	z_mean = Dense(latent_dim)(flat)
	z_log_var = Dense(latent_dim)(flat)


	np.random.seed(0)
	def sampling(args):
		z_mean, z_log_var = args
		epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
								  mean=0., stddev=epsilon_std)
		return z_mean + K.exp(z_log_var/2) * epsilon

	z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

	#####################
	#		Decoder		#
	#####################

	decoder_upsample = Dense(filters * 32 * 32, activation='relu')

	output_shape = (batch_size, 32, 32, filters)

	decoder_reshape = Reshape(output_shape[1:])
	decoder_deconv_1 = Conv2DTranspose(filters,
									   kernel_size=num_conv,
									   padding='same',
									   strides=1,
									   activation='relu')
	decoder_deconv_2 = Conv2DTranspose(filters,
									   kernel_size=num_conv,
									   padding='same',
									   strides=1,
									   activation='relu')

	decoder_deconv_3_upsamp = Conv2DTranspose(filters,
											  kernel_size=(3, 3),
											  strides=(2, 2),
											  padding='valid',
											  activation='relu')
	decoder_mean_squash = Conv2D(img_chns,
								 kernel_size=2,
								 padding='valid',
								 activation='sigmoid')

	up_decoded = decoder_upsample(z)
	reshape_decoded = decoder_reshape(up_decoded)
	deconv_1_decoded = decoder_deconv_1(reshape_decoded)
	deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
	x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
	x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

	#####################
	#		Loss		#
	#####################

	lambda_kl = 3e-5
	mse_loss = K.mean(K.square(K.flatten(x) - K.flatten(x_decoded_mean_squash)))
	kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	vae_loss = mse_loss + kl_loss * lambda_kl 

	def loss1 (args):
		x, x_decoded_mean_squash = args
		mse_loss = K.mean(K.square(K.flatten(x) - K.flatten(x_decoded_mean_squash)))
		return mse_loss

	def loss2 (args):
		x, x_decoded_mean_squash = args
		kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
		return kl_loss

	loss1 = Lambda(loss1)([x, x_decoded_mean_squash])
	loss2 = Lambda(loss2)([x, x_decoded_mean_squash])

	vae = Model(x, [x_decoded_mean_squash, loss1, loss2])
	
	vae.add_loss(vae_loss)

	#vae.summary()

	vae.load_weights('./vae_weights.h5')

	#############################
	#		Reconstruct 		#
	#############################

	y_test = vae.predict(x_test)[0]
	for i in range(1,11):
		plt.subplot(2, 10, i)
		plt.imshow(x_test[i])
		plt.axis('off')
		plt.subplot(2, 10, i + 10)
		plt.imshow(y_test[i])
		plt.axis('off')
	plt.savefig(sys.argv[2]+'/figure1_3.jpg')
	plt.close()

	print('reconstruct.png saved')

	#############################
	#		Generated_pics		#
	#############################

	decoder_input = Input(shape=(latent_dim,))
	_up_decoded = decoder_upsample(decoder_input)
	_reshape_decoded = decoder_reshape(_up_decoded)
	_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
	_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
	_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
	_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
	generator = Model(decoder_input, _x_decoded_mean_squash)

	np.random.seed(87)
	random_input = np.random.random((32, latent_dim))
	generated_pics = generator.predict(random_input)
	generated_pics = generated_pics.reshape(-1, 64, 64, 3)

	for i in range(1,33):
		plt.subplot(4, 8, i)
		plt.imshow(generated_pics[i-1])
		plt.axis('off')
	plt.savefig(sys.argv[2]+'/figure1_4.jpg')
	plt.close()

	print('generated_pics.png saved')

	#############################
	#		TSNE_pics			#
	#############################

	x = Input(shape=original_img_size)
	conv_1 = Conv2D(img_chns,
					kernel_size=(2, 2),
					padding='same', activation='relu')(x)
	conv_2 = Conv2D(filters,
					kernel_size=(2, 2),
					padding='same', activation='relu',
					strides=(2, 2))(conv_1)
	conv_3 = Conv2D(filters,
					kernel_size=num_conv,
					padding='same', activation='relu',
					strides=1)(conv_2)
	conv_4 = Conv2D(filters,
					kernel_size=num_conv,
					padding='same', activation='relu',
					strides=1)(conv_3)
	
	flat = Flatten()(conv_4)

	z_mean = Dense(latent_dim)(flat)
	z_log_var = Dense(latent_dim)(flat)
	z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
	Encoder = Model(x, z)
	latent = Encoder.predict(x_test)

	y_label = y_label[:, 12]
	x_tsne = TSNE(n_components=2,learning_rate=10).fit_transform(latent)

	plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_label, s=5)
	plt.xlim(-10, 10)
	plt.ylim(-10, 10)
	plt.savefig(sys.argv[2]+'/figure1_5.jpg')

	print('tSNE.png saved')

if __name__ == '__main__':
	main()