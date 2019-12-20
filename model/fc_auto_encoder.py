import tensorflow as tf
from model.model_op import *
from Environment.env import *
from random import random,randint,sample
import numpy as np
import cv2


class VAE_Encoder(object):

    def __init__(self,kp):

        with tf.device('/gpu:0'):

            self.sess = tf.Session()
            self.input_image =  tf.placeholder(tf.float32,[None,84,84,3], 'State_image')
            self.decoded_images,  self.loss = self._build_autoencoder(input_image=self.input_image,keep_prob=kp)
            self._prepare_update_op()
            self.sess.run(tf.global_variables_initializer())



    # Gaussian MLP as encoder
    def _build_gaussian_MLP_encoder(self,input_image, keep_prob):
        with tf.variable_scope("gaussian_MLP_encoder"):
            # flatten_feature = flatten(input_image)
            flatten_feature = tf.reshape(input_image, [-1,84*84*3])

            h = tf.nn.elu(tf.layers.dense(flatten_feature, 4096*2, name="fc1"))
            h = tf.nn.elu(tf.layers.dense(h, 4096, name="fc2"))
            h = tf.nn.dropout(h, keep_prob)

            h = tf.nn.elu(tf.layers.dense(h, 2048, name="fc3"))
            gaussian_params = tf.nn.dropout(h, keep_prob)

            mean = gaussian_params[:, :1024]

            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, 1024:])

            return mean, stddev

    # Bernoulli MLP as decoder
    def _build_bernoulli_MLP_decoder(self,z, keep_prob):

        with tf.variable_scope("bernoulli_MLP_decoder"):
            h = tf.nn.elu(tf.layers.dense(z, 2048, name="fc4"))
            h = tf.nn.elu(tf.layers.dense(h, 4096, name="fc5"))
            h = tf.nn.dropout(h, keep_prob)

            h = tf.nn.elu(tf.layers.dense(h, 4096*2, name="fc6"))
            h = tf.nn.sigmoid(tf.layers.dense(h, 84*84*3, name="fc7"))

        # decoded_image = inverse_flatten(h,shape=[84,84,3])
            decoded_image = tf.reshape(h, [-1,84,84,3])
        return decoded_image

    # Gateway
    def _build_autoencoder(self,input_image, keep_prob):

        mu, sigma = self._build_gaussian_MLP_encoder(input_image, keep_prob)

        # sampling by re-parameterization technique
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # decoding
        y = self._build_bernoulli_MLP_decoder(z,  keep_prob)
        y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

        # loss
        marginal_likelihood = tf.reduce_sum(tf.square(input_image-y))
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

        marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = marginal_likelihood - KL_divergence

        loss = -ELBO

        return y,  loss

    def _prepare_update_op(self):
        self.OPT_A = tf.train.RMSPropOptimizer(0.0001, name='RMSPropA')

        # self.grads = [tf.clip_by_norm(item, 10) for item in tf.gradients(self.loss,
        #                                     self.encoder_params+self.deocoder_params)]
        # self.update_op = self.OPT_A.apply_gradients(list(zip(self.grads,
        #                             self.encoder_params+self.deocoder_params)))

        self.update_op = self.OPT_A.minimize(self.loss)

    def update(self,images):
        self.sess.run(self.update_op,feed_dict ={self.input_image:images})
        loss = self.sess.run(self.loss,feed_dict ={self.input_image:images})
        return loss

    def get_decoded_image(self,image):
        decoded_image = self.sess.run(self.decoded_images,feed_dict={self.input_image:image[np.newaxis, :]})
        return  decoded_image


if __name__ == '__main__':

    env = load_thor_env(scene_name='bedroom_04',
                             random_start=True,
                             random_terminal=True,
                             terminal_id=None, start_id=None, num_of_frames=1)

    VAE = VAE_Encoder(0.9)

    n = env.n_locations # 408

    for i in range(100000):

        result = sample(range(n),128)
        data = [ env.get_image_state(id)for id in result]

        data = np.array(data)

        loss = VAE.update(images=data).tolist()

        # print(loss)
        #
        # mean_loss = sum(loss)*1.0/len(loss)

        print("Episode:%s   Loss:%s"%(i,loss))

        if i > 4000:
            for i in range(0,128):
                input_image = env.get_image_state(i)
                decoded_image= VAE.get_decoded_image(input_image)[0]

                image_path = '/home/wyz/PycharmProjects/JOC-NET/data/images/'
                raw_name     = image_path +'raw_'+str(i)+'.jpeg'
                decoded_name = image_path + 'decoded_'+str(i)+'.jpeg'

                print(input_image.shape)
                # print(decoded_image)
                # print(type(decoded_image))
                print(decoded_image.shape)

                cv2.imwrite(raw_name     , input_image*256.0)
                cv2.imwrite(decoded_name , decoded_image*256.0)

            break











