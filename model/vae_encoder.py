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
            self.decoded_images, z, self.loss = self._build_autoencoder(input_image=self.input_image,keep_prob=kp)
            self._prepare_update_op()
            self.sess.run(tf.global_variables_initializer())



    # Gaussian MLP as encoder
    def _build_gaussian_MLP_encoder(self,input_image, keep_prob):
        with tf.variable_scope("gaussian_MLP_encoder"):
            # self.conv1_weight = generate_conv2d_weight(shape=[3,3,3,8]  ,name="conv1_weight_encode")
            # self.conv1_bias   = generate_conv2d_bias(shape=8            ,name='conv1_bias_encode')
            # self.conv2_weight = generate_conv2d_weight(shape=[3,3,8,16] ,name="conv2_weight_encode")
            # self.conv2_bias   = generate_conv2d_bias(shape=16           ,name='conv2_bias_encode')
            # self.conv3_weight = generate_conv2d_weight(shape=[3,3,16,32],name="conv3_weight_encode")
            # self.conv3_bias   = generate_conv2d_bias(shape=32           ,name='conv3_bias_encode')
            # self.conv4_weight = generate_conv2d_weight(shape=[3,3,32,64],name="conv4_weight_encode")
            # self.conv4_bias   = generate_conv2d_bias(shape=64           ,name='conv4_bias_encode')
            # self.fc_weight_for_encoder  = generate_fc_weight(shape=[576,1024]    ,name='fc_weight_encode')
            # self.fc_bias_for_encoder    = generate_fc_weight(shape=[1024]         ,name='fc_bias_encode')
            # self.encoder_params = [
            #     self.conv1_weight,self.conv1_bias,
            #     self.conv2_weight,self.conv2_bias,
            #     self.conv3_weight,self.conv3_bias,
            #     self.conv4_weight,self.conv4_bias,
            #     self.fc_weight_for_encoder   ,self.fc_bias_for_encoder
            # ]
            #
            # conv1 = tf.nn.conv2d(input_image, self.conv1_weight, strides=[1, 4, 4, 1], padding='SAME')
            # relu1 = tf.nn.elu(tf.nn.bias_add(conv1, self.conv1_bias))
            # relu1 = tf.nn.dropout(relu1, keep_prob)
            #
            # conv2 = tf.nn.conv2d(relu1, self.conv2_weight, strides=[1, 2, 2, 1], padding='SAME')
            # relu2 = tf.nn.elu(tf.nn.bias_add(conv2, self.conv2_bias))
            # relu2 = tf.nn.dropout(relu2, keep_prob)
            #
            # conv3 = tf.nn.conv2d(relu2, self.conv3_weight, strides=[1, 2, 2, 1], padding='SAME')
            # relu3 = tf.nn.elu(tf.nn.bias_add(conv3, self.conv3_bias))
            # relu3 = tf.nn.dropout(relu3, keep_prob)
            #
            # conv4 = tf.nn.conv2d(relu3, self.conv4_weight, strides=[1, 2, 2, 1], padding='SAME')
            # relu4 = tf.nn.elu(tf.nn.bias_add(conv4,self.conv4_bias))
            # relu4 = tf.nn.dropout(relu4, keep_prob)
            #
            # flatten_feature = flatten(relu4)
            # gaussian_params = (tf.matmul(flatten_feature, self.fc_weight_for_encoder) + self.fc_bias_for_encoder)
            #
            # mean = gaussian_params[:, :512]
            # stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, 512:])

            h1 = tf.layers.conv2d(input_image, 32, 8, strides=2, activation=tf.nn.elu, name="enc_conv1")
            h2 = tf.layers.conv2d(h1,          64, 6, strides=2, activation=tf.nn.elu, name="enc_conv2")
            h3 = tf.layers.conv2d(h2,         128, 4, strides=2, activation=tf.nn.elu, name="enc_conv3")
            h4 = tf.layers.conv2d(h3,         256, 3, strides=2, activation=tf.nn.elu, name="enc_conv4")
            print("h4",h4)
            h_a = tf.reshape(h4, [-1, 3*3*256])

            # h5 = tf.layers.conv2d(input_image, 8,  15, strides=3, activation=tf.nn.elu, name="enc_conv5")
            # h6 = tf.layers.conv2d(h5,         16,  5, strides=2, activation=tf.nn.elu, name="enc_conv6")
            # # h7 = tf.layers.conv2d(h6,         32,  3, strides=2, activation=tf.nn.elu, name="enc_conv7")

            # print('h6 : ',h6)

            # h_b = tf.reshape(h6, [-1,10*10*16])

            # h = tf.concat([h_a,h_b],axis=1)
            # print(h)
            # gaussian_params = tf.layers.dense(h, 4096, name="gaussian_params")
            # mean = gaussian_params[:, :2048]
            # stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, 2048:])

            self.mu     = tf.layers.dense(h_a, 2048, name="enc_fc_mu")
            self.logvar = tf.layers.dense(h_a, 2048, name="enc_fc_log_var")
            self.sigma = tf.exp(self.logvar / 2.0)
            self.epsilon = tf.random_normal([408, 2048],0.0,1.0)
            self.z = self.mu + self.sigma * self.epsilon

            return self.mu, self.sigma

    # Bernoulli MLP as decoder
    def _build_bernoulli_MLP_decoder(self,z, keep_prob):

        with tf.variable_scope("bernoulli_MLP_decoder"):
            # self.fc_for_decoder = generate_fc_weight(shape=[512,576]   ,name='fc_decoder_weight')
            # self.bias_for_decoder    = generate_fc_bias(shape=[576]     ,name='bias_decoder_weight')
            # self.conv5_weight = generate_conv2d_weight(shape=[3,3,64,32] ,name='conv5_weight_encode')
            # self.conv5_bias   = generate_conv2d_bias(shape=32            ,name='conv5_weight_bias')
            # self.conv6_weight = generate_conv2d_weight(shape=[3,3,32,16] ,name='conv6_weight_encode')
            # self.conv6_bias   = generate_conv2d_bias(shape=16            ,name='conv6_weight_bias')
            # self.conv7_weight = generate_conv2d_weight(shape=[3,3,16,8]  ,name='conv7_weight_encode')
            # self.conv7_bias   = generate_conv2d_bias(shape=8             ,name='conv7_weight_bias')
            # self.conv8_weight = generate_conv2d_weight(shape=[3,3,8,3]   ,name='conv8_weight_encode')
            # self.conv8_bias   = generate_conv2d_bias(shape=3             ,name='conv8_weight_bias')
            # self.conv9_weight = generate_conv2d_weight(shape=[3,3,3,3]   ,name='conv9_weight_encode')
            # self.conv9_bias   = generate_conv2d_bias(shape=3             ,name='conv9_weight_bias')
            # self.deocoder_params = [
            #     self.fc_for_decoder,self.bias_for_decoder,
            #     self.conv5_weight  ,self.conv5_bias,
            #     self.conv6_weight  ,self.conv6_bias,
            #     self.conv7_weight  ,self.conv7_bias,
            #     self.conv8_weight  ,self.conv8_bias,
            #     self.conv9_weight  ,self.conv9_bias
            # ]
            #
            # input_feature = (tf.matmul(z, self.fc_for_decoder) + self.bias_for_decoder)  # 576-d
            #
            # input_feature = inverse_flatten(input_feature,(-1,3,3,64)) # 3x3x64
            #
            # resize5 = tf.image.resize_images(input_feature,size=(6,6),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # conv5 = tf.nn.conv2d(resize5, self.conv5_weight,strides=[1,1,1,1],  padding='SAME')
            # conv5 = tf.nn.elu(tf.nn.bias_add(conv5, self.conv5_bias))  # 6x6x32
            # conv5 = tf.nn.dropout(conv5, keep_prob)
            #
            # resize6 = tf.image.resize_images(conv5,size=(12,12),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # conv6 = tf.nn.conv2d(resize6, self.conv6_weight,strides=[1,1,1,1] , padding='SAME')
            # conv6 = tf.nn.elu(tf.nn.bias_add(conv6, self.conv6_bias)) # 12x12x16
            # conv6 = tf.nn.dropout(conv6, keep_prob)
            #
            #
            # resize7 = tf.image.resize_images(conv6,size=(24,24),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # conv7 = tf.nn.conv2d(resize7, self.conv7_weight, strides=[1,1,1,1],padding='SAME')
            # conv7 = tf.nn.elu(tf.nn.bias_add(conv7, self.conv7_bias))  # 24x24x8
            # conv7 = tf.nn.dropout(conv7, keep_prob)
            #
            # resize8 = tf.image.resize_images(conv7,size=(42,42),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # conv8 = tf.nn.conv2d(resize8, self.conv8_weight,strides=[1,1,1,1], padding='SAME')
            # conv8 = tf.nn.elu(tf.nn.bias_add(conv8, self.conv8_bias))  # 42x42x3
            # conv8 = tf.nn.dropout(conv8, keep_prob)
            #
            # resize9 = tf.image.resize_images(conv8,size=(84,84),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # conv9 = tf.nn.conv2d(resize9, self.conv9_weight,strides=[1,1,1,1], padding='SAME')
            # decoded_image  = tf.nn.relu(tf.nn.bias_add(conv9, self.conv9_bias))
            h = tf.layers.dense(self.z, 3*3*256, name="dec_fc")
            h = tf.reshape(h, [-1, 3, 3, 256])
            print("fuck u")

            h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.elu, name="dec_deconv1")  # 9, 9, 128
            print(h)

            h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.elu, name="dec_deconv2")   #  21, 21, 64
            print(h)

            h = tf.layers.conv2d_transpose(h, 32, 4, strides=2, activation=tf.nn.elu, name="dec_deconv3")   #  44, 44, 32
            print(h)

            h = tf.layers.conv2d_transpose(h, 3, 4, strides=2, activation=tf.nn.elu, name="dec_deconv4")    # 90, 90, 3
            print(h)

        # h = tf.layers.conv2d(h, 3, 5, strides=1, activation=tf.nn.relu, name="enc_conv4")

            # h = tf.layers.conv2d_transpose(h, 3, 4, strides=2, activation=tf.nn.sigmoid, name="dec_deconv5")

            print("fuck",h)
            decoded_image = tf.image.resize_images(h,size=(84,84),method=tf.image.ResizeMethod.BILINEAR)
            decoded_image = tf.nn.sigmoid(decoded_image)

            # decoded_image = tf.layers.conv2d_transpose(h, 3, 2, strides=1, activation=tf.nn.sigmoid, name="dec_deconv5")

        return decoded_image

    # Gateway
    def _build_autoencoder(self,input_image, keep_prob):

        target_image = tf.stop_gradient(input_image)
        # encoding
        mu, sigma = self._build_gaussian_MLP_encoder(input_image,  keep_prob)

        # sampling by re-parameterization technique
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # decoding
        y = self._build_bernoulli_MLP_decoder(z=z,keep_prob=keep_prob)

        self.output_image = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

        # loss
        eps = 1e-6 # avoid taking log of zero

        # reconstruction loss
        self.r_loss = tf.reduce_sum(
            tf.square(self.input_image - self.output_image),
            reduction_indices = [1,2,3]
        )
        self.r_loss = tf.reduce_mean(self.r_loss)

        # augmented kl loss per dim
        self.kl_loss = - 0.5 * tf.reduce_sum(
            (1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)),
            reduction_indices = 1
        )
        self.kl_loss = tf.maximum(self.kl_loss, 0.5 * 2048)
        self.kl_loss = tf.reduce_mean(self.kl_loss)

        self.loss = self.r_loss + self.kl_loss


        return self.output_image, z, self.loss


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


    def get_encode_feature(self,image):
        feature = self.sess.run(self.z,feed_dict={self.input_image:image[np.newaxis, :]})
        return feature


if __name__ == '__main__':

    env = load_thor_env(scene_name='bedroom_04',
                             random_start=True,
                             random_terminal=True,
                             terminal_id=None, start_id=None, num_of_frames=1)

    VAE = VAE_Encoder(0.9)

    n = env.n_locations # 408

    for i in range(100000):

        result = sample(range(n),408)
        data = [ env.get_image_state(id)for id in result]

        data = np.array(data)

        loss = VAE.update(images=data).tolist()

        # print(loss)
        #
        # mean_loss = sum(loss)*1.0/len(loss)

        print("Episode:%s   Loss:%s"%(i,loss))

        if i > 4000:
            for i in range(0,n):
                input_image = env.get_image_state(i)
                decoded_image= VAE.get_decoded_image(input_image)[0]

                image_path = '/home/wyz/PycharmProjects/JOC-NET/data/images/decoded_images/'
                raw_name     = image_path +'raw_'+str(i)+'.jpeg'
                decoded_name = image_path + 'decoded_'+str(i)+'.jpeg'

                print(input_image.shape)
                # print(decoded_image)
                # print(type(decoded_image))
                print(decoded_image.shape)

                cv2.imwrite(raw_name     , input_image*256.0)
                cv2.imwrite(decoded_name , decoded_image*256.0)

            feature_dict = dict()
            for state_id in range(env.n_locations):
                # current_observation = env.h5_file['observation'][state_id]/255.0
                # current_observation = np.array(current_observation)
                # current_observation = cv2.resize(current_observation,(84,84))
                current_img = env.get_image_state(state_id)
                current_feature = VAE.get_encode_feature(current_img)
                feature = current_feature[0].tolist()
                feature_dict["State_"+str(state_id)] = feature
            print('特征长度： ',len(feature))
            file_path = '/home/wyz/PycharmProjects/JOC-NET/data/feature_encoded.csv'
            df = pd.DataFrame(feature_dict)
            df.to_csv(file_path)
            break











