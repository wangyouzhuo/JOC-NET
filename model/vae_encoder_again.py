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
            self.input_image =  tf.placeholder(tf.float32,[408,84,84,3], 'State_image')
            self.decoded_images, self.z, self.loss = self._build_autoencoder(input_image=self.input_image,keep_prob=kp)
            self._prepare_update_op()
            self.sess.run(tf.global_variables_initializer())



    # Gaussian MLP as encoder
    def _build_gaussian_MLP_encoder(self,input_image, keep_prob):
        with tf.variable_scope("gaussian_MLP_encoder"):
            self.conv1_weight = generate_conv2d_weight(shape=[8,8,3,32]   ,name="conv1_weight_encode")
            self.conv1_bias   = generate_conv2d_bias(shape=32             ,name='conv1_bias_encode')
            self.conv2_weight = generate_conv2d_weight(shape=[6,6,32,64]  ,name="conv2_weight_encode")
            self.conv2_bias   = generate_conv2d_bias(shape=64             ,name='conv2_bias_encode')
            self.conv3_weight = generate_conv2d_weight(shape=[4,4,64,128] ,name="conv3_weight_encode")
            self.conv3_bias   = generate_conv2d_bias(shape=128            ,name='conv3_bias_encode')
            self.conv4_weight = generate_conv2d_weight(shape=[3,3,128,256],name="conv4_weight_encode")
            self.conv4_bias   = generate_conv2d_bias(shape=256            ,name='conv4_bias_encode')
            self.fc_weight_for_mu  = generate_fc_weight(shape=[3*3*256,2048]    ,name='fc_weight_encode_mu')
            self.fc_bias_for_mu    = generate_fc_bias(shape=[2048]              ,name='fc_bias_encode_mu')
            self.fc_weight_for_logvar  = generate_fc_weight(shape=[3*3*256,2048],name='fc_weight_encode_logvar')
            self.fc_bias_for_logvar    = generate_fc_bias(shape=[2048]          ,name='fc_bias_encode_logvar')

            self.encoder_params = [
                self.conv1_weight     ,
                self.conv1_bias,
                self.conv2_weight     ,
                self.conv2_bias,
                self.conv3_weight     ,
                self.conv3_bias,
                self.conv4_weight     ,
                self.conv4_bias,
                self.fc_weight_for_mu ,
                self.fc_bias_for_mu,
                self.fc_weight_for_logvar,
                self.fc_bias_for_logvar
            ]

            h1 = (tf.nn.conv2d(input=input_image,filter=self.conv1_weight,strides=[1,2,2,1],padding='VALID'))
            h1 = tf.nn.bias_add(h1,self.conv1_bias)
            h1 = tf.nn.elu(h1)

            h2 = (tf.nn.conv2d(input=h1,filter=self.conv2_weight,strides=[1,2,2,1],padding='VALID'))
            h2 = tf.nn.bias_add(h2,self.conv2_bias)
            h2 = tf.nn.elu(h2)

            h3 = (tf.nn.conv2d(input=h2,filter=self.conv3_weight,strides=[1,2,2,1],padding='VALID'))
            h3 = tf.nn.bias_add(h3,self.conv3_bias)
            h3 = tf.nn.elu(h3)

            h4 = (tf.nn.conv2d(input=h3,filter=self.conv4_weight,strides=[1,2,2,1],padding='VALID'))
            h4 = tf.nn.bias_add(h4,self.conv4_bias)
            h4 = tf.nn.elu(h4)

            print("h4",h4)

            h4 = tf.reshape(h4, [-1, 3*3*256])

            self.mu     = tf.matmul(h4,self.fc_weight_for_mu)     + self.fc_bias_for_mu
            self.logvar = tf.matmul(h4,self.fc_weight_for_logvar) + self.fc_bias_for_logvar
            self.sigma = tf.exp(self.logvar / 2.0)
            # self.epsilon = tf.random_normal([408, 2048],0.0,1.0)
            # print('self.logvar :',self.logvar)
            # print('self.epsilon : ',self.epsilon)
            # self.z = self.mu + self.sigma * self.epsilon

            return self.mu, self.sigma

    # Bernoulli MLP as decoder
    def _build_bernoulli_MLP_decoder(self,z, keep_prob):

        with tf.variable_scope("bernoulli_MLP_decoder"):
            self.fc_for_decoder   = generate_fc_weight(shape=[2048,3*3*256]   ,name='fc_decoder_weight')
            self.bias_for_decoder = generate_fc_bias(shape=[3*3*256]     ,name='bias_decoder_weight')
            self.dconv5_weight    = generate_conv2d_weight(shape=[5,5,128,256] ,name='conv5_weight_encode')
            self.dconv5_bias      = generate_conv2d_bias(shape=128            ,name='conv5_weight_bias')
            self.dconv6_weight    = generate_conv2d_weight(shape=[5,5,64,128] ,name='conv6_weight_encode')
            self.dconv6_bias      = generate_conv2d_bias(shape=64            ,name='conv6_weight_bias')
            self.dconv7_weight    = generate_conv2d_weight(shape=[4,4,32,64]  ,name='conv7_weight_encode')
            self.dconv7_bias      = generate_conv2d_bias(shape=32             ,name='conv7_weight_bias')
            self.dconv8_weight    = generate_conv2d_weight(shape=[4,4,3,32]   ,name='conv8_weight_encode')
            self.dconv8_bias      = generate_conv2d_bias(shape=3             ,name='conv8_weight_bias')

            self.decoder_params = [
                self.dconv5_weight  ,
                # self.dconv5_bias,
                self.dconv6_weight  ,
                # self.dconv6_bias,
                self.dconv7_weight  ,
                # self.dconv7_bias,
                self.dconv8_weight  ,
                # self.dconv8_bias,
                self.fc_for_decoder,
                self.bias_for_decoder
            ]

            h = (tf.matmul(z,self.fc_for_decoder) + self.bias_for_decoder)
            h = tf.reshape(h, [-1, 3, 3, 256])
            # h = tf.image.resize_images(h,size=(6,6),method=tf.image.ResizeMethod.BILINEAR)

            # h = tf.nn.elu(h)
            print("h0 : ",h)
            # h = tf.image.resize_images(h,size=(6,6),method=tf.image.ResizeMethod.BILINEAR)
            #
            h = tf.nn.conv2d_transpose(value=h, filter=self.dconv5_weight, output_shape=[408,11,11,128], strides=[1, 3, 3, 1], padding='VALID', name='deconv_5')
            # h = tf.nn.bias_add(h,self.dconv5_bias)
            h = tf.nn.elu(h)
            # h1 = tf.image.resize_images(h,size=(7,7),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # h1 = tf.nn.conv2d(h1, self.dconv5_weight,strides=[1,1,1,1],  padding='SAME')
            # h1 = tf.nn.elu(tf.nn.bias_add(h1, self.dconv5_bias))
            # print("h1 : ",h1)

            # h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name="dc_deconv2")
            h = tf.nn.conv2d_transpose(value=h, filter=self.dconv6_weight, output_shape=[408,21,21,64], strides=[1, 2, 2, 1], padding='SAME', name='deconv_6')
            # h = tf.nn.bias_add(h,self.dconv6_bias)
            # h2 = tf.image.resize_images(h1,size=(21,21),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # h = tf.nn.conv2d(h, self.dconv6_weight,strides=[1,2,2,1], padding='SAME')
            h = tf.nn.elu(h)
            # h = tf.nn.elu(tf.nn.bias_add(h, self.dconv6_bias))
            # print("h2 : ",h2)

            h = tf.nn.conv2d_transpose(value=h, filter=self.dconv7_weight, output_shape=[408,42,42,32], strides=[1, 2, 2, 1], padding='SAME', name='deconv_7')
            # # h = tf.nn.bias_add(h,self.dconv7_bias)
            # h = tf.nn.elu(h)
            # h3 = tf.image.resize_images(h2,size=(42,42),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            #h = tf.nn.conv2d(h, self.dconv7_weight,strides=[1,2,2,1], padding='SAME')
            h = tf.nn.elu(h)

        # h = tf.nn.elu(tf.nn.bias_add(h, self.dconv7_bias))
            # print("h3 : ",h3)

            h = tf.nn.conv2d_transpose(value=h, filter=self.dconv8_weight, output_shape=[408,84,84, 3],  strides=[1, 2, 2, 1], padding='SAME', name='deconv_8')
            # h = tf.nn.bias_add(h,self.dconv8_bias)
            # h = tf.nn.elu(h)
            # h4 = tf.image.resize_images(h3,size=(84,84),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            #h = tf.nn.conv2d(h, self.dconv8_weight,strides=[1,2,2,1], padding='SAME')
            decoded_image = tf.nn.sigmoid(h)
            # print("h4 : ",h4)


            # print("fuck",h)
            # decoded_image = tf.image.resize_images(h,size=(84,84),method=tf.image.ResizeMethod.BILINEAR)
            # decoded_image = tf.nn.sigmoid(h)

        return decoded_image

    # Gateway
    def _build_autoencoder(self,input_image, keep_prob):

        target_image = tf.stop_gradient(input_image)
        # encoding
        mu, sigma = self._build_gaussian_MLP_encoder(input_image,  keep_prob)

        # sampling by re-parameterization technique
        self.z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # decoding
        y = self._build_bernoulli_MLP_decoder(z=self.z,keep_prob=keep_prob)

        self.output_image = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

        # loss
        eps = 1e-6 # avoid taking log of zero

        # reconstruction loss
        self.r_loss = tf.reduce_sum(
            tf.square(input_image - self.output_image),
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

        loss = self.r_loss + self.kl_loss


        return self.output_image, self.z, loss


    def _prepare_update_op(self):
        self.OPT_A = tf.train.RMSPropOptimizer(0.001, name='RMSPropA')
        #
        # self.grads = [tf.clip_by_norm(item, 40) for item in tf.gradients(self.loss,
        #                                                 self.encoder_params+self.decoder_params)]
        # self.update_op = self.OPT_A.apply_gradients(list(zip(self.grads,
        #                                                 self.encoder_params + self.decoder_params)))
        self.update_op = self.OPT_A.minimize(self.loss)



    def update(self,images):
        self.sess.run(self.update_op,feed_dict ={self.input_image:images})
        loss = self.sess.run(self.loss,feed_dict ={self.input_image:images})
        return loss

    def get_decoded_image(self,image):
        decoded_image = self.sess.run(self.decoded_images,feed_dict={self.input_image:image})
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

        if i > 2000:
            input_image = []
            for i in range(0,n):
                input_image.append(env.get_image_state(i))
            input_image = np.array(input_image)
            print(input_image.shape)
            decoded_image = VAE.get_decoded_image(input_image)
            for i in range(0,n):
                # input_image = env.get_image_state(i)
                # decoded_image= VAE.get_decoded_image(input_image)[0]

                image_path = '/home/wyz/PycharmProjects/JOC-NET/data/images/decoded_images/'
                raw_name     = image_path +'raw_'+str(i)+'.jpeg'
                decoded_name = image_path + 'decoded_'+str(i)+'.jpeg'


                cv2.imwrite(raw_name     , input_image[i]*256.0)
                cv2.imwrite(decoded_name , decoded_image[i]*256.0)
            print("图像输出完毕！！！！")
            feature_dict = dict()
            for state_id in range(env.n_locations):
                # current_observation = env.h5_file['observation'][state_id]/255.0
                # current_observation = np.array(current_observation)
                # current_observation = cv2.resize(current_observation,(84,84))
                current_img = env.get_image_state(state_id)
                current_feature = VAE.get_encode_feature(current_img)
                feature = current_feature[0].tolist()
                feature_dict["State_"+str(state_id)] = feature
            # print('特征长度： ',len(feature))
            # file_path = '/home/wyz/PycharmProjects/JOC-NET/data/feature_encoded.csv'
            # df = pd.DataFrame(feature_dict)
            # df.to_csv(file_path)
            break











