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
            self.current_image =  tf.placeholder(tf.float32,[None,84,84,3], 'Current_image')
            self.next_image =  tf.placeholder(tf.float32,[None,84,84,3], 'Next_image')
            self.action = tf.placeholder(tf.int32, [None, ], 'Action')

            self._build_autoencoder(keep_prob=kp)
            self._prepare_loss_for_autoencoder()
            self._prepare_state_action_predict_loss()
            self._prepare_update_op()
            self.sess.run(tf.global_variables_initializer())



    # Gaussian MLP as encoder
    def _build_gaussian_MLP_encoder(self,input_image, keep_prob):
        with tf.variable_scope("gaussian_MLP_encoder"):
            h1 = tf.layers.conv2d(input_image, 32, 8, strides=2, activation=tf.nn.elu, name="encoder_conv1")
            h2 = tf.layers.conv2d(h1,          64, 6, strides=2, activation=tf.nn.elu, name="encoder_conv2")
            h3 = tf.layers.conv2d(h2,         128, 4, strides=2, activation=tf.nn.elu, name="encoder_conv3")
            h4 = tf.layers.conv2d(h3,         256, 3, strides=2, activation=tf.nn.elu, name="encoder_conv4")
            h_a = tf.reshape(h4, [-1, 3*3*256])

            self.mu     = tf.layers.dense(h_a, 2048, name="encoder_fc_mu")
            self.logvar = tf.layers.dense(h_a, 2048, name="encoder_fc_log_var")
            self.sigma = tf.exp(self.logvar / 2.0)

            z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)

            return z

    # Bernoulli MLP as decoder
    def _build_bernoulli_MLP_decoder(self,z, keep_prob):

        with tf.variable_scope("bernoulli_MLP_decoder"):

            h = tf.layers.dense(z, 3*3*256, name="decoder_fc")
            h = tf.reshape(h, [-1, 3, 3, 256])

            h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.elu, name="decoder_deconv1")  # 9, 9, 128

            h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.elu, name="decoder_deconv2")   #  21, 21, 64

            h = tf.layers.conv2d_transpose(h, 32, 4, strides=2, activation=tf.nn.elu, name="decoder_deconv3")   #  44, 44, 32

            h = tf.layers.conv2d_transpose(h, 3, 4, strides=2, activation=tf.nn.elu, name="decoder_deconv4")    # 90, 90, 3

            decoded_image = tf.image.resize_images(h,size=(84,84),method=tf.image.ResizeMethod.BILINEAR)
            decoded_image = tf.nn.sigmoid(decoded_image)
        return decoded_image

    # Gateway
    def _build_autoencoder(self, keep_prob):

        # encoding
        with tf.variable_scope('encoder'):
            self.current_z = self._build_gaussian_MLP_encoder(self.current_image,  keep_prob)
        with tf.variable_scope('encoder',reuse=True):
            self.next_z = self._build_gaussian_MLP_encoder(self.next_image,  keep_prob)


        # predict next state
        self.cur_action_concat = tf.concat([self.current_z, tf.one_hot(self.action,4,dtype=tf.float32)], axis=1)
        self.z_for_next_state = tf.layers.dense(self.cur_action_concat,2048,name="next_z_predictor")


        # predict the action between two state
        self.state_coacat =  tf.concat([self.current_z, self.next_z], axis=1)
        action_predict = tf.layers.dense(self.state_coacat,4,name='action_predictor')
        self.action_predicted = tf.nn.softmax(action_predict)


        # decoding
        with tf.variable_scope('decoder'):
            self.current_decoded_image = self._build_bernoulli_MLP_decoder(z=self.current_z,keep_prob=keep_prob)

        with tf.variable_scope('decoder',reuse=True):
            self.next_decoded_image = self._build_bernoulli_MLP_decoder(z=self.z_for_next_state,keep_prob=keep_prob)

        self.current_decoded_image = tf.clip_by_value(self.current_decoded_image, 1e-8, 1 - 1e-8)
        self.next_decoded_image    = tf.clip_by_value(self.next_decoded_image, 1e-8, 1 - 1e-8)



    def _prepare_loss_for_autoencoder(self):
        # reconstruction loss
        self.r_loss = tf.reduce_sum(
            tf.square(self.current_image - self.current_decoded_image),reduction_indices = [1,2,3])
        self.r_loss = tf.reduce_mean(self.r_loss)

        # augmented kl loss per dim
        self.kl_loss = - 0.5 * tf.reduce_sum(
            (1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)),
            reduction_indices = 1)

        self.kl_loss = tf.maximum(self.kl_loss, 0.5 * 2048)
        self.kl_loss = tf.reduce_mean(self.kl_loss)

        self.auto_loss = self.r_loss + self.kl_loss

    def _prepare_state_action_predict_loss(self):
        self.action_predict_loss = \
            -tf.reduce_mean(tf.one_hot(self.action,4,dtype=tf.float32)*
                            tf.log(tf.clip_by_value(self.action_predicted,1e-10,1.0)))

        state_predict_loss = tf.reduce_sum(
            tf.square(self.next_decoded_image - self.next_image),reduction_indices = [1,2,3])
        self.state_predict_loss = tf.reduce_mean(state_predict_loss)



    def _prepare_update_op(self):
        self.loss = self.auto_loss + self.action_predict_loss + self.state_predict_loss
        self.OPT_A = tf.train.RMSPropOptimizer(0.0001, name='RMSPropA')
        self.update_op = self.OPT_A.minimize(self.loss)

    def update(self,current_image,next_image,action):
        self.sess.run(self.update_op,feed_dict ={self.current_image:current_image,
                                                 self.next_image:next_image,
                                                 self.action:action})
        r_loss,state_loss,action_loss = self.sess.run([self.r_loss,self.state_predict_loss,self.action_predict_loss]
                             ,feed_dict ={self.current_image:current_image,
                                          self.next_image:next_image,
                                          self.action:action})

        action_predicted,action_truth = self.sess.run([self.action_predicted,self.action],feed_dict ={
            self.current_image  : current_image,
            self.next_image     : next_image,
            self.action         : action     } )
        count = 0
        for i in range(len(action_predicted)):
            a_p_l = action_predicted[i].tolist()
            a_p = a_p_l.index(max(a_p_l))
            a_t = action_truth[i]
            if int(a_p) == int(a_t):
                count = count + 1
        acc = count/len(action_predicted)
        return r_loss,state_loss,action_loss,acc


    def get_decoded_image(self,current_image):
        decoded_image = self.sess.run(self.current_decoded_image,feed_dict={self.current_image:current_image[np.newaxis, :]})
        return  decoded_image

    def get_predicted_image(self,current_image,action):
        predicted_image = self.sess.run(self.next_decoded_image,feed_dict={self.current_image : current_image[np.newaxis, :],
                                                                           self.action        : action})
        return predicted_image

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

        current_id = sample(range(n),408)

        current_image = np.array([ env.get_image_state(id)for id in current_id])

        current_action_index_list = [ sample(range(4),1)[0] for i in range(408)]
        current_action_index = np.array(current_action_index_list)

        next_image = np.array([env.get_next_state_image(current_state_id=j,action_id=i)
                              for i,j in zip(current_action_index_list,current_id)])

        auto_loss,state_loss,action_loss,acc = VAE.update(current_image=current_image,next_image=next_image,action=current_action_index)

        print("Episode:%s  ||  Auto_loss:%s  || State_Loss :%s ||  Action_Loss: %s || Acc : %s"%
              (i,auto_loss,state_loss,action_loss,acc))

        if i > 5000:
            for i in range(0,408):
                input_image = env.get_image_state(i)
                decoded_image= VAE.get_decoded_image(input_image)[0]
                action_index = sample(range(4),1)[0]
                action = np.array([action_index])
                predicted_image = VAE.get_predicted_image(current_image=input_image,action=action)[0]

                image_path = '/home/wyz/PycharmProjects/JOC-NET/data/images/decoded_images/'
                raw_name     = image_path +'raw_'+str(i)+'.jpeg'
                decoded_name = image_path + 'decoded_'+str(i)+'.jpeg'
                cv2.imwrite(raw_name       , input_image*256.0)
                cv2.imwrite(decoded_name   , decoded_image*256.0)

                truth = env.get_next_state_image(current_state_id=i,action_id=action_index)
                predicted_path = '/home/wyz/PycharmProjects/JOC-NET/data/images/predicted_images/'
                predicted_result_name = predicted_path + 'predicted_' + str(i) +'.jpeg'
                predicted_truth_name  = predicted_path + 'truth_' + str(i) +'.jpeg'
                cv2.imwrite(predicted_result_name , predicted_image*256.0)
                cv2.imwrite(predicted_truth_name , truth*256.0)

            for item in tf.trainable_variables():
                print(item)

            # feature_dict = dict()
            # for state_id in range(env.n_locations):
            #     # current_observation = env.h5_file['observation'][state_id]/255.0
            #     # current_observation = np.array(current_observation)
            #     # current_observation = cv2.resize(current_observation,(84,84))
            #     current_img = env.get_image_state(state_id)
            #     current_feature = VAE.get_encode_feature(current_img)
            #     feature = current_feature[0].tolist()
            #     feature_dict["State_"+str(state_id)] = feature
            # print('特征长度： ',len(feature))
            # file_path = '/home/wyz/PycharmProjects/JOC-NET/data/feature_encoded.csv'
            # df = pd.DataFrame(feature_dict)
            # df.to_csv(file_path)
            break











