# -*- coding: utf-8 -*-
import sys
import h5py
import json
import numpy as np
import random
import skimage.io
from skimage.transform import resize
import tensorflow as tf
from config.config import *
import cv2
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片


SCREEN_WIDTH = 84
SCREEN_HEIGHT = 84
HISTORY_LENGTH = 4
ACTION_SIZE = 4 # action size


# (300, 400, 3)


class THORDiscreteEnvironment(object):

    def __init__(self, config=dict()):

        # configurations
        self.scene_name          = config.get('scene_name', 'bedroom_04')

        self.random_start        = config.get('random_start', True)
        self.random_terminal     = config.get('random_terminal',False)

        self.terminal_state_id   = config.get('terminal_state_id', 406)
        self.start_state_id      = config.get('start_state_id',0)
        self.whe_show_observation= config.get('whether_show' ,False)
        self.h5_file_path = config.get('h5_file_path', 'data/%s.h5'%self.scene_name)
        self.h5_file     = h5py.File(self.h5_file_path, 'r')
        self.locations   = self.h5_file['location'][()]
        self.rotations   = self.h5_file['rotation'][()]
        self.n_locations = self.locations.shape[0]

        self.transition_graph = self.h5_file['graph'][()]
        #print(self.transition_graph)
        self.shortest_path_distances = self.h5_file['shortest_path_distance'][()]


    def reset_env(self):
        #  choose terminal_state_id
        if self.random_terminal:
            self.terminal_state_id = random.sample(TARGET_ID_LIST,1)[0]
        #  choose start_state_id
        if self.random_start:
            while True:
                random_start_id = random.randrange(self.n_locations)
                dist = self.shortest_path_distances[random_start_id][self.terminal_state_id]
                if dist > 0 and random_start_id!=self.terminal_state_id:
                    break
            self.start_state_id = random_start_id
        else:
            pass
        self.current_state_id = self.start_state_id
        # reset parameters
        self.short_dist = self.shortest_path_distances[self.start_state_id][self.terminal_state_id]
        if self.short_dist<0:
            raise NameError('The distance between start and terminal must large than 0 ! Please Check env_loader! ')
        self.reward   = 0
        self.collided = False
        self.terminal = False
        # self.memory   = np.zeros([M_SIZE,2048])
        self.step_count = 0
        return self.state,self.terminal_state


    def take_action(self, action):
        #assert not self.terminal  , 'step() called in terminal_state'
        currrent_id = self.current_state_id
        if self.transition_graph[self.current_state_id][action] != -1:
            self.collided = False
            self.current_state_id = self.transition_graph[self.current_state_id][action].reshape([-1])[0]
            next_id = self.current_state_id
            # print("self.terminal_state_id:%3s  ||  self.current_state_id:%3s"%(self.terminal_state_id,self.current_state_id))
            if self.terminal_state_id == self.current_state_id:
                self.terminal = True
            else:
                self.terminal = False
        else:
            self.terminal = False
            self.collided = True
            next_id = self.current_state_id
        reward = self.reward_env(self.terminal, self.collided)
        self.step_count = self.step_count + 1
        return self.state,reward,self.terminal,(currrent_id,next_id)


    def get_id_state(self,id):
        return self.h5_file['resnet_feature'][id][0][:,np.newaxis].reshape([1,-1])[0]

    # s_t = s_t1
    def update(self):
        self.s_t = self.s_t1
        return

    # private methods

    def _tiled_state(self, state_id):
        f = self.h5_file['resnet_feature'][state_id][0][:,np.newaxis]
        return np.tile(f, (1, self.history_length))

    def reward_env(self, terminal, collided):
        if terminal:
            return 10.0
        if collided:
            return -0.1
        else:
            return -0.01

    # properties

    @property
    def action_size(self):
        # move forward/backward, turn left/right for navigation
        return ACTION_SIZE

    @property
    def action_definitions(self):
        action_vocab = ["Forward", "Right", "Left", "Backward"]
        return action_vocab[:ACTION_SIZE]

    @property
    def current_distance(self):
        distance = self.shortest_path_distances[self.current_state_id][self.terminal_state_id]
        return distance

    @property
    def observation(self):
        return self.h5_file['observation'][self.current_state_id]


    @property
    def state(self):
        # read from hdf5 cache
        current_state = self.observation/255.0
        return current_state


    @property
    def terminal_state(self):
        terminal_state = self.h5_file['observation'][self.terminal_state_id]/255.0
        return terminal_state




def load_thor_env(scene_name,random_start,random_terminal,
                  num_of_frames,start_id=None,
                  terminal_id=None):
    if random_start == False and start_id is None :
        raise NameError('If you want start random,please enter a Start_ID!')
        return
    if random_terminal == False and terminal_id is None:
        raise NameError('If you need random terminal,please enter a Ternimal_ID!')
        return

    config = {
        'terminal_state_id':terminal_id,
        'start_state_id'   :start_id,
        'scene_name':scene_name,
        'random_start':random_start,
        'random_terminal': random_terminal,
        'terminal_state_id': terminal_id,
        'start_state_id':start_id,
        'number_of_frames': num_of_frames,
        #'h5_file_path': ROOT_PATH + '/data/%s.h5'%scene_name,
        'h5_file_path': DATA_PATH
        #-'h5_file_path': '/data1/wyz/PyProject/memory-based-visual-navigation'+'/data/%s.h5'%scene_name, # data path for my ubuntu server

    }
    env = THORDiscreteEnvironment(config)
    return env


def get_dim(self):
    return 4,2048



# if __name__ == "__main__":
#
#     env = load_thor_env(scene_name='bedroom_04', random_start=True, random_terminal=True,
#                          terminal_id=True, start_id=True,
#                          num_of_frames=1)
#
#
#     cur_state,tar_state = env.reset_env()
#
#     from utils.op import *
#
#     def flatten(x):
#         return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
#
#     image_input = tf.placeholder(tf.float32,[None,300,400,3],name='test_image')
#
#     conv1_weight = generate_conv2d_weight(shape=[5,5,3,8],name="conv1_weight")
#     conv1_bais   = generate_conv2d_bias(shape=8,name='conv1_bias')
#     conv2_weight = generate_conv2d_weight(shape=[3,3,8,16],name="conv2_weight")
#     conv2_bais   = generate_conv2d_bias(shape=16,name='conv2_bias')
#     conv3_weight = generate_conv2d_weight(shape=[3,3,16,32],name="conv3_weight")
#     conv3_bais   = generate_conv2d_bias(shape=32,name='conv3_bias')
#     conv4_weight = generate_conv2d_weight(shape=[3,3,32,64],name="conv4_weight")
#     conv4_bais   = generate_conv2d_bias(shape=64,name='conv4_bias')
#     fc_weight    = generate_fc_weight(shape=[8320,2048],name='fc_weight')
#     fc_bias      = generate_fc_weight(shape=[2048],name='fc_bias')
#
#
#     conv1 = tf.nn.conv2d(image_input, conv1_weight, strides=[1, 4, 4, 1], padding='SAME')
#     relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bais))
#     conv2 = tf.nn.conv2d(relu1, conv2_weight, strides=[1, 2, 2, 1], padding='SAME')
#     relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bais))
#     conv3 = tf.nn.conv2d(relu2, conv3_weight, strides=[1, 2, 2, 1], padding='SAME')
#     relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_bais))
#     conv4 = tf.nn.conv2d(relu3, conv4_weight, strides=[1, 2, 2, 1], padding='SAME')
#     relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_bais))
#     flatten_feature = flatten(relu4)
#     output_feature = s_encode = tf.nn.elu(tf.matmul(flatten_feature, fc_weight) + fc_bias)
#
#
#
#
#
# with tf.Session() as sess:
#
#         tf.global_variables_initializer().run()
#
#         output_feature = sess.run(output_feature,feed_dict={image_input:tar_state[np.newaxis, :]})
#
#         print(output_feature.shape)
