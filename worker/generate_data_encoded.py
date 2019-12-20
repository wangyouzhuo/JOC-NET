from Environment.env import *
from tensorflow.python import pywrap_tensorflow
from model.model_op import *
import numpy as np
import pandas as pd
import cv2



def get_weight(encoder_weight_dict,name):
    return  encoder_weight_dict[name]

def prepare_encoder_weight(encoder_weight_dict):
    conv1_weight = get_weight(encoder_weight_dict,name="conv1_weight_encode")
    conv1_bias   = get_weight(encoder_weight_dict,name='conv1_bias_encode')
    conv2_weight = get_weight(encoder_weight_dict,name="conv2_weight_encode")
    conv2_bias   = get_weight(encoder_weight_dict,name='conv2_bias_encode')
    conv3_weight = get_weight(encoder_weight_dict,name="conv3_weight_encode")
    conv3_bias   = get_weight(encoder_weight_dict,name='conv3_bias_encode')
    conv4_weight = get_weight(encoder_weight_dict,name="conv4_weight_encode")
    conv4_bias   = get_weight(encoder_weight_dict,name='conv4_bias_encode')
    fc_weight    = get_weight(encoder_weight_dict,name='fc_weight_encode')
    fc_bias      = get_weight(encoder_weight_dict,name='fc_bias_encode')
    return conv1_weight,conv1_bias,\
           conv2_weight,conv2_bias,\
           conv3_weight,conv3_bias,\
           conv4_weight,conv4_bias, \
           fc_weight   ,fc_bias

encoder_weight_path = "/home/wyz/PycharmProjects/JOC-NET/weight/encode_weight1.0.ckpt"

model_reader = pywrap_tensorflow.NewCheckpointReader(encoder_weight_path)
var_dict = model_reader.get_variable_to_shape_map()
result_dict = dict()
for key in var_dict:
    if 'encode' in key.split('/')[-1]:
        result_dict[key.split('/')[-1]] = model_reader.get_tensor(key)
encoder_weight_dict = result_dict

conv1_weight,conv1_bias,\
conv2_weight,conv2_bias,\
conv3_weight,conv3_bias,\
conv4_weight,conv4_bias,\
fc_weight   ,fc_bias \
    = prepare_encoder_weight(encoder_weight_dict)

input = tf.placeholder(tf.float32,[None,SCREEN_HEIGHT,SCREEN_WIDTH,3], 'State_image')

conv1 = tf.nn.conv2d(input, conv1_weight, strides=[1, 2, 2, 1], padding='SAME')
# conv1_pooling = tf.layers.max_pooling2d(conv1,(2,2),(2,2),padding='same')
elu1 = tf.nn.elu(tf.nn.bias_add(conv1, conv1_bias))

conv2 = tf.nn.conv2d(elu1, conv2_weight, strides=[1, 2, 2, 1], padding='SAME')
# conv2_pooling = tf.layers.max_pooling2d(conv2,(2,2),(2,2),padding='same')
elu2 = tf.nn.elu(tf.nn.bias_add(conv2, conv2_bias))

conv3 = tf.nn.conv2d(elu2, conv3_weight, strides=[1, 2, 2, 1], padding='SAME')
elu3 = tf.nn.elu(tf.nn.bias_add(conv3, conv3_bias))

conv4 = tf.nn.conv2d(elu3, conv4_weight, strides=[1, 2, 2, 1], padding='SAME')
relu4 = tf.nn.elu(tf.nn.bias_add(conv4,conv4_bias))

flatten_feature = flatten(relu4)
print('flatten_feature:',flatten_feature.shape)
print('fc_weight:',fc_weight.shape)
# print('flatten_feature:',flatten_feature)

state_feature = (tf.matmul(flatten_feature,fc_weight) + fc_bias)




env = load_thor_env(scene_name='bedroom_04', random_start=True, random_terminal=True,
                         terminal_id=None, start_id=None, num_of_frames=1)



SESS = tf.Session()

feature_dict = dict()

for state_id in range(env.n_locations):
    current_observation = env.h5_file['observation'][state_id]/255.0
    current_observation = np.array(current_observation)
    current_observation = cv2.resize(current_observation,(84,84))
    current_feature = SESS.run(state_feature,feed_dict={input : current_observation[np.newaxis,:]})
    feature = current_feature[0].tolist()
    feature_dict["State_"+str(state_id)] = feature

print('特征长度： ',len(feature))

file_path = '/home/wyz/PycharmProjects/JOC-NET/data/feature_encoded.csv'

df = pd.DataFrame(feature_dict)

df.to_csv(file_path)






