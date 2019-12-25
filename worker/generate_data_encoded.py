from Environment.env import *
from tensorflow.python import pywrap_tensorflow
from model.model_op import *
import numpy as np
import pandas as pd
import cv2



def get_weight(encoder_weight_dict,name):
    return  encoder_weight_dict[name]

def prepare_encoder_weight(encoder_weight_dict):
    fc1_weight = get_weight(encoder_weight_dict,name="fc1_weight")
    fc1_bias   = get_weight(encoder_weight_dict,name='fc1_bias')
    fc2_weight = get_weight(encoder_weight_dict,name="fc3_weight")
    fc2_bias   = get_weight(encoder_weight_dict,name='fc3_bias')
    return fc1_weight,fc1_bias,fc2_weight,fc2_bias

encoder_weight_path = "/home/wyz/PycharmProjects/JOC-NET/weight/encode_weight1.0.ckpt"

model_reader = pywrap_tensorflow.NewCheckpointReader(encoder_weight_path)
var_dict = model_reader.get_variable_to_shape_map()
result_dict = dict()
for key in var_dict:
    if 'encode' in key.split('/')[-1]:
        result_dict[key.split('/')[-1]] = model_reader.get_tensor(key)
encoder_weight_dict = result_dict



fc1_weight,fc1_bias,fc3_weight,fc3_bias= prepare_encoder_weight(encoder_weight_dict)

input = tf.placeholder(tf.float32,[None,SCREEN_HEIGHT,SCREEN_WIDTH,3], 'State_image')
flatten_feature = tf.reshape(input, [-1,84*84*3])
fc1 = tf.nn.elu(tf.matmul(flatten_feature,fc1_weight) + fc1_bias)
state_feature = tf.nn.elu(tf.matmul(fc1,fc3_weight) + fc3_bias)

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






