import tensorflow as tf
import numpy as np
from Environment.env import *
from config.config import *
from random import random,randint,sample



weight_path = "/home/wyz/PycharmProjects/JOC-NET/weight/encode_weight"


def generate_fc_weight(shape, name):
    threshold = 1.0 / np.sqrt(shape[0])
    weight_matrix = tf.random_uniform(shape, minval=-threshold, maxval=threshold)
    weight = tf.Variable(weight_matrix, name=name)
    return weight

def generate_fc_bias(shape, name):
    # bias_distribution = np.zeros(shape)
    bias_distribution = tf.constant(0.0, shape=shape)
    bias = tf.Variable(bias_distribution, name=name)
    return bias

def generate_conv2d_weight(shape,name):
    weight = tf.Variable(np.random.rand(shape[0],shape[1],shape[2],shape[3]),dtype=np.float32,name=name)
    return weight

def generate_conv2d_bias(shape,name):
    bias = tf.Variable(np.random.rand(shape),dtype=np.float32,name=name)
    return bias

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def inverse_flatten(x,shape):
    return tf.reshape(x,shape )

class VAE_Encoder(object):

    def __init__(self):

        self.session = tf.Session()

        self.OPT = tf.train.RMSPropOptimizer(0.0001, name='RMSPropA')

        self.current_image = tf.placeholder(tf.float32,[None,84,84,3], 'Current_image')
        self.next_image    = tf.placeholder(tf.float32,[None,84,84,3], 'Next_image')
        self.action = tf.placeholder(tf.int32, [None, ], 'Action')

        self.lr = tf.placeholder(tf.float32,None,'learning_rate')

        self.lr_pa = tf.placeholder(tf.float32,None,'learning_rate_pa')

        self.OPT = tf.train.RMSPropOptimizer(self.lr, name='RMSPropA')

        self.OPT_PA = tf.train.RMSPropOptimizer(self.lr_pa, name='RMSPropA_PA')

        self._prepare_weight()

        # build_encoder
        self.current_state_feature = self._build_encode_net(input_image=self.current_image)
        # build_decoder
        self.next_state_feature = self._build_encode_net(input_image=self.next_image)

        #-------------------------current_state||next_state --> action  逆向动力学  预测两个状态之间的动作---------------------
        self.cur_next_concat = tf.concat([self.current_state_feature,self.next_state_feature], axis=1)
        self.action_predicted = tf.nn.softmax(tf.matmul(self.cur_next_concat, self.weight_p_a_1) + self.bias_p_a_1)
        # self.action_predicted = tf.nn.softmax(tf.matmul(self.action_predicted_temp, self.weight_p_a_2) + self.bias_p_a_2)
        self.action_predict_loss = -tf.reduce_sum(tf.one_hot(self.action,4,dtype=tf.float32)*
                                                   tf.log(tf.clip_by_value(self.action_predicted,1e-10,1.0)))
        self.action_predict_grads = [tf.clip_by_norm(item, 40) for item in
                                     tf.gradients(self.action_predict_loss,
                                                  self.action_predict_params + self.encode_params )]
        self.update_action_predict_op = self.OPT_PA.apply_gradients(list(zip(self.action_predict_grads,
                                                                             self.action_predict_params + self.encode_params )))


        #----------------------------------------current_state||action --> next_state---------------------------------------
        self.cur_action_concat = tf.concat([self.current_state_feature, tf.one_hot(self.action,4,dtype=tf.float32)], axis=1)
        self.state_feature_predict = (tf.matmul(self.cur_action_concat, self.weight_p_n_s_1) + self.bias_p_n_s_1)
        self.next_image_predicted = self._build_decode_net(input_feature=self.state_feature_predict)
        self.loss_raw = tf.subtract(self.next_image,self.next_image_predicted)
        self.state_predict_loss =  tf.reduce_sum(tf.square(self.loss_raw))
        self.state_predict_grads = [tf.clip_by_norm(item, 40) for item in tf.gradients(self.state_predict_loss,
                                                        self.state_predict_params + self.encode_params + self.decode_params)]
        self.update_state_predict_op = self.OPT.apply_gradients(list(zip(self.state_predict_grads,
                                                        self.state_predict_params + self.encode_params + self.decode_params)))

        #-----------------------------current_image --> current_feature --> current_image-------------------------------
        self.decoded_image = self._build_decode_net(input_feature=self.current_state_feature)
        self.auto_decoder_loss  =  tf.reduce_sum(tf.square(self.decoded_image - self.current_image))
        self.auto_decoder_grads = [tf.clip_by_norm(item, 40) for item in tf.gradients(self.auto_decoder_loss,
                                                                                      self.decode_params + self.encode_params)]
        self.update_decoder = self.OPT.apply_gradients(list(zip(self.auto_decoder_grads,
                                                                self.decode_params + self.encode_params)))

        self._prepare_store()

        self.session.run(tf.global_variables_initializer())


    def _prepare_weight(self):
        # encode
        self.fc1_weight = generate_fc_weight(shape=[84*84*3 , 2048]  ,name="fc1_weight")
        self.fc1_bias   = generate_fc_bias(shape=[2048],name='fc1_bias')
        # self.fc3_weight = generate_fc_weight(shape=[4096,2048],name="fc3_weight")
        # self.fc3_bias   = generate_fc_bias(shape=[2048] ,name='fc3_bias')
        self.encode_params = [
            self.fc1_weight  ,  self.fc1_bias,
            # self.fc3_weight  ,  self.fc3_bias,
        ]


        # decode weight
        # self.fc4_weight = generate_fc_weight(shape=[2048,4096]   ,name='fc4_weight')
        # self.fc4_bias   = generate_fc_bias(shape=[4096]          ,name='fc4_bias')
        self.fc6_weight = generate_fc_weight(shape=[2048,84*84*3],name='fc6_weight')
        self.fc6_bias   = generate_fc_bias(shape=[84*84*3]       ,name='fc6_bias')
        self.decode_params = [
            # self.fc4_weight  , self.fc4_bias,
            self.fc6_weight  , self.fc6_bias,
        ]


        # weight for predict action
        self.weight_p_a_1    = generate_fc_weight(shape=[2048*2,4],name='p_action_weight_1')
        self.bias_p_a_1      = generate_fc_bias(shape=[4]         ,name='p_action_bias_1')
        # self.weight_p_a_2    = generate_fc_weight(shape=[32,4]     ,name='p_action_weight_2')
        # self.bias_p_a_2      = generate_fc_bias(shape=[4]            ,name='p_action_bias_2')
        self.action_predict_params = [self.weight_p_a_1,self.bias_p_a_1,
                                      # self.weight_p_a_2,self.bias_p_a_2
                                      ]

        # current_state||action --> next_state  前向动力学  预测下一时刻的特质
        self.weight_p_n_s_1   = generate_fc_weight(shape=[2048+4,2048],name='p_n_state_weight_1')
        self.bias_p_n_s_1     = generate_fc_bias(shape=[2048]         ,name='p_n_state_bias_1')
        self.state_predict_params = [
            self.weight_p_n_s_1,self.bias_p_n_s_1,
        ]


    def _build_encode_net(self,input_image):

        flatten_feature = tf.reshape(input_image, [-1,84*84*3])
        fc1 = tf.nn.elu(tf.matmul(flatten_feature,self.fc1_weight) + self.fc1_bias)
        # fc3 = tf.nn.elu(tf.matmul(fc1,self.fc3_weight) + self.fc3_bias)
        return fc1

    def _build_decode_net(self,input_feature):
        # 4096 --> 4160
        # fc4 = tf.nn.elu(tf.matmul(input_feature, self.fc4_weight) + self.fc4_bias)
        fc6 = tf.nn.sigmoid(tf.matmul(input_feature,self.fc6_weight) + self.fc6_bias)
        decoded_image = tf.reshape(fc6, [-1,84,84,3])
        return decoded_image


    def update_state_action_predict_network(self,current_image,action,next_image,lr=0.0001,lr_pa=0.0001):
        self.session.run([
            self.update_action_predict_op,
            self.update_state_predict_op
        ],
            feed_dict={
                self.current_image  : current_image,
                self.next_image     : next_image,
                self.action         : action  ,
                self.lr             : lr,
                self.lr_pa          : lr_pa})



    def update_auto_decoder(self,current_image,lr=0.0001,lr_pa=0.0001):
        self.session.run([self.update_decoder],feed_dict = {
            self.current_image:current_image,
            self.lr:lr,
            self.lr_pa:lr_pa
        })
        loss = self.session.run(self.auto_decoder_loss,feed_dict ={self.current_image:current_image})
        return loss



    def get_loss(self,current_image,action,next_image):
        action_predicted,action_truth = self.session.run([self.action_predicted,self.action],feed_dict ={
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
        action_predict_loss,state_predict_loss ,auto_decoder_loss = self.session.run([
            self.action_predict_loss,self.state_predict_loss,self.auto_decoder_loss],
            feed_dict={
                self.current_image  : current_image,
                self.next_image     : next_image,
                self.action         : action     })
        acc = count/len(action_predicted)
        return action_predict_loss,state_predict_loss,auto_decoder_loss,acc


    def random_choice_action(self):
        action = np.random.choice(range(4),p=[0.25,0.25,0.25,0.25])
        return action

    def _prepare_store(self):
        var = tf.global_variables()
        var_to_restore = [val for val in var if 'fc' in val.name ]
        self.saver = tf.train.Saver(var_to_restore )

    def store(self,weight_path):
        self.saver.save(self.session,weight_path)

    def get_decoded_image(self,image):
        decoded_image = self.session.run(self.decoded_image,feed_dict={self.current_image:image[np.newaxis, :]})
        return  decoded_image

    def get_predicted_imge(self,image,action):
        predicted_image = self.session.run(self.next_image_predicted,feed_dict={self.current_image:image[np.newaxis, :],
                                                                                self.action:action})
        return predicted_image

    def get_feature(self,image):
        feature = self.session.run(self.current_state_feature,feed_dict={self.current_image:image[np.newaxis, :]})
        return feature


if __name__ == '__main__':

    env = load_thor_env(scene_name='bedroom_04',
                        random_start=True,
                        random_terminal=True,
                        terminal_id=None, start_id=None, num_of_frames=1)

    VAE = VAE_Encoder()

    n = env.n_locations # 408

    for i in range(100000):

        result = sample(range(n),128)
        current_data = [ env.get_image_state(id)for id in result]

        current_data = np.array(current_data)

        loss = VAE.update_auto_decoder(current_image=current_data).tolist()

        current_action_index = [ sample(range(4),1)[0] for i in range(128)]

        next_data = [env.get_next_state_image(current_state_id=j,action_id=i) for i,j in zip(current_action_index,result)]
        next_data = np.array(next_data)

        current_action_index = np.array(current_action_index)

        VAE.update_state_action_predict_network(current_image=current_data,action=current_action_index,next_image=next_data)

        action_predict_loss,state_predict_loss,auto_decoder_loss,acc = VAE.get_loss(current_image=current_data,
                                                                                    action = current_action_index,
                                                                                    next_image = next_data)

        print("Episode:%s   Auto Decoder Loss:%s   Action Predict Loss:%s  State Predict Loss:%s  Acc:%s"%
              (i,auto_decoder_loss,action_predict_loss,state_predict_loss,acc))

        if i > 10000:
            for i in range(0,128):
                input_image = env.get_image_state(i)
                decoded_image= VAE.get_decoded_image(input_image)[0]
                action_index = sample(range(4),1)[0]
                action = np.array([action_index])
                predicted_image = VAE.get_predicted_imge(image=input_image,action=action)[0]

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

            # VAE.store(weight_path)
            feature_dict = dict()
            for state_id in range(env.n_locations):
                # current_observation = env.h5_file['observation'][state_id]/255.0
                # current_observation = np.array(current_observation)
                # current_observation = cv2.resize(current_observation,(84,84))
                current_img = env.get_image_state(state_id)
                current_feature = VAE.get_feature(current_img)
                feature = current_feature[0].tolist()
                feature_dict["State_"+str(state_id)] = feature
            print('特征长度： ',len(feature))
            file_path = '/home/wyz/PycharmProjects/JOC-NET/data/feature_encoded.csv'
            df = pd.DataFrame(feature_dict)
            df.to_csv(file_path)

            break


