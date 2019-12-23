import tensorflow as tf
import numpy as np
from Environment.env import *
from config.config import *


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

class Encoder_Network(object):

    def __init__(self,sess,scope):

        with tf.name_scope(scope):

            self.session = sess

            self.OPT = tf.train.RMSPropOptimizer(0.0001, name='RMSPropA')

            self.current_image = tf.placeholder(tf.float32,[None,SCREEN_WIDTH,SCREEN_HEIGHT,3], 'Current_image')
            self.next_image    = tf.placeholder(tf.float32,[None,SCREEN_WIDTH,SCREEN_HEIGHT,3], 'Next_image')
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
            self.action_predicted_temp = tf.nn.elu(tf.matmul(self.cur_next_concat, self.weight_p_a_1) + self.bias_p_a_1)
            self.action_predicted = tf.nn.softmax(tf.matmul(self.action_predicted_temp, self.weight_p_a_2) + self.bias_p_a_2)
            self.action_predict_loss = -tf.reduce_mean(tf.one_hot(self.action,4,dtype=tf.float32)*
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
            self.state_predict_loss =  tf.reduce_mean(tf.square(self.loss_raw))
            self.state_predict_grads = [tf.clip_by_norm(item, 40) for item in tf.gradients(self.state_predict_loss,
                                         self.state_predict_params + self.encode_params + self.decode_params)]
            self.update_state_predict_op = self.OPT.apply_gradients(list(zip(self.state_predict_grads,
                                         self.state_predict_params + self.encode_params + self.decode_params)))

            #-----------------------------current_image --> current_feature --> current_image-------------------------------
            self.decoded_image = self._build_decode_net(input_feature=self.current_state_feature)
            self.auto_decoder_loss  =  tf.reduce_mean(tf.square(self.decoded_image - self.current_image))
            self.auto_decoder_grads = [tf.clip_by_norm(item, 40) for item in tf.gradients(self.auto_decoder_loss,
                                                                    self.decode_params + self.encode_params)]
            self.update_decoder = self.OPT.apply_gradients(list(zip(self.auto_decoder_grads,
                                                                    self.decode_params + self.encode_params)))

            self._prepare_store()

    def _prepare_weight(self):
        # encode
        self.fc1_weight = generate_fc_weight(shape=[84*84*3 , 4096*2]  ,name="fc1_weight")
        self.fc1_bias   = generate_fc_bias(shape=[4096*2],name='fc1_bias')
        self.fc2_weight = generate_fc_weight(shape=[4096*2,4096] ,name="fc2_weight")
        self.fc2_bias   = generate_fc_bias(shape=[4096] ,name='fc2_bias')
        self.fc3_weight = generate_fc_weight(shape=[4096,2048],name="fc3_weight")
        self.fc3_bias   = generate_fc_bias(shape=2048 ,name='fc3_bias')
        self.encode_params = [
            self.fc1_weight  ,  self.fc1_bias,
            self.fc2_weight  ,  self.fc2_bias,
            self.fc3_weight  ,  self.fc3_bias,
        ]


        # decode weight
        self.fc4_weight = generate_fc_weight(shape=[2048,4096]   ,name='fc4_weight')
        self.fc4_bias   = generate_fc_bias(shape=[4096]          ,name='fc4_bias')
        self.fc5_weight = generate_fc_weight(shape=[4096,4096*2] ,name='fc5_weight')
        self.fc5_bias   = generate_fc_bias(shape=[4096*2]        ,name='fc5_bias')
        self.fc6_weight = generate_fc_weight(shape=[4096,84*84*3],name='fc6_weight')
        self.fc6_bias   = generate_fc_bias(shape=[84*84*3]       ,name='fc6_bias')
        self.decode_params = [
            self.fc4_weight  , self.fc4_bias,
            self.fc5_weight  , self.fc5_bias,
            self.fc6_weight  , self.fc6_bias,
        ]


        # weight for predict action
        self.weight_p_a_1    = generate_fc_weight(shape=[2048*2,1024],name='p_action_weight_1')
        self.bias_p_a_1      = generate_fc_bias(shape=[1024]         ,name='p_action_bias_1')
        self.weight_p_a_2    = generate_fc_weight(shape=[1024,4]     ,name='p_action_weight_2')
        self.bias_p_a_2      = generate_fc_bias(shape=[4]            ,name='p_action_bias_2')
        self.action_predict_params = [self.weight_p_a_1,self.bias_p_a_1,
                                      self.weight_p_a_2,self.bias_p_a_2
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
        fc2 = tf.nn.elu(tf.matmul(fc1,self.fc2_weight) + self.fc2_bias)
        fc3 = tf.nn.elu(tf.matmul(fc3,self.fc3_weight) + self.fc3_bias)
        return fc3

    def _build_decode_net(self,input_feature):
        # 4096 --> 4160
        fc4 = tf.nn.elu(tf.matmul(input_feature, self.fc4_weight) + self.fc4_bias)
        fc5 = tf.nn.elu(tf.matmul(fc4,self.fc5_weight) + self.fc5_bias)
        fc6 = tf.nn.sigmoid(tf.matmul(fc5,self.fc6_weight) + self.fc6_bias)       
        decoded_image = tf.reshape(fc6, [-1,84,84,3])

        return decoded_image


    def update_state_action_predict_network(self,current_image,action,next_image,lr,lr_pa):
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



    def update_auto_decoder(self,current_image,lr,lr_pa):
        self.session.run([self.update_decoder],feed_dict = {
            self.current_image:current_image,
            self.lr:lr,
            self.lr_pa:lr_pa
        })



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
        var_to_restore = [val for val in var if '_encode' in val.name ]
        self.saver = tf.train.Saver(var_to_restore )

    def store(self,weight_path):
        self.saver.save(self.session,weight_path)



class Worker_for_encoder(object):

    def __init__(self,sess):
        self.net = Encoder_Network(sess,'Worker')
        self.env = load_thor_env(scene_name='bedroom_04',
                                 random_start=True,
                                 random_terminal=True,
                                 terminal_id=None, start_id=None, num_of_frames=1)

    def work(self):
        episode = 0
        s_loss,accuracy = 0,0.9
        while episode <= 100000:
            current_image, _ = self.env.reset_env()
            step = 0
            buffer_state,buffer_action,buffer_next_state = [],[],[]
            while True:
                action = self.net.random_choice_action()
                current_image_next,_,_,_ = self.env.take_action(action)
                step = step + 1
                buffer_state.append(current_image)
                buffer_action.append(action)
                buffer_next_state.append(current_image_next)
                # if episode>300:
                #     learning_rate = 0.00001
                #     learning_rate_for_pa = 0.00005
                # else:
                learning_rate = 0.0001
                learning_rate_for_pa = 0.0001

                if step%128 == 0 :
                    buffer_state, buffer_action, buffer_next_state =\
                        np.array(buffer_state),np.array(buffer_action),np.array(buffer_next_state)
                    self.net.update_state_action_predict_network(current_image=buffer_state,
                                                                 action=buffer_action,
                                                                 next_image=buffer_next_state,
                                                                 lr = learning_rate,
                                                                 lr_pa=learning_rate_for_pa)
                    # self.net.update_auto_decoder(current_image=buffer_state,lr=learning_rate,lr_pa=learning_rate_for_pa)
                    action_loss,state_loss,decoder_loss,acc= self.net.get_loss(current_image = buffer_state,
                                                                               action        = buffer_action,
                                                                               next_image    = buffer_next_state)
                    buffer_state,buffer_action,buffer_next_state = [],[],[]
                    if episode > 4000 and acc >0.5:
                        self.net.store(weight_path+str(acc)+'.ckpt')
                        print("store! Acc:%s"%acc)
                        episode = 99999999
                    # print(action_loss.shape)
                    # print(action_loss)
                    # print(np.sum(action_loss))
                    action_loss,state_loss = np.sum(action_loss),np.sum(state_loss)
                    print("Episode:%6s  ||  Steps:%4s  || State_predict_loss: %5s || Action_predict_loss: %5s || Decoder_Loss:%5s ||Accuracy:%s  "%(
                        episode ,step,round(state_loss,3),round(action_loss,3),round(decoder_loss,3),round(acc,3)
                    ))
                current_image = current_image_next
                if step >= 1000:
                    episode = episode + 1
                    break


if __name__ == '__main__':

    with tf.device(device):


        config = tf.ConfigProto(allow_soft_placement=True)

        SESS = tf.Session(config=config)

        worker = Worker_for_encoder(SESS)

        SESS.run(tf.global_variables_initializer())

        worker.work()
