import tensorflow as tf
import numpy as np
from Environment.env import *
from config.config import *


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


class Encoder_Network(object):

    def __init__(self,sess):

        self.session = sess

        self.OPT = tf.train.RMSPropOptimizer(0.0001, name='RMSPropA')


        self.current_image = tf.placeholder(tf.float32,[None,300,400,3], 'Current_image')
        self.next_image = tf.placeholder(tf.float32,[None,300,400,3], 'Next_image')
        self.action = tf.placeholder(tf.int32, [None, ], 'Action')

        self.lr = tf.placeholder(tf.float32,None,'learning_rate')

        self.OPT = tf.train.RMSPropOptimizer(self.lr, name='RMSPropA')


        self.conv1_weight = generate_conv2d_weight(shape=[5,5,3,8],name="conv1_weight")
        self.conv1_bias   = generate_conv2d_bias(shape=8,name='conv1_bias')
        self.conv2_weight = generate_conv2d_weight(shape=[3,3,8,16],name="conv2_weight")
        self.conv2_bias   = generate_conv2d_bias(shape=16,name='conv2_bias')
        self.conv3_weight = generate_conv2d_weight(shape=[3,3,16,32],name="conv3_weight")
        self.conv3_bias   = generate_conv2d_bias(shape=32,name='conv3_bias')
        self.conv4_weight = generate_conv2d_weight(shape=[3,3,32,64],name="conv4_weight")
        self.conv4_bias   = generate_conv2d_bias(shape=64,name='conv4_bias')
        self.fc_weight    = generate_fc_weight(shape=[8320,2048],name='fc_weight')
        self.fc_bias      = generate_fc_weight(shape=[2048],name='fc_bias')
        self.encode_params = [self.conv1_weight,self.conv1_bias,self.conv2_weight,self.conv2_bias,
                              self.conv3_weight,self.conv3_bias,self.conv4_weight,self.conv4_bias,
                              self.fc_weight,self.fc_bias]

        self.weight_p_a_1    = generate_fc_weight(shape=[4096,1024],name='p_a_weight_1')
        self.bias_p_a_1      = generate_fc_weight(shape=[1024],name='p_a_bias_1')
        self.weight_p_a_2    = generate_fc_weight(shape=[1024,4],name='p_a_weight_2')
        self.bias_p_a_2      = generate_fc_weight(shape=[4],name='p_a_bias_2')
        self.action_predict_params = [self.weight_p_a_1,self.bias_p_a_1,
                                      self.weight_p_a_2,self.bias_p_a_2
                                      ]

        # current_state||action --> next_state  前向动力学  预测下一时刻的特质
        self.weight_p_n_s    = generate_fc_weight(shape=[2048+4,2048],name='p_ns_weight')
        self.bias_p_n_s      = generate_fc_weight(shape=[2048],name='p_ns__bias')
        self.state_predict_params = [self.weight_p_n_s,self.bias_p_n_s]

        self.current_feature = self._build_encode_net(input_image=self.current_image)
        self.next_feature    = self._build_encode_net(input_image=self.next_image)

        # current_state||next_state --> action  逆向动力学  预测两个状态之间的动作
        self.cur_next_concat = tf.concat([self.current_feature, self.next_feature], axis=1)
        self.action_predicted_temp = (tf.matmul(self.cur_next_concat, self.weight_p_a_1) + self.bias_p_a_1)
        self.action_predicted = tf.nn.softmax(tf.matmul(self.action_predicted_temp, self.weight_p_a_2) + self.bias_p_a_2)
        self.action_predict_loss = -tf.reduce_mean(tf.one_hot(self.action,4,dtype=tf.float32)
                                                   *tf.log(tf.clip_by_value(self.action_predicted,1e-10,1.0)))
        self.action_predict_grads = [tf.clip_by_norm(item, 40) for item in
                                     tf.gradients(self.action_predict_loss,
                                                  self.action_predict_params + self.encode_params )]
        self.update_action_predict_op = self.OPT.apply_gradients(list(zip(self.action_predict_grads,
                                                                self.action_predict_params + self.encode_params )))

        # current_state||action --> next_state
        self.cur_action_concat = tf.concat([self.current_feature, tf.one_hot(self.action,4,dtype=tf.float32)], axis=1)
        self.state_predict = tf.matmul(self.cur_action_concat, self.weight_p_n_s) + self.bias_p_n_s
        self.loss_raw = tf.subtract(tf.stop_gradient(self.next_feature),self.state_predict)
        self.state_predict_loss =  tf.reduce_mean(tf.square(self.loss_raw))
        self.state_predict_grads = [tf.clip_by_norm(item, 40) for item in tf.gradients(self.state_predict_loss,
                                                 self.state_predict_params + self.encode_params)]
        self.update_state_predict_op = self.OPT.apply_gradients(list(zip(self.state_predict_grads,
                                                                           self.state_predict_params + self.encode_params)))




    def _build_encode_net(self,input_image):
        conv1 = tf.nn.conv2d(input_image, self.conv1_weight, strides=[1, 4, 4, 1], padding='SAME')
        relu1 = tf.nn.elu(tf.nn.bias_add(conv1, self.conv1_bias))
        conv2 = tf.nn.conv2d(relu1, self.conv2_weight, strides=[1, 2, 2, 1], padding='SAME')
        relu2 = tf.nn.elu(tf.nn.bias_add(conv2, self.conv2_bias))
        conv3 = tf.nn.conv2d(relu2, self.conv3_weight, strides=[1, 2, 2, 1], padding='SAME')
        relu3 = tf.nn.elu(tf.nn.bias_add(conv3, self.conv3_bias))
        conv4 = tf.nn.conv2d(relu3, self.conv4_weight, strides=[1, 2, 2, 1], padding='SAME')
        relu4 = tf.nn.elu(tf.nn.bias_add(conv4,self.conv4_bias))
        flatten_feature = flatten(relu4)
        state_feature = tf.matmul(flatten_feature, self.fc_weight) + self.fc_bias
        return state_feature

    def update_state_action_predict_network(self,current_image,action,next_image,lr):
        self.session.run([self.update_action_predict_op,self.update_state_predict_op],
                         feed_dict={
                             self.current_image  : current_image,
                             self.next_image     : next_image,
                             self.action         : action  ,
                             self.lr             : lr})

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
        action_predict_loss,state_predict_loss = self.session.run([self.action_predict_loss,self.state_predict_loss],
                         feed_dict={
                             self.current_image  : current_image,
                             self.next_image     : next_image,
                             self.action         : action     })
        acc = count/len(action_predicted)
        return action_predict_loss,state_predict_loss,acc


    def random_choice_action(self):
        action = np.random.choice(range(4),p=[0.25,0.25,0.25,0.25])
        return action






class Worker_for_encoder(object):

    def __init__(self,sess):
        self.net = Encoder_Network(sess)
        self.env = load_thor_env(scene_name='bedroom_04', random_start=True, random_terminal=True,
                                        terminal_id=None, start_id=None, num_of_frames=1)

    def work(self):
        episode = 0
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
                if episode>400:
                    learning_rate = 0.00001
                else:
                    learning_rate = 0.0001
                if step%10 == 0 :
                    buffer_state, buffer_action, buffer_next_state =\
                        np.array(buffer_state),np.array(buffer_action),np.array(buffer_next_state)
                    self.net.update_state_action_predict_network(current_image=buffer_state,
                                                                 action=buffer_action,
                                                                 next_image=buffer_next_state,
                                                                 lr = learning_rate)
                    action_loss,state_loss,acc= self.net.get_loss(current_image=buffer_state,action=buffer_action,next_image=buffer_next_state)
                    # print(action_loss.shape)
                    # print(action_loss)
                    # print(np.sum(action_loss))
                    action_loss,state_loss = np.sum(action_loss),np.sum(state_loss)
                    print("Episode:%6s  ||  Steps:%4s  || State_predict_loss: %5s || Action_predict_loss: %5s || Accuracy:%s"%(
                        episode  ,step,round(state_loss,3),round(action_loss,3),acc
                    ))
                    buffer_state,buffer_action,buffer_next_state = [],[],[]
                current_image = current_image_next
                if step >= 500:
                    episode = episode + 1
                    break


if __name__ == '__main__':

    with tf.device(device):


        config = tf.ConfigProto(allow_soft_placement=True)

        SESS = tf.Session(config=config)

        worker = Worker_for_encoder(SESS)

        SESS.run(tf.global_variables_initializer())

        worker.work()
