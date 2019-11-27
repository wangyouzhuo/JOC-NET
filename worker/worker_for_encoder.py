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

            self.current_image = tf.placeholder(tf.float32,[None,300,400,3], 'Current_image')
            self.next_image    = tf.placeholder(tf.float32,[None,300,400,3], 'Next_image')
            self.action = tf.placeholder(tf.int32, [None, ], 'Action')

            self.lr = tf.placeholder(tf.float32,None,'learning_rate')

            self.lr_pa = tf.placeholder(tf.float32,None,'learning_rate_pa')

            self.OPT = tf.train.RMSPropOptimizer(self.lr, name='RMSPropA')

            self.OPT_PA = tf.train.RMSPropOptimizer(self.lr_pa, name='RMSPropA_PA')

            self._prepare_weight()

            self.current_flatten_feature,self.current_state_feature,self.current_feature_map \
                = self._build_encode_net(input_image=self.current_image)
            self.next_flatten_feature,self.next_state_feature,self.next_feature_map\
                = self._build_encode_net(input_image=self.next_image)
            print("self.current_feature_map:",self.current_feature_map)

            #-------------------------current_state||next_state --> action  逆向动力学  预测两个状态之间的动作---------------------
            self.cur_next_concat = tf.concat([self.current_state_feature, self.next_state_feature], axis=1)
            self.action_predicted_temp = tf.nn.elu(tf.matmul(self.cur_next_concat, self.weight_p_a_1) + self.bias_p_a_1)
            self.action_predicted = tf.nn.softmax(tf.matmul(self.action_predicted_temp, self.weight_p_a_2) + self.bias_p_a_2)
            self.action_predict_loss = -tf.reduce_mean(
                tf.one_hot(self.action,4,dtype=tf.float32)
               *tf.log(tf.clip_by_value(self.action_predicted,1e-10,1.0)))
            self.action_predict_grads = [tf.clip_by_norm(item, 40) for item in
                                         tf.gradients(self.action_predict_loss,
                                                      self.action_predict_params + self.encode_params )]
            self.update_action_predict_op = self.OPT_PA.apply_gradients(list(zip(self.action_predict_grads,
                                                      self.action_predict_params + self.encode_params )))

            #----------------------------------------current_state||action --> next_state---------------------------------------
            self.cur_action_concat = tf.concat([self.current_state_feature, tf.one_hot(self.action,4,dtype=tf.float32)], axis=1)
            self.state_feature_predict_temp      =\
                tf.nn.elu(tf.matmul(self.cur_action_concat, self.weight_p_n_s_1) + self.bias_p_n_s_1)
            self.state_feature_predict = \
                tf.nn.elu(tf.matmul(self.state_feature_predict_temp, self.weight_p_n_s_2) + self.bias_p_n_s_2)
            self.predict_decoded_image = self._build_decode_net(input_feature=self.state_feature_predict)
            self.loss_raw = tf.subtract(self.next_image,self.predict_decoded_image)
            self.state_predict_loss =  tf.reduce_mean(tf.square(self.loss_raw))
            self.state_predict_grads = [tf.clip_by_norm(item, 40) for item in tf.gradients(self.state_predict_loss,
                                        self.decode_params + self.state_predict_params + self.encode_params)]
            self.update_state_predict_op = self.OPT.apply_gradients(list(zip(self.state_predict_grads,
                                        self.decode_params + self.state_predict_params + self.encode_params)))

            # current_image --> current_feature --> current_image
            self._build_and_prepare_auto_encoder(input=self.current_flatten_feature)

            self._prepare_store()

    def _prepare_weight(self):
        # encode
        self.conv1_weight = generate_conv2d_weight(shape=[3,3,3,8]  ,name="conv1_weight_encode")
        self.conv1_bias   = generate_conv2d_bias(shape=8            ,name='conv1_bias_encode')
        self.conv2_weight = generate_conv2d_weight(shape=[3,3,8,16] ,name="conv2_weight_encode")
        self.conv2_bias   = generate_conv2d_bias(shape=16           ,name='conv2_bias_encode')
        self.conv3_weight = generate_conv2d_weight(shape=[2,2,16,32],name="conv3_weight_encode")
        self.conv3_bias   = generate_conv2d_bias(shape=32           ,name='conv3_bias_encode')
        # self.conv4_weight = generate_conv2d_weight(shape=[3,3,32,64],name="conv4_weight_encode")
        # self.conv4_bias   = generate_conv2d_bias(shape=64,name='conv4_bias_encode')
        self.fc_weight    = generate_fc_weight(shape=[4160,4096],name='fc_weight_encode')
        self.fc_bias      = generate_fc_weight(shape=[4096],name='fc_bias_encode')
        self.encode_params = [
            self.conv1_weight,self.conv1_bias,
            self.conv2_weight,self.conv2_bias,
            self.conv3_weight,self.conv3_bias,
            # self.conv4_weight,self.conv4_bias,
            self.fc_weight   ,self.fc_bias
        ]

        # decode weight
        self.conv4_weight = generate_conv2d_weight(shape=[3,3,32,16],name='conv4_weight_encode')
        self.conv4_bias   = generate_conv2d_bias(shape=16,name='conv4_weight_bias')
        self.conv5_weight = generate_conv2d_weight(shape=[3,3,16,8],name='conv5_weight_encode')
        self.conv5_bias   = generate_conv2d_bias(shape=8,name='conv5_weight_bias')
        self.conv6_weight = generate_conv2d_weight(shape=[3,3,8,3],name='conv6_weight_encode')
        self.conv6_bias   = generate_conv2d_bias(shape=3,name='conv6_weight_bias')
        self.conv7_weight = generate_conv2d_weight(shape=[3,3,3,3],name='conv7_weight_encode')
        self.conv7_bias   = generate_conv2d_bias(shape=3,name='conv7_weight_bias')
        self.decode_params = [
            self.conv4_weight,self.conv4_bias,
            self.conv5_weight,self.conv5_bias,
            self.conv6_weight,self.conv6_bias,
            self.conv7_weight,self.conv7_bias
        ]



        # weight for predict action
        self.weight_p_a_1    = generate_fc_weight(shape=[4096*2,2048],name='p_a_weight_1')
        self.bias_p_a_1      = generate_fc_weight(shape=[2048],name='p_a_bias_1')
        self.weight_p_a_2    = generate_fc_weight(shape=[2048,4],name='p_a_weight_2')
        self.bias_p_a_2      = generate_fc_weight(shape=[4],name='p_a_bias_2')
        self.action_predict_params = [self.weight_p_a_1,self.bias_p_a_1,
                                      self.weight_p_a_2,self.bias_p_a_2
                                      ]

        # current_state||action --> next_state  前向动力学  预测下一时刻的特质
        self.weight_p_n_s_1   = generate_fc_weight(shape=[4096+4,4160],name='p_ns_weight_1')
        self.bias_p_n_s_1     = generate_fc_weight(shape=[4160],name='p_ns__bias_1')
        self.weight_p_n_s_2   = generate_fc_weight(shape=[4160,4160],name='p_ns_weight_2')
        self.bias_p_n_s_2     = generate_fc_weight(shape=[4160],name='p_ns__bias_2')
        self.state_predict_params = [
                                self.weight_p_n_s_1,self.bias_p_n_s_1,
                                self.weight_p_n_s_2,self.bias_p_n_s_2
                                    ]



    def _build_encode_net(self,input_image):
        conv1 = tf.nn.conv2d(input_image, self.conv1_weight, strides=[1, 2, 2, 1], padding='SAME')
        conv1_pooling = tf.layers.max_pooling2d(conv1,(2,2),(2,2),padding='same')
        elu1 = tf.nn.elu(tf.nn.bias_add(conv1_pooling, self.conv1_bias))

        conv2 = tf.nn.conv2d(elu1, self.conv2_weight, strides=[1, 2, 2, 1], padding='SAME')
        conv2_pooling = tf.layers.max_pooling2d(conv2,(2,2),(2,2),padding='same')
        elu2 = tf.nn.elu(tf.nn.bias_add(conv2_pooling, self.conv2_bias))

        conv3 = tf.nn.conv2d(elu2, self.conv3_weight, strides=[1, 2, 2, 1], padding='SAME')
        feature_map = tf.nn.elu(tf.nn.bias_add(conv3, self.conv3_bias))
        # conv4 = tf.nn.conv2d(relu3, self.conv4_weight, strides=[1, 2, 2, 1], padding='SAME')
        # relu4 = tf.nn.elu(tf.nn.bias_add(conv4,self.conv4_bias))
        flatten_feature = flatten(feature_map)
        # print("before flatten : ",elu3)
        # print("after flatten : ",flatten_feature)
        state_feature = tf.nn.sigmoid(tf.matmul(flatten_feature, self.fc_weight) + self.fc_bias)
        return flatten_feature,state_feature,feature_map

    def _build_decode_net(self,input_feature):
        input_feature = inverse_flatten(input_feature,(-1,10,13,32))
        print("input_feature ：",input_feature)
        resize4 = tf.image.resize_images(input_feature,size=(13,17),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv4 = tf.nn.conv2d(resize4, self.conv4_weight,strides=[1,1,1,1],  padding='SAME')
        conv4 = tf.nn.elu(tf.nn.bias_add(conv4, self.conv4_bias))

        resize5 = tf.image.resize_images(conv4,size=(25,34),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv5 = tf.nn.conv2d(resize5, self.conv5_weight,strides=[1,1,1,1] , padding='SAME')
        conv5 = tf.nn.elu(tf.nn.bias_add(conv5, self.conv5_bias))


        resize6 = tf.image.resize_images(conv5,size=(75,100),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv6 = tf.nn.conv2d(resize6, self.conv6_weight, strides=[1,1,1,1],padding='SAME')
        conv6 = tf.nn.elu(tf.nn.bias_add(conv6, self.conv6_bias))


        resize7 = tf.image.resize_images(conv6,size=(300,400),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv7 = tf.nn.conv2d(resize7, self.conv7_weight,strides=[1,1,1,1], padding='SAME')
        decoded_image = tf.nn.sigmoid(tf.nn.bias_add(conv7, self.conv7_bias))
        return decoded_image

    def _build_and_prepare_auto_encoder(self,input):
        self.decoded_image = self._build_decode_net(input)
        # self.decoder_loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #                     labels = self.current_image,
        #                     logits = self.decoded_image))
        self.decoder_loss =  tf.reduce_mean(tf.square(self.current_image - self.decoded_image))
        self.decoder_grads = [tf.clip_by_norm(item, 40) for item in tf.gradients(self.decoder_loss,
                                                            self.encode_params[:-2] + self.decode_params)]
        self.update_decoder = self.OPT.apply_gradients(list(zip(self.decoder_grads,
                                                            self.encode_params[:-2] + self.decode_params)))
        # self.decoder_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.current_image,
        #                                                             logits = self.decoded_image)
        # self.decoder_cost = tf.reduce_mean(self.decoder_loss)
        # self.update_decoder = tf.train.AdamOptimizer(self.lr).minimize(self.decoder_cost)
        return

    def udpate_auto_decoder(self,current_image,lr,lr_pa):
        self.session.run(self.update_decoder,
                         feed_dict={self.current_image : current_image,
                                    self.lr            : lr,
                                    self.lr_pa         : lr_pa})

    def update_state_action_predict_network(self,current_image,action,next_image,lr,lr_pa):
        self.session.run([self.update_action_predict_op,self.update_state_predict_op],
                         feed_dict={
                             self.current_image  : current_image,
                             self.next_image     : next_image,
                             self.action         : action  ,
                             self.lr             : lr,
                             self.lr_pa          : lr_pa})

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
        action_predict_loss,state_predict_loss,decoder_loss = self.session.run([
            self.action_predict_loss,self.state_predict_loss,self.decoder_loss],
                         feed_dict={
                             self.current_image  : current_image,
                             self.next_image     : next_image,
                             self.action         : action     })
        acc = count/len(action_predicted)
        return action_predict_loss,state_predict_loss,acc,decoder_loss


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
                if episode>600:
                    learning_rate = 0.00001
                    learning_rate_for_pa = 0.00005
                else:
                    learning_rate = 0.0001
                    learning_rate_for_pa = 0.0001

                if step%64 == 0 :
                    buffer_state, buffer_action, buffer_next_state =\
                        np.array(buffer_state),np.array(buffer_action),np.array(buffer_next_state)
                    self.net.update_state_action_predict_network(current_image=buffer_state,
                                                                 action=buffer_action,
                                                                 next_image=buffer_next_state,
                                                                 lr = learning_rate,
                                                                 lr_pa=learning_rate_for_pa)
                    self.net.udpate_auto_decoder(current_image=buffer_state,lr=learning_rate,lr_pa=learning_rate_for_pa)
                    action_loss,state_loss,acc,decoder_cost= self.net.get_loss(current_image=buffer_state,
                                                                               action=buffer_action,
                                                                               next_image=buffer_next_state)
                    if episode > 600 and acc >0.6:
                        self.net.store(weight_path+str(acc)+'.ckpt')
                        print("store! Acc:%s"%acc)
                        episode = 99999999
                    # print(action_loss.shape)
                    # print(action_loss)
                    # print(np.sum(action_loss))
                    action_loss,state_loss = np.sum(action_loss),np.sum(state_loss)
                    print("Episode:%6s  ||  Steps:%4s  || State_predict_loss: %5s || Action_predict_loss: %5s || Accuracy:%s  || Decoder_loss:%s"%(
                        episode ,step,round(state_loss,3),round(action_loss,3),round(acc,3),round(decoder_cost,4)
                    ))
                    buffer_state,buffer_action,buffer_next_state = [],[],[]
                current_image = current_image_next
                if step >= 200:
                    episode = episode + 1
                    break


if __name__ == '__main__':

    with tf.device(device):


        config = tf.ConfigProto(allow_soft_placement=True)

        SESS = tf.Session(config=config)

        worker = Worker_for_encoder(SESS)

        SESS.run(tf.global_variables_initializer())

        worker.work()
