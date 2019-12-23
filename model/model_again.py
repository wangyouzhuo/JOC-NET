from utils.op import *
import tensorflow as tf
import numpy as np
from config.config import *
from tensorflow.python import pywrap_tensorflow
from model.model_op import *


encoder_weight_path = "/home/wyz/PycharmProjects/JOC-NET/weight/encode_weight.ckpt"


class ACNet(object):
    def __init__(self, scope,session,device,type,globalAC=None):
        tf.set_random_seed(50)

        with tf.device(device):
            self.session = session


            if scope == 'Weight_Store':
                with tf.variable_scope(scope):
                    self.global_a_params = self._build_global_params_dict(scope)
                    self.special_a_params_dict,self.special_c_params_dict = self._build_special_params_dict(scope)


            elif type == 'Target_Special':
                with tf.variable_scope(scope):

                    self.global_AC = globalAC

                    self.import_weight(encoder_weight_path)
                    self.prepare_encoder_weight()

                    self.state_image = tf.placeholder(tf.float32,[None,300,400,3], 'State_image')
                    self.action = tf.placeholder(tf.int32, [None, ], 'Action')
                    self.state_feature = self._build_encode_net(input_image=self.state_image)


                    self.special_v_target = tf.placeholder(tf.float32, [None, 1], 'special_V_target')

                    self.OPT_SPE_A = tf.train.RMSPropOptimizer(LR_SPE_A, name='Spe_RMSPropA')
                    self.OPT_SPE_C = tf.train.RMSPropOptimizer(LR_SPE_C, name='Spe_RMSPropC')

                    self.special_a_prob,self.special_v,self.special_a_params,self.special_c_params \
                        = self._build_special_net(scope)

                    self._prepare_special_loss(scope)

                    self._prepare_special_grads(scope)

                    self._prepare_special_update_op(scope)

                    self._prepare_special_pull_op(scope)

            elif type == 'Target_Universal':
                with tf.variable_scope(scope):

                    self.global_AC = globalAC

                    self.import_weight(encoder_weight_path)
                    self.prepare_encoder_weight()

                    self.state_image  = tf.placeholder(tf.float32 ,[None,300,400,3], 'State_image')
                    self.target_image = tf.placeholder(tf.float32,[None,300,400,3], 'Target_image')

                    self.state_feature = self._build_encode_net(input_image=self.state_image)
                    self.target_feature = self._build_encode_net(input_image=self.target_image)


                    self.action       = tf.placeholder(tf.int32,   [None, ], 'Action')

                    self.adv      = tf.placeholder(tf.float32, [None,1], 'Advantage')

                    self.learning_rate = tf.placeholder(tf.float32, None, 'Learning_rate')

                    self.OPT_A = tf.train.RMSPropOptimizer(self.learning_rate, name='Glo_RMSPropA')

                    # target_general_network
                    self.global_a_prob ,self.global_a_params = self._build_global_net(scope)

                    # target_special_network
                    self.special_a_prob,self.special_v,self.special_a_params,self.special_c_params \
                        = self._build_special_net(scope)

                    self._prepare_global_loss(scope)

                    self._prepare_global_grads(scope)

                    # self._prepare_special_update_op(scope)
                    self._prepare_global_update_op(scope)

                    self._prepare_global_pull_op(scope)
                    self._prepare_special_pull_op(scope)



    def _build_global_params_dict(self, scope):
        with tf.variable_scope(scope):
            # encode
            w_encode = generate_fc_weight(shape=[2048, 512], name='global_w_encode')
            b_encode = generate_fc_bias(shape=[512]        , name='global_b_encode')
            # fusion
            w_fusion = generate_fc_weight(shape=[1024, 512], name='global_w_f')
            b_fusion = generate_fc_bias(shape=[512]        , name='global_b_f')
            # scene
            w_scene  = generate_fc_weight(shape=[512, 512]  , name='global_w_s')
            b_scene  = generate_fc_bias(shape=[512]         , name='global_b_s')
            # actor
            w_actor  = generate_fc_weight(shape=[512, 4] , name='global_w_a')
            b_actor  = generate_fc_bias(shape=[4]        , name='global_b_a')

            a_params = [w_encode, b_encode, w_fusion, b_fusion,w_scene, b_scene, w_actor,  b_actor]

            return a_params

    def _build_special_params_dict(self,scope):
        with tf.variable_scope(scope):
            a_params_dict,c_params_dict = dict(),dict()
            for target_key in TARGET_ID_LIST:
                w_actor  = generate_fc_weight(shape=[2048, 4], name='actor_w'+str(target_key))
                b_actor  = generate_fc_bias(shape=[4],         name='actor_b'+str(target_key))
                w_critic = generate_fc_weight(shape=[2048, 1], name='critic_w'+str(target_key))
                b_critic = generate_fc_bias(shape=[1],         name='critic_b'+str(target_key))
                a_params = [w_actor  ,b_actor ]
                c_params = [w_critic ,b_critic]
                kv_a = {target_key:a_params}
                kv_c = {target_key:c_params}
                a_params_dict.update(kv_a)
                c_params_dict.update(kv_c)
            return  a_params_dict,c_params_dict

    def _build_global_net(self, scope):
        with tf.variable_scope(scope):
            # global_network only need actor

            w_encode = generate_fc_weight(shape=[2048, 512], name='global_w_encode')
            b_encode = generate_fc_bias(shape=[512], name='global_b_encode')

            # encode current_state into s_encode
            s_encode = tf.nn.elu(tf.matmul(self.state_feature, w_encode) + b_encode)
            # encode target_state  into t_encode
            t_encode = tf.nn.elu(tf.matmul(self.target_feature, w_encode) + b_encode) # encode target_state  into t_encode

            # s_encode||t_encode --> concat
            concat = tf.concat([s_encode, t_encode], axis=1)  # s_encode||t_encode --> concat

            # concat --> fusion_layer
            w_fusion = generate_fc_weight(shape=[1024, 512], name='global_w_f')
            b_fusion = generate_fc_bias(shape=[512], name='global_b_f')
            fusion_layer = tf.nn.elu(tf.matmul(concat, w_fusion) + b_fusion)

            # fusion_layer --> scene_layer
            w_scene = generate_fc_weight(shape=[512, 512], name='global_w_s')
            b_scene = generate_fc_bias(shape=[512], name='global_b_s')
            scene_layer = tf.nn.elu(tf.matmul(fusion_layer, w_scene) + b_scene)

            # scene_layer --> prob
            w_actor = generate_fc_weight(shape=[512, 4], name='global_w_a')
            b_actor = generate_fc_bias(shape=[4], name='global_b_a')
            self.global_logits = tf.matmul(scene_layer, w_actor) + b_actor
            prob = tf.nn.softmax(self.global_logits)

            a_params = [w_encode, b_encode,w_fusion, b_fusion,
                        w_scene , b_scene ,w_actor , b_actor ]

            return prob , a_params

    def _build_special_net(self, scope):
        with tf.variable_scope(scope):

            # special_actor
            w_actor = generate_fc_weight(shape=[2048, 4], name='special_w_a')
            b_actor = generate_fc_bias(shape=[4], name='special_b_a')
            self.special_logits = tf.matmul(self.state_feature, w_actor) + b_actor
            prob = tf.nn.softmax(self.special_logits)

            # special_critic
            w_critic = generate_fc_weight(shape=[2048, 1], name='special_w_c')
            b_critic = generate_fc_bias(shape=[1], name='special_b_c')
            value = tf.matmul(self.state_feature, w_critic) + b_critic

            a_params = [w_actor,  b_actor ]
            c_params = [w_critic, b_critic]

            return prob, value, a_params, c_params

    def _prepare_global_loss(self,scope):
        with tf.name_scope(scope+'global_loss'):

            with tf.name_scope('global_a_loss'):
                # prob from target_special_net (dont update)
                p_target = tf.stop_gradient(self.special_a_prob)
                # prob from global_special_net (need update)
                p_update = self.global_a_prob

                self.spe_actor_reg_loss = -tf.reduce_mean(p_target*tf.log(tf.clip_by_value(p_update,1e-10,1.0)))

                glo_log_prob = tf.reduce_sum(tf.log(self.global_a_prob + 1e-5)*tf.one_hot(self.action,4,dtype=tf.float32),
                                             axis=1,keep_dims=True)

                # self.adv is calculated by target_special_network
                actor_loss = glo_log_prob*self.adv

                self.glo_entropy = -tf.reduce_mean(self.global_a_prob*tf.log(self.global_a_prob + 1e-5), axis=1,keep_dims=True)  # encourage exploration

                # self.loss = ENTROPY_BETA*self.glo_entropy + actor_loss - self.kl_beta*self.spe_actor_reg_loss

                if SOFT_LOSS_TYPE == "hard_imitation":
                    self.loss = self.spe_actor_reg_loss

                elif SOFT_LOSS_TYPE == 'no_soft_imitation':
                    self.loss =  ENTROPY_BETA*self.glo_entropy + actor_loss

                elif SOFT_LOSS_TYPE == 'with_soft_imitation':
                    self.loss = ENTROPY_BETA*self.glo_entropy + actor_loss - self.spe_actor_reg_loss


                self.reg_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(L2_REG),self.global_a_params)

                self.global_a_loss = tf.reduce_mean(-self.loss )

                #self.global_a_loss = tf.reduce_mean(-self.loss + self.reg_loss)

    def _prepare_special_loss(self,scope):
        with tf.name_scope(scope+'special_loss'):

            with tf.name_scope('special_c_loss'):
                self.special_td = tf.subtract(self.special_v_target, self.special_v, name='special_TD_error')
                self.special_c_loss = tf.reduce_mean(tf.square(self.special_td))

            with tf.name_scope('special_a_loss'):
                special_log_prob = tf.reduce_sum(
                    tf.log(self.special_a_prob+1e-9)*tf.one_hot(self.action,4,dtype=tf.float32),axis=1,keep_dims=True)
                spe_exp_v = special_log_prob*tf.stop_gradient(self.special_td)
                self.spe_entropy = -tf.reduce_mean(self.special_a_prob*tf.log(self.special_a_prob+1e-5),axis=1,keep_dims=True)  # encourage exploration
                self.spe_exp_v = ENTROPY_BETA*self.spe_entropy + spe_exp_v
                self.special_a_loss = tf.reduce_mean(-self.spe_exp_v)

    def _prepare_global_grads(self,scope):
        with tf.name_scope(scope+'global_grads'):
            with tf.name_scope('global_net_grad'):
                self.global_a_grads = [tf.clip_by_norm(item, 40) for item in
                                       tf.gradients(self.global_a_loss, self.global_a_params)]

    def _prepare_special_grads(self,scope):
        with tf.name_scope(scope+'special_grads'):
            with tf.name_scope('special_net_grad'):
                self.special_a_grads = [tf.clip_by_norm(item, 40) for item in
                                        tf.gradients(self.special_a_loss, self.special_a_params)]

                self.special_c_grads = [tf.clip_by_norm(item, 40) for item in
                                        tf.gradients(self.special_c_loss, self.special_c_params)]

    def _prepare_global_update_op(self,scope):
        with tf.name_scope(scope+'_global_update'):
            self.update_global_a_op = self.OPT_A.apply_gradients(list(zip(self.global_a_grads, self.global_AC.global_a_params)))

    def _prepare_special_update_op(self,scope):
        with tf.name_scope(scope+'_special_update'):
            self.update_special_a_dict, self.update_special_c_dict = dict(), dict()
            self.update_special_q_dict = dict()
            for key in TARGET_ID_LIST:
                kv_a = {key: self.OPT_SPE_A.apply_gradients(list(zip(self.special_a_grads , self.global_AC.special_a_params_dict[key])))}
                kv_c = {key: self.OPT_SPE_C.apply_gradients(list(zip(self.special_c_grads , self.global_AC.special_c_params_dict[key])))}
                self.update_special_a_dict.update(kv_a)
                self.update_special_c_dict.update(kv_c)

    def _prepare_global_pull_op(self,scope):
        with tf.name_scope(scope+'pull_global_params'):
            self.pull_a_params_global = [l_p.assign(g_p) for l_p, g_p in
                                         zip(self.global_a_params, self.global_AC.global_a_params)]

    def _prepare_special_pull_op(self,scope):
        with tf.name_scope(scope+'pull_special_params'):
            self.pull_a_params_special_dict, self.pull_c_params_special_dict = dict(), dict()
            self.pull_q_params_special_dict = dict()
            for key in TARGET_ID_LIST:
                kv_a = {key: [l_p.assign(g_p) for l_p, g_p in
                              zip(self.special_a_params, self.global_AC.special_a_params_dict[key])]}
                kv_c = {key: [l_p.assign(g_p) for l_p, g_p in
                              zip(self.special_c_params, self.global_AC.special_c_params_dict[key])]}
                self.pull_a_params_special_dict.update(kv_a)
                self.pull_c_params_special_dict.update(kv_c)

    def update_special(self, feed_dict,target_id):  # run by a local
        self.session.run([self.update_special_a_dict[target_id],
                          self.update_special_c_dict[target_id]],feed_dict)

    def update_global(self,feed_dict):
        self.session.run(self.update_global_a_op, feed_dict)  # local grads applies to global net

    def pull_global(self):
        self.session.run([self.pull_a_params_global])

    def pull_special(self,target_id):  # run by a local
        self.session.run([self.pull_a_params_special_dict[target_id]
                             ,self.pull_c_params_special_dict[target_id]])

    def spe_choose_action(self, state_image):  # run by a local
        prob_weights = self.session.run(self.special_a_prob, feed_dict={self.state_image: state_image[np.newaxis, :]} )
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action,prob_weights

    def glo_choose_action(self, current_image, target_image):  # run by a local
        prob_weights = self.session.run(self.global_a_prob,
                                        feed_dict={self.state_image: current_image[np.newaxis, :],
                                                   self.target_image: target_image[np.newaxis, :]} )
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action,prob_weights.ravel()

    def load_weight(self,target_id):
        self.session.run([self.pull_a_params_special_dict[target_id],self.pull_c_params_special_dict[target_id]])

    def get_special_value(self,feed_dict):
        special_value = self.session.run(self.special_v,feed_dict)
        return special_value

    def get_weight(self,name):
        return  self.encoder_weight_dict[name]

    def prepare_encoder_weight(self):
        self.conv1_weight = self.get_weight(name="conv1_weight_encode")
        self.conv1_bias   = self.get_weight(name='conv1_bias_encode')
        self.conv2_weight = self.get_weight(name="conv2_weight_encode")
        self.conv2_bias   = self.get_weight(name='conv2_bias_encode')
        self.conv3_weight = self.get_weight(name="conv3_weight_encode")
        self.conv3_bias   = self.get_weight(name='conv3_bias_encode')
        self.conv4_weight = self.get_weight(name="conv4_weight_encode")
        self.conv4_bias   = self.get_weight(name='conv4_bias_encode')
        self.fc_weight    = self.get_weight(name='fc_weight_encode')
        self.fc_bias      = self.get_weight(name='fc_bias_encode')
        self.encode_params = [self.conv1_weight,self.conv1_bias,self.conv2_weight,self.conv2_bias,
                              self.conv3_weight,self.conv3_bias,self.conv4_weight,self.conv4_bias,
                              self.fc_weight,self.fc_bias]

    def import_weight(self,weight_path):
        model_reader = pywrap_tensorflow.NewCheckpointReader(weight_path)
        var_dict = model_reader.get_variable_to_shape_map()
        result_dict = dict()
        for key in var_dict:
            if 'encode' in key.split('/')[-1]:
                result_dict[key.split('/')[-1]] = model_reader.get_tensor(key)
        self.encoder_weight_dict = result_dict

    def _build_encode_net(self,input_image):
        conv1 = tf.nn.conv2d(input_image, self.conv1_weight, strides=[1, 4, 4, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, self.conv1_bias))
        conv2 = tf.nn.conv2d(relu1, self.conv2_weight, strides=[1, 2, 2, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.conv2_bias))
        conv3 = tf.nn.conv2d(relu2, self.conv3_weight, strides=[1, 2, 2, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, self.conv3_bias))
        conv4 = tf.nn.conv2d(relu3, self.conv4_weight, strides=[1, 2, 2, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4,self.conv4_bias))
        flatten_feature = flatten(relu4)
        state_feature = tf.nn.elu(tf.matmul(flatten_feature, self.fc_weight) + self.fc_bias)
        return state_feature



