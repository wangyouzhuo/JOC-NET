from utils.op import *
import tensorflow as tf
import numpy as np
from model.model_op import _build_encode_params_dict,_build_universal_network_params_dict,_build_special_params_dict
from model.model_op import _build_special_net
from config.config import *
from model.model_op import *


class ACNet(object):
    def __init__(self, scope,session,device,type,globalAC=None):
        tf.set_random_seed(50)

        with tf.device(device):
            self.session = session

            if scope == 'Weight_Store':
                with tf.variable_scope(scope):
                    # 把300*400*3的图像 编码成 2048维向量
                    self.conv1_weight = generate_conv2d_weight(shape=[5,5,3,8],name="conv1_weight")
                    self.conv1_bais   = generate_conv2d_bias(shape=8,name='conv1_bias')
                    self.conv2_weight = generate_conv2d_weight(shape=[3,3,8,16],name="conv2_weight")
                    self.conv2_bais   = generate_conv2d_bias(shape=16,name='conv2_bias')
                    self.conv3_weight = generate_conv2d_weight(shape=[3,3,16,32],name="conv3_weight")
                    self.conv3_bais   = generate_conv2d_bias(shape=32,name='conv3_bias')
                    self.conv4_weight = generate_conv2d_weight(shape=[3,3,32,64],name="conv4_weight")
                    self.conv4_bais   = generate_conv2d_bias(shape=64,name='conv4_bias')
                    self.fc_weight    = generate_fc_weight(shape=[8320,2048],name='fc_weight')
                    self.fc_bias      = generate_fc_weight(shape=[2048],name='fc_bias')
                    self.encode_params = [self.conv1_weight,self.conv1_bais,self.conv2_weight,self.conv2_bais,
                                          self.conv3_weight,self.conv3_bais,self.conv4_weight,self.conv4_bais,
                                          self.fc_weight,self.fc_bias]

                    # current_state||next_state --> the action 逆向动力学
                    self.weight_p_a    = generate_fc_weight(shape=[4096,4],name='p_a_weight')
                    self.bias_p_a      = generate_fc_weight(shape=[4],name='p_a_bias')

                    # current_state||action --> next_state  前向动力学
                    self.weight_p_n_s    = generate_fc_weight(shape=[2048+4,2048],name='p_ns_weight')
                    self.bias_p_n_s      = generate_fc_weight(shape=[2048],name='p_ns__bias')

                    # 多目标通用策略
                    self.w_fusion = generate_fc_weight(shape=[4096, 512], name='global_w_fusion')
                    self.b_fusion = generate_fc_bias(shape=[512]        , name='global_b_fusion')
                    self.w_scene  = generate_fc_weight(shape=[512, 512]  , name='global_w_scene')
                    self.b_scene  = generate_fc_bias(shape=[512]         , name='global_b_scene')
                    self.w_actor  = generate_fc_weight(shape=[512, 4] , name='global_w_actor')
                    self.b_actor  = generate_fc_bias(shape=[4]        , name='global_b_actor')


                    a_params_dict,c_params_dict = dict(),dict()
                    for target_key in TARGET_ID_LIST:
                        w_actor  = generate_fc_weight(shape=[2048, 4], name='actor_w_'+str(target_key))
                        b_actor  = generate_fc_bias(shape=[4],         name='actor_b_'+str(target_key))
                        w_critic = generate_fc_weight(shape=[2048, 1], name='critic_w_'+str(target_key))
                        b_critic = generate_fc_bias(shape=[1],         name='critic_b_'+str(target_key))
                        a_params = [w_actor  ,b_actor ]
                        c_params = [w_critic ,b_critic]
                        kv_a = {target_key:a_params}
                        kv_c = {target_key:c_params}
                        a_params_dict.update(kv_a)
                        c_params_dict.update(kv_c)
                    self.special_actor_params_dict,self.special_critic_params_dict = a_params_dict,c_params_dict

            elif type == 'Target_Special':
                with tf.variable_scope(scope):

                    self.global_AC = globalAC

                    self.state_image = tf.placeholder(tf.float32,[None,300,400,3], 'State_image')
                    self.action = tf.placeholder(tf.int32, [None, ], 'Action')

                    self.state_feature = self._build_encode_net(input_image=self.state_image)

                    self.special_v_target = tf.placeholder(tf.float32, [None, 1], 'special_V_target')

                    self.OPT_A = tf.train.RMSPropOptimizer(LR_SPE_A, name='Spe_RMSPropA')
                    self.OPT_C = tf.train.RMSPropOptimizer(LR_SPE_C, name='Spe_RMSPropC')

                    self.special_a_prob,self.special_v,self.special_a_params,self.special_c_params = \
                        _build_special_net(state_feature=self.state_feature)

                    # prepare_special_network
                    # special_actor
                    self.special_w_actor = generate_fc_weight(shape=[2048, 4], name='special_w_a')
                    self.special_b_actor = generate_fc_bias(shape=[4], name='special_b_a')
                    self.special_logits = tf.matmul(self.state_feature, self.special_w_actor) + self.special_b_actor
                    self.special_prob = tf.nn.softmax(self.special_logits)
                    # special_critic
                    self.special_w_critic = generate_fc_weight(shape=[2048, 1], name='special_w_c')
                    self.special_b_critic = generate_fc_bias(shape=[1], name='special_b_c')
                    self.special_value = tf.matmul(self.state_feature, self.special_w_critic) + self.special_b_critic
                    self.special_actor_params  = [self.special_w_actor ,self.special_b_actor]
                    self.special_critic_params = [self.special_w_critic,self.special_b_critic]



                    # prepare special_loss
                    with tf.name_scope('special_c_loss'):
                        self.special_td = tf.subtract(self.special_v_target, self.special_value, name='special_TD_error')
                        self.special_c_loss = tf.reduce_mean(tf.square(self.special_td))
                    with tf.name_scope('special_a_loss'):
                        special_log_prob = tf.reduce_sum(
                            tf.log(self.special_prob+1e-9)*tf.one_hot(self.action,4,dtype=tf.float32),axis=1,keep_dims=True)
                        spe_exp_v = special_log_prob*tf.stop_gradient(self.special_td)
                        self.spe_entropy = -tf.reduce_mean(self.special_a_prob*tf.log(self.special_a_prob+1e-5),axis=1,keep_dims=True)  # encourage exploration
                        self.spe_exp_v = ENTROPY_BETA*self.spe_entropy + spe_exp_v
                        self.special_a_loss = tf.reduce_mean(-self.spe_exp_v)
                    # prepare special_grads 计算梯度
                    with tf.name_scope('special_actor_grads'):
                        self.special_a_grads = [tf.clip_by_norm(item, 40) for item in
                                                tf.gradients(self.special_a_loss,
                                                             self.special_actor_params+self.global_AC.encode_params)]
                    with tf.name_scope('special_actor_grads'):
                        self.special_c_grads = [tf.clip_by_norm(item, 40) for item in
                                                tf.gradients(self.special_c_loss,
                                                             self.special_critic_params+self.global_AC.encode_params)]
                    # 用梯度更新 对应参数
                    with tf.name_scope('special_update'):
                        self.update_special_a_dict, self.update_special_c_dict = dict(), dict()
                        self.update_special_q_dict = dict()
                        for key in TARGET_ID_LIST:
                            kv_a = {key: self.OPT_A.apply_gradients(list(zip(self.special_a_grads ,
                                                 self.global_AC.special_actor_params_dict[key]+self.global_AC.encode_params)))}
                            kv_c = {key: self.OPT_C.apply_gradients(list(zip(self.special_c_grads ,
                                                 self.global_AC.special_critic_params_dict[key]+self.global_AC.encode_params)))}
                            self.update_special_a_dict.update(kv_a)
                            self.update_special_c_dict.update(kv_c)

                    # 把weight_temp的值取出来，赋值给special_net
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


                    # self._prepare_special_grads(scope)
                    # self._prepare_special_update_op(scope)

                    self._prepare_special_pull_op(scope)

                    self._build_action_state_predict_net(
                        encode_params         = self.global_AC.encode_params
                        ,state_predict_params  = self.global_AC.state_predict_params
                        ,action_predict_params = self.global_AC.action_predict_params )

            elif type == 'Target_General':

                self.global_AC = globalAC

                self.state_image = tf.placeholder(tf.float32 ,[None,300,400,3], 'State_image')
                self.target_image = tf.placeholder(tf.float32,[None,300,400,3], 'Target_image')

                self.state_feature  = self._build_encode_net(input_image=self.state_image ,encode_params=self.global_AC.encode_params)
                self.target_feature = self._build_encode_net(input_image=self.target_image,encode_params=self.global_AC.encode_params)


                self.a        = tf.placeholder(tf.int32,   [None, ], 'Action')
                self.adv      = tf.placeholder(tf.float32, [None,1], 'Advantage')
                self.kl_beta  = tf.placeholder(tf.float32, [None,], 'KL_BETA')

                self.learning_rate = tf.placeholder(tf.float32, None, 'Learning_rate')

                self.OPT_A = tf.train.RMSPropOptimizer(self.learning_rate, name='Glo_RMSPropA')

                # target_general_network
                self._build_global_net()

                # target_special_network
                self.special_a_prob,self.special_v,self.special_a_params,self.special_c_params = \
                    self._build_special_net(state_feature=self.state_feature)

                self._prepare_global_loss(scope)

                self._prepare_global_grads(scope)

                # self._prepare_special_update_op(scope)
                self._prepare_global_update_op(scope)

                # self._prepare_global_pull_op(scope)
                # self._prepare_special_pull_op(scope)

                self._prepare_kl_devergance(scope)

                self._prepare_many_goals_loss_grads_update()

    def _prepare_global_loss(self,scope):
        with tf.name_scope(scope+'global_loss'):

            with tf.name_scope('global_a_loss'):
                # prob from target_special_net (dont update)
                p_target = tf.stop_gradient(self.special_a_prob)
                # prob from global_special_net (need update)
                p_update = self.global_a_prob

                self.spe_actor_reg_loss = -tf.reduce_mean(p_target*tf.log(tf.clip_by_value(p_update,1e-10,1.0)))

                glo_log_prob = tf.reduce_sum(tf.log(self.global_a_prob + 1e-5)*tf.one_hot(self.a, 4, dtype=tf.float32),
                                             axis=1,keep_dims=True)

                # self.adv is calculated by target_special_network
                actor_loss = glo_log_prob*self.adv

                self.glo_entropy = -tf.reduce_mean(self.global_a_prob*tf.log(self.global_a_prob + 1e-5), axis=1,keep_dims=True)  # encourage exploration

                # self.loss = ENTROPY_BETA*self.glo_entropy + actor_loss - self.kl_beta*self.spe_actor_reg_loss

                if SOFT_LOSS_TYPE == "hard_imitation":
                    self.loss = -self.kl_beta*self.spe_actor_reg_loss

                elif SOFT_LOSS_TYPE == 'no_soft_imitation':
                    self.loss =  ENTROPY_BETA*self.glo_entropy + actor_loss

                elif SOFT_LOSS_TYPE == 'with_soft_imitation':
                    self.loss = ENTROPY_BETA*self.glo_entropy + actor_loss - self.kl_beta*self.spe_actor_reg_loss


                #self.reg_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(L2_REG),self.global_a_params)

                self.global_a_loss = tf.reduce_mean(-self.loss )

                #self.global_a_loss = tf.reduce_mean(-self.loss + self.reg_loss)

    def _prepare_special_loss(self,scope):
        with tf.name_scope(scope+'special_loss'):

            with tf.name_scope('special_c_loss'):
                self.special_td = tf.subtract(self.special_v_target, self.special_v, name='special_TD_error')
                self.special_c_loss = tf.reduce_mean(tf.square(self.special_td))

            with tf.name_scope('special_a_loss'):
                special_log_prob = tf.reduce_sum(
                    tf.log(self.special_a_prob+1e-9)*tf.one_hot(self.a,4,dtype=tf.float32),axis=1,keep_dims=True)
                spe_exp_v = special_log_prob*tf.stop_gradient(self.special_td)
                self.spe_entropy = -tf.reduce_mean(self.special_a_prob*tf.log(self.special_a_prob+1e-5),axis=1,keep_dims=True)  # encourage exploration
                self.spe_exp_v = ENTROPY_BETA*self.spe_entropy + spe_exp_v
                self.special_a_loss = tf.reduce_mean(-self.spe_exp_v)

    def _prepare_global_grads(self,scope):
        with tf.name_scope(scope+'global_grads'):
            with tf.name_scope('global_net_grad'):
                self.global_a_grads = [tf.clip_by_norm(item, 40) for item in
                                       tf.gradients(self.global_a_loss, self.global_AC.universal_net_params+self.global_AC.encode_params)]

    def _prepare_global_update_op(self,scope):
        with tf.name_scope(scope+'_global_update'):
            self.update_global_a_op = self.OPT_A.apply_gradients(list(zip(self.global_a_grads,
                                                                          self.global_AC.universal_net_params+self.global_AC.encode_params)))

    def _prepare_special_grads(self,scope):
        with tf.name_scope(scope+'special_grads'):
            with tf.name_scope('special_net_grad'):
                self.special_a_grads = [tf.clip_by_norm(item, 40) for item in
                                        tf.gradients(self.special_a_loss,self.special_a_params+self.global_AC.encode_params)]

                self.special_c_grads = [tf.clip_by_norm(item, 40) for item in
                                        tf.gradients(self.special_c_loss,self.special_c_params+self.global_AC.encode_params)]

    def _prepare_special_update_op(self,scope):
        with tf.name_scope(scope+'_special_update'):
            self.update_special_a_dict, self.update_special_c_dict = dict(), dict()
            self.update_special_q_dict = dict()
            for key in TARGET_ID_LIST:
                kv_a = {key: self.OPT_SPE_A.apply_gradients(list(zip(self.special_a_grads ,
                                                                     self.global_AC.special_a_params_dict[key]+self.global_AC.encode_params)))}
                kv_c = {key: self.OPT_SPE_C.apply_gradients(list(zip(self.special_c_grads ,
                                                                     self.global_AC.special_c_params_dict[key]+self.global_AC.encode_params)))}
                self.update_special_a_dict.update(kv_a)
                self.update_special_c_dict.update(kv_c)

    def _prepare_kl_devergance(self,scope):
        with tf.name_scope(scope+"kl_devergance"):
            p_target = tf.stop_gradient(self.special_a_prob)
            p_update = self.global_a_prob
            self.kl = self.KL_divergence(p_stable=p_target, p_advance=p_update)
            self.kl_mean = tf.reduce_mean(self.kl)

    # def _prepare_global_pull_op(self,scope):
    #     with tf.name_scope(scope+'pull_global_params'):
    #         self.pull_a_params_global = [l_p.assign(g_p) for l_p, g_p in
    #                                      zip(self.global_a_params, self.global_AC.universal_net_params)]
    #
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

    def compute_kl(self,feed_dict):
        special_logits,global_logits = self.session.run([self.special_logits,self.global_logits],feed_dict)
        p_target,p_update = self.session.run([self.special_a_prob,self.global_a_prob],feed_dict)
        kl = self.session.run(self.kl_mean,feed_dict=feed_dict)
        if kl>1:
            kl = 1
        #print('special_logits:%s  global_logits:%s  kl:%s '%(special_logits,global_logits,kl))
        return kl

    def update_special(self, feed_dict,target_id):  # run by a local
        self.session.run([self.update_special_a_dict[target_id],
                          self.update_special_c_dict[target_id]],feed_dict)

    def update_global(self,feed_dict):
        self.session.run(self.update_global_a_op, feed_dict)  # local grads applies to global net

    def pull_global(self):
        # self.session.run([self.pull_a_params_global])
        return

    def pull_special(self,target_id):  # run by a local
        self.session.run([self.pull_a_params_special_dict[target_id]
                             ,self.pull_c_params_special_dict[target_id]])

    def spe_choose_action(self, state_image):  # run by a local
        prob_weights = self.session.run(self.special_a_prob, feed_dict={self.state_image: state_image[np.newaxis, :]} )
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action,prob_weights

    def glo_choose_action(self, current_image, target_image):  # run by a local
        prob_weights = self.session.run(self.global_a_prob,
                                        feed_dict={self.state_image: current_image[np.newaxis, :],self.target_image: target_image[np.newaxis, :]} )
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action,prob_weights.ravel()

    def load_weight(self,target_id):
        self.session.run([self.pull_a_params_special_dict[target_id],self.pull_c_params_special_dict[target_id]])

    def KL_divergence(self,p_stable,p_advance):
        X = tf.distributions.Categorical(probs = p_stable )
        Y = tf.distributions.Categorical(probs = p_advance)
        return tf.clip_by_value(tf.distributions.kl_divergence(X, Y), clip_value_min=0.0, clip_value_max=10)
        #distance = tf.nn.l2_loss(p_stable-p_advance)
        #return distance

    def get_special_value(self,feed_dict):
        special_value = self.session.run(self.special_v,feed_dict)
        return special_value

    def _prepare_store(self):
        var = tf.global_variables()
        var_to_restore = [val for val in var if 'Global_Net' in val.name ]
        self.saver = tf.train.Saver(var_to_restore )

    def store(self,weight_path):
        self.saver.save(self.session,weight_path)

    def _build_action_state_predict_net(self,encode_params,state_predict_params,action_predict_params):
        self.current_image  = tf.placeholder(tf.float32,[None,300,400,3],name='current_image')
        self.next_image     = tf.placeholder(tf.float32,[None,300,400,3],name='next_image')
        self.action = tf.placeholder(tf.int32, [None, ],name="transition_action")

        current_feature = self._build_encode_net(input_image=self.current_image,encode_params=encode_params)
        next_feature = self._build_encode_net(input_image=self.next_image,encode_params=encode_params)

        # current_state||next_state --> action
        cur_next_concat = tf.concat([current_feature, next_feature], axis=1)
        action_predicted = tf.nn.elu(tf.matmul(cur_next_concat, action_predict_params[0]) + action_predict_params[1])

        self.action_predict_loss = tf.reduce_sum(tf.log(action_predicted+1e-9)*tf.one_hot(self.action,4,dtype=tf.float32),
                                                 axis=1,keep_dims=True)
        self.action_predict_grads = [tf.clip_by_norm(item, 40) for item in
                                     tf.gradients(self.action_predict_loss, encode_params+state_predict_params)]
        self.update_action_predict_op = self.OPT_A.apply_gradients(list(zip(self.action_predict_grads, encode_params+state_predict_params)))

        # current_state||action --> next_state
        self.cur_action_concat = tf.concat([current_feature, self.action], axis=1)
        state_predict =  tf.nn.elu(tf.matmul(cur_next_concat, state_predict_params[0]) + state_predict_params[1])

        loss_raw = tf.subtract(tf.stop_gradient(self.next_feature),state_predict)
        state_predict_loss =  tf.reduce_mean(tf.square(loss_raw))
        self.state_predict_grads = [tf.clip_by_norm(item, 40) for item in
                                    tf.gradients(state_predict_loss, encode_params+state_predict_params)]
        self.update_state_predict_op = self.OPT_A.apply_gradients(list(zip(self.state_predict_grads, encode_params+state_predict)))

    def _prepare_many_goals_loss_grads_update(self):
        self.action_many_goals = tf.placeholder(tf.int32, [None, ], 'Action_mg')
        self.mg_loss = -tf.reduce_mean(self.global_a_prob*tf.log(tf.clip_by_value(tf.one_hot(self.action_many_goals, 4, dtype=tf.float32),1e-10,1.0)))
        self.mg_grads =  self.special_a_grads = [tf.clip_by_norm(item, 40) for item in
                                                 tf.gradients(self.mg_loss,self.global_AC.universal_net_params+self.global_AC.encode_params)]
        self.update_global_with_mg = self.OPT_A.apply_gradients(list(zip(self.mg_grads,
                                                                         self.global_AC.universal_net_params+self.global_AC.encode_params)))

    def update_with_action_and_state_predict(self,current_image,next_image,action):
        self.session.run([self.update_action_predict_op,self.update_state_predict_op],
                         feed_dict={self.current_image: current_image[np.newaxis,:],
                                    self.next_image   : next_image[np.newaxis,:],
                                    self.action       : action[np.newaxis,:]})

    def update_with_mg(self,current_image,next_image,action):
        self.session.run(self.update_global_with_mg,feed_dict={
            self.action_many_goals : action[np.newaxis,:],
            self.state_image : current_image[np.newaxis,:],
            self.target_image : next_image[np.newaxis,:]
        })

    def _build_global_net(self):
        # s_encode||t_encode --> concat
        concat = tf.concat([self.state_feature, self.target_feature], axis=1)  # s_encode||t_encode --> concat
        # concat --> fusion_layer
        self.fusion_layer  = tf.nn.elu(tf.matmul(concat, self.global_AC.universal_net_params[0])
                                       + self.global_AC.universal_net_params[1])
        print("fuck")
        # fusion_layer --> scene_layer
        self.scene_layer   = tf.nn.elu(tf.matmul(self.fusion_layer, self.global_AC.universal_net_params[2]) + self.global_AC.universal_net_params[3])
        # scene_layer --> prob
        self.global_logits = tf.matmul(self.scene_layer, self.global_AC.universal_net_params[4]) + self.global_AC.universal_net_params[5]

        self.global_a_prob = tf.nn.softmax(self.global_logits)

    def _build_universal_network_params_dict(self):
        # fusion
        self.w_fusion = generate_fc_weight(shape=[4096, 512], name='global_w_fusion')
        self.b_fusion = generate_fc_bias(shape=[512]        , name='global_b_fusion')
        # scene
        self.w_scene  = generate_fc_weight(shape=[512, 512]  , name='global_w_scene')
        self.b_scene  = generate_fc_bias(shape=[512]         , name='global_b_scene')
        # actor
        self.w_actor  = generate_fc_weight(shape=[512, 4] , name='global_w_actor')
        self.b_actor  = generate_fc_bias(shape=[4]        , name='global_b_actor')

        self.universal_net_params = [ self.w_fusion,self.b_fusion, self.w_scene,self.b_scene, self.w_actor, self.b_actor]

    def _build_encode_net(self,input_image):
        conv1 = tf.nn.conv2d(input_image, self.global_AC.conv1_weight, strides=[1, 4, 4, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,self.global_AC.conv1_bias))
        conv2 = tf.nn.conv2d(relu1, self.global_AC.conv2_weight, strides=[1, 2, 2, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,self.global_AC.conv2_bias))
        conv3 = tf.nn.conv2d(relu2, self.global_AC.conv3_weight, strides=[1, 2, 2, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, self.global_AC.conv3_bias))
        conv4 = tf.nn.conv2d(relu3, self.global_AC.conv4_weight, strides=[1, 2, 2, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4,self.global_AC.conv4_bias))
        flatten_feature = flatten(relu4)
        state_feature = tf.nn.elu(tf.matmul(flatten_feature, self.global_AC.fc_weight) + self.global_AC.fc_bias)
        return state_feature

    def _build_special_net(self,state_feature):
        # special_actor
        w_actor = generate_fc_weight(shape=[2048, 4], name='special_w_a')
        b_actor = generate_fc_bias(shape=[4], name='special_b_a')
        special_logits = tf.matmul(state_feature, w_actor) + b_actor
        prob = tf.nn.softmax(special_logits)
        # special_critic
        w_critic = generate_fc_weight(shape=[2048, 1], name='special_w_c')
        b_critic = generate_fc_bias(shape=[1], name='special_b_c')
        value = tf.matmul(state_feature, w_critic) + b_critic
        a_params = [w_actor,  b_actor ]
        c_params = [w_critic, b_critic]
        return prob, value, a_params, c_params



