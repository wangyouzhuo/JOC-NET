from utils.op import *
import tensorflow as tf
import numpy as np
from model.model_op import _build_encode_net,_build_encode_params_dict,_build_universal_network_params_dict,_build_special_params_dict
from model.model_op import _build_special_net,_build_global_net


class ACNet(object):
    def __init__(self, scope,session,device,type,globalAC=None):
        tf.set_random_seed(50)

        with tf.device(device):
            self.session = session

            if scope == 'Global_Net':
                with tf.variable_scope(scope):
                    self.encode_params,self.action_predict_params,self.state_predict_params = _build_encode_params_dict()
                    self.universal_net_params = _build_universal_network_params_dict()
                    self.special_params_dict  = _build_special_params_dict()


            elif type == 'Target_Special':
                with tf.variable_scope(scope):

                    self.global_AC = globalAC

                    self.state_image = tf.placeholder(tf.float32,[None,300,400,3], 'State_image')
                    self.a = tf.placeholder(tf.int32, [None, ], 'Action')

                    self.state_feature = _build_encode_net(input_image=self.state_image,encode_params=self.global_AC.encode_params)

                    self.special_v_target = tf.placeholder(tf.float32, [None, 1], 'special_V_target')

                    self.OPT_A = tf.train.RMSPropOptimizer(LR_SPE_A, name='Spe_RMSPropA')
                    self.OPT_C = tf.train.RMSPropOptimizer(LR_SPE_C, name='Spe_RMSPropC')

                    self.special_a_prob,self.special_v,self.special_a_params,self.special_c_params =\
                        _build_special_net(input=self.state_feature)

                    self._prepare_special_loss(scope)

                    self._prepare_special_grads(scope)

                    self._prepare_special_update_op(scope)

                    self._prepare_special_pull_op(scope)

            elif type == 'Target_General':

                self.global_AC = globalAC

                self.state_image = tf.placeholder(tf.float32 ,[None,300,400,3], 'State_image')
                self.target_image = tf.placeholder(tf.float32,[None,300,400,3], 'Target_image')

                self.state_feature  = _build_encode_net(input_image=self.state_image,encode_params=self.global_AC.encode_params)
                self.target_feature = _build_encode_net(input_image=self.state_image,encode_params=self.global_AC.encode_params)

                self.target_image = tf.placeholder(tf.float32, [None, self.dim_s], 'Target_image')
                self.target_feature = self._build_encode_net(scope,self.target_image,self.global_AC.global_encode_params)

                self.a        = tf.placeholder(tf.int32, [None, ], 'Action')
                self.adv      = tf.placeholder(tf.float32, [None,1], 'Advantage')
                self.kl_beta  = tf.placeholder(tf.float32, [None,], 'KL_BETA')

                self.learning_rate = tf.placeholder(tf.float32, None, 'Learning_rate')

                self.OPT_A = tf.train.RMSPropOptimizer(self.learning_rate, name='Glo_RMSPropA')

                # target_general_network
                self.global_a_prob ,self.global_a_params = \
                    _build_global_net(state_feature=self.state_feature,target_feature=self.target_feature,
                                      universal_params = self.global_AC.universal_net_params)
                # target_special_network
                self.special_a_prob,self.special_v,self.special_a_params,self.special_c_params =\
                    _build_special_net(input=self.state)


                self._prepare_global_loss(scope)

                self._prepare_global_grads(scope)

                # self._prepare_special_update_op(scope)
                self._prepare_global_update_op(scope)

                self._prepare_global_pull_op(scope)
                self._prepare_special_pull_op(scope)

                self._prepare_kl_devergance(scope)

                self._prepare_many_goals_loss_grads_update



    def _prepare_global_loss(self,scope):
        with tf.name_scope(scope+'global_loss'):

            with tf.name_scope('global_a_loss'):
                # prob from target_special_net (dont update)
                p_target = tf.stop_gradient(self.special_a_prob)
                # prob from global_special_net (need update)
                p_update = self.global_a_prob

                self.spe_actor_reg_loss = -tf.reduce_mean(p_target*tf.log(tf.clip_by_value(p_update,1e-10,1.0)))

                glo_log_prob = tf.reduce_sum(tf.log(self.global_a_prob + 1e-5)*tf.one_hot(self.a, self.dim_a, dtype=tf.float32),
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
                              tf.gradients(self.global_a_loss, self.global_a_params+self.global_AC.encode_params)]

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

    def _prepare_global_pull_op(self,scope):
        with tf.name_scope(scope+'pull_global_params'):
            self.pull_a_params_global = [l_p.assign(g_p) for l_p, g_p in
                                         zip(self.global_a_params, self.global_AC.universal_net_params)]

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
        self.session.run([self.pull_a_params_global])

    def pull_special(self,target_id):  # run by a local
        self.session.run([self.pull_a_params_special_dict[target_id]
                             ,self.pull_c_params_special_dict[target_id]])

    def spe_choose_action(self, s, t):  # run by a local
        prob_weights = self.session.run(self.special_a_prob, feed_dict={self.s: s[np.newaxis, :]} )
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action,prob_weights

    def glo_choose_action(self, s, t):  # run by a local
        prob_weights = self.session.run(self.global_a_prob, feed_dict={self.s: s[np.newaxis, :],self.t: t[np.newaxis, :]} )
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

        current_feature = _build_encode_net(input_image=self.current_image,encode_params=encode_params)
        next_feature = _build_encode_net(input_image=self.next_image,encode_params=encode_params)

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
        self.update_state_predict_op = self.OPT_A.apply_gradients(list(zip(self.state_predict_grads, encode_params+state_predict

    def _prepare_many_goals_loss_grads_update(self):
        self.action_many_goals = tf.placeholder(tf.int32, [None, ], 'Action_mg')
        self.mg_loss = -tf.reduce_mean(self.global_a_prob*tf.log(tf.clip_by_value(tf.one_hot(self.action_many_goals, 4, dtype=tf.float32),1e-10,1.0)))
        self.mg_grads =  self.special_a_grads = [tf.clip_by_norm(item, 40) for item in
                    tf.gradients(self.mg_loss,self.global_a_params+self.global_AC.encode_params)]
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



