import tensorflow as tf



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


"""
  对于glo_network，用来初始化参数的一些op
"""
def _build_encode_params_dict():
    # 300*400*3 image --> 2048-d features  编码图像
    conv1_weight = generate_conv2d_weight(shape=[5,5,3,8],name="conv1_weight")
    conv1_bais   = generate_conv2d_bias(shape=8,name='conv1_bias')
    conv2_weight = generate_conv2d_weight(shape=[3,3,8,16],name="conv2_weight")
    conv2_bais   = generate_conv2d_bias(shape=16,name='conv2_bias')
    conv3_weight = generate_conv2d_weight(shape=[3,3,16,32],name="conv3_weight")
    conv3_bais   = generate_conv2d_bias(shape=32,name='conv3_bias')
    conv4_weight = generate_conv2d_weight(shape=[3,3,32,64],name="conv4_weight")
    conv4_bais   = generate_conv2d_bias(shape=64,name='conv4_bias')
    fc_weight    = generate_fc_weight(shape=[8320,2048],name='fc_weight')
    fc_bias      = generate_fc_weight(shape=[2048],name='fc_bias')
    encode_params = [conv1_weight,conv1_bais,conv2_weight,conv2_bais,conv3_weight,conv3_bais,conv4_weight,conv4_bais,fc_weight,fc_bias]

    # current_state||next_state --> the action 逆向动力学
    weight_p_a    = generate_fc_weight(shape=[4096,4],name='p_a_weight')
    bias_p_a      = generate_fc_weight(shape=[4],name='p_a_bias')
    action_predict_params = [weight_p_a,bias_p_a]

    # current_state||action --> next_state  前向动力学
    weight_p_n_s    = generate_fc_weight(shape=[4096,4],name='p_ns_weight')
    bias_p_n_s      = generate_fc_weight(shape=[4],name='p_ns__bias')
    state_predict_params = [weight_p_n_s,bias_p_n_s]

    return encode_params,action_predict_params,state_predict_params

def _build_universal_network_params_dict():
    # fusion
    w_fusion = generate_fc_weight(shape=[4096, 512], name='global_w_fusion')
    b_fusion = generate_fc_bias(shape=[512]        , name='global_b_fusion')
    # scene
    w_scene  = generate_fc_weight(shape=[512, 512]  , name='global_w_scene')
    b_scene  = generate_fc_bias(shape=[512]         , name='global_b_scene')
    # actor
    w_actor  = generate_fc_weight(shape=[512, 4] , name='global_w_actor')
    b_actor  = generate_fc_bias(shape=[4]        , name='global_b_actor')
    a_params = [w_fusion, b_fusion,w_scene, b_scene, w_actor,  b_actor]
    return a_params

def _build_special_params_dict():
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
    return a_params_dict,c_params_dict

"""
  网络的初始化
"""
# 初始化一个编码网络  image --> 2048-d feature
def _build_encode_net(input_image,encode_params):
    conv1 = tf.nn.conv2d(input_image, encode_params[0], strides=[1, 4, 4, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, encode_params[1]))
    conv2 = tf.nn.conv2d(relu1, encode_params[2], strides=[1, 2, 2, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, encode_params[3]))
    conv3 = tf.nn.conv2d(relu2, encode_params[4], strides=[1, 2, 2, 1], padding='SAME')
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, encode_params[5]))
    conv4 = tf.nn.conv2d(relu3, encode_params[6], strides=[1, 2, 2, 1], padding='SAME')
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, encode_params[7]))
    flatten_feature = flatten(relu4)
    state_feature = tf.nn.elu(tf.matmul(flatten_feature, encode_params[8]) + encode_params[9])
    return state_feature

# 初始化一个target-universal 网络，输入为current_state_feature和target_state_feature, 输出为acrion_distribution
def _build_global_net(state_feature,target_feature,universal_params):
    # s_encode||t_encode --> concat
    concat = tf.concat([state_feature, target_feature], axis=1)  # s_encode||t_encode --> concat
    # concat --> fusion_layer
    fusion_layer = tf.nn.elu(tf.matmul(concat, universal_params[0]) + universal_params[1])
    # fusion_layer --> scene_layer
    scene_layer = tf.nn.elu(tf.matmul(fusion_layer, universal_params[2]) + universal_params[3])
    # scene_layer --> prob
    self.global_logits = tf.matmul(scene_layer, universal_params[4]) + universal_params[5]
    prob = tf.nn.softmax(self.global_logits)
    return prob

# 初始化一个target_special_network
def _build_special_net(state_feature):
    # special_actor
    w_actor = generate_fc_weight(shape=[2048, 4], name='special_w_a')
    b_actor = generate_fc_bias(shape=[4], name='special_b_a')
    self.special_logits = tf.matmul(state_feature, w_actor) + b_actor
    prob = tf.nn.softmax(self.special_logits)
    # special_critic
    w_critic = generate_fc_weight(shape=[2048, 1], name='special_w_c')
    b_critic = generate_fc_bias(shape=[1], name='special_b_c')
    value = tf.matmul(state_feature, w_critic) + b_critic
    a_params = [w_actor,  b_actor ]
    c_params = [w_critic, b_critic]
    return prob, value, a_params, c_params

# 初始化action_predict网络 和 state_predict网络
def _build_action_state_predict_net(current_image,transition_action,next_image
                                    ,encode_params,state_predict_params,action_predict_params):
    # state_image  = tf.placeholder(tf.float32,[None,300,400,3],name=scope+'_current_image')
    # next_image   = tf.placeholder(tf.float32,[None,300,400,3],name=scope+'_next_image')
    # self.action = tf.placeholder(tf.int32, [None, ],name="transition_action")

    current_feature = _build_encode_net(input_image=current_image,encode_params=encode_params)
    next_feature = _build_encode_net(input_image=next_image,encode_params=encode_params)

    # current_state||next_state --> action
    cur_next_concat = tf.concat([current_feature, next_feature], axis=1)
    action_predicted = tf.nn.elu(tf.matmul(cur_next_concat, action_predict_params[0]) + action_predict_params[1])

    # self.action_predict_loss = tf.reduce_sum(tf.log(action_predicted+1e-9)*tf.one_hot(transition_action,4,dtype=tf.float32),
    #                                          axis=1,keep_dims=True)
    # self.action_predict_grads = [tf.clip_by_norm(item, 40) for item in
    #                              tf.gradients(self.action_predict_loss, encode_params+state_predict_params)]
    # self.update_action_predict_op = self.OPT_A.apply_gradients(list(zip(self.action_predict_grads, encode_params+state_predict_params)))

    # current_state||action --> next_state
    self.cur_action_concat = tf.concat([current_feature, self.action], axis=1)
    state_predict =  tf.nn.elu(tf.matmul(cur_next_concat, state_predict_params[0]) + state_predict_params[1])

    # loss_raw = tf.subtract(tf.stop_gradient(self.next_feature),state_predict)
    # state_predict_loss =  tf.reduce_mean(tf.square(loss_raw))
    # self.state_predict_grads = [tf.clip_by_norm(item, 40) for item in
    #                             tf.gradients(state_predict_loss, encode_params+state_predict_params)]
    # self.update_state_predict_op = self.OPT_A.apply_gradients(list(zip(self.state_predict_grads, encode_params+state_predict
