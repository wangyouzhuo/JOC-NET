from worker.global_worker import *
from worker.special_worker import *
import threading
import datetime
from utils.global_episode_count import _init_result_mean_list,_append_result_mean_list,_reset_result_mean_list
from utils.global_episode_count import _init_train_count,_get_train_mean_roa_reward
from utils.global_episode_count import _init_show_list,_get_show_list,_init_roa_list
from utils.global_episode_count import _init_kl_list,_append_kl_list,_get_kl_mean
from utils.global_episode_count import _init_kl_beta
from utils.global_episode_count import _init_steps_count,_add_steps_count,_reset_steps_count
from utils.global_episode_count import _init_reward_roa_show,_append_reward_roa_show,_get_reward_roa_show
from utils.global_episode_count import _init_target_special_roa_dict
from utils.global_episode_count import _init_targets_have_been_finished
from utils.global_episode_count import _init_max_reward
from model.model_hybrid import *
from config.config import *
import matplotlib.pyplot as plt
from utils.op import *



if __name__ == "__main__":

    with tf.device(device):

        config = tf.ConfigProto(allow_soft_placement=True)

        SESS = tf.Session(config=config)

        COORD = tf.train.Coordinator()

        N_S,N_A = 2048,4

        tf.set_random_seed(-1)

        # scope,session,device,type,globalAC=None

        GLOBAL_AC = ACNet(scope='global_weight_store',session=SESS,type='Weight_Store',device=device)  # we only need its params

        workers = []
        # Create worker
        for i in range(int(N_WORKERS * 0.8)):
            i_name = 'Spe_W_%i' % i  # worker name
            workers.append(Spe_Worker(name=i_name, globalAC=GLOBAL_AC, sess=SESS, coord=COORD, device=device))
        for i in range(int(N_WORKERS * 0.2)):
            i_name = 'Glo_W_%i' % i  # worker name
            workers.append(Glo_Worker(name=i_name, globalAC=GLOBAL_AC, sess=SESS, coord=COORD, device=device))

        # GLOBAL_AC._prepare_store()

        SESS.run(tf.global_variables_initializer())

    # if OUTPUT_GRAPH:
    #     if os.path.exists(LOG_DIR):
    #         shutil.rmtree(LOG_DIR)
    #     tf.summary.FileWriter(LOG_DIR, SESS.graph)
    _init_train_count()
    _init_show_list()
    _init_roa_list()
    _init_kl_list()
    _init_kl_beta()
    _init_steps_count()
    _init_result_mean_list()
    _init_reward_roa_show()
    _init_target_special_roa_dict()
    _init_targets_have_been_finished()
    _init_max_reward()



    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
    #
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M ')

    #  roa
    title = now_time

    REWARD_SHOW_TRAIN,ROA_SHOW_TRAIN,_,_ = _get_reward_roa_show()

    plt.figure(figsize=(20, 5))
    plt.figure(1)
    plt.axis([0,len(ROA_SHOW_TRAIN),0,1])
    plt.plot(np.arange(len(ROA_SHOW_TRAIN)), ROA_SHOW_TRAIN, color="r")
    plt.xlabel('hundred episodes')
    plt.ylabel('Total mean roa train !')
    title = 'ROA_%s_targets-my-net'%(len(TARGET_ID_LIST))
    plt.title(title+now_time)

    plt.figure(figsize=(20, 5))
    plt.figure(2)
    plt.axis([0,len(REWARD_SHOW_TRAIN),-20,20])
    plt.plot(np.arange(len(REWARD_SHOW_TRAIN)), REWARD_SHOW_TRAIN, color="b")
    plt.xlabel('hundred episodes')
    plt.ylabel('Total mean reward train!')
    title = 'Reward_%s_targets-my-net'%(len(TARGET_ID_LIST))

    filepath = ROOT_PATH + '/output_record/experiments_about_soft_imitation/'

    reward_file = str(len(TARGET_ID_LIST)) + "_targets_" + SOFT_LOSS_TYPE + "_of_reward.csv"
    output_record(result_list=REWARD_SHOW_TRAIN,attribute_name='mean_rewards',
                  target_count=len(TARGET_ID_LIST),file_path=filepath+reward_file)

    roa_file = str(len(TARGET_ID_LIST)) + "_targets_" + SOFT_LOSS_TYPE + "_of_roa.csv"
    output_record(result_list=ROA_SHOW_TRAIN,attribute_name='mean_roa',
              target_count=len(TARGET_ID_LIST),file_path=filepath+roa_file)

    plt.title(title+now_time)

    plt.show()
