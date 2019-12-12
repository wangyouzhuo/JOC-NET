from Environment.env import *
from utils.global_episode_count import _get_train_count,_add_train_count
from utils.global_episode_count import _get_steps_count,_add_steps_count
from utils.global_episode_count import _append_kl_list,_reset_kl_list,_get_kl_mean,_get_adaptive_learning_rate
from utils.global_episode_count import _increase_kl_beta,_decrease_kl_beta,_get_kl_beta
from utils.global_episode_count import _init_result_mean_list,_append_result_mean_list,_reset_result_mean_list,_get_train_mean_roa_reward
from utils.global_episode_count import _init_reward_roa_show,_append_reward_roa_show,_get_reward_roa_show
from utils.global_episode_count import _get_max_reward,_update_max_reward
from config.config import *
import numpy as np
from worker.worker import Worker
from config.config import *



def count_list(target):
    if len(target)>0:
        count = 0
        for item in target:
            if item<0.2:
                count = count + 1
        return count*1.0/len(target)
    else:
        return 0


class Glo_Worker(Worker):

    def __init__(self, name,globalAC,sess,coord,device,type='Target_Universal'):
        super().__init__(name=name, globalAC=globalAC, sess=sess, coord=coord,type=type,device=device)


    def work(self):
        buffer_s, buffer_a, buffer_r, buffer_t = [], [], [], []
        buffer_s_next = []
        buffer_q = []
        buffer_vgg_state_next = []
        buffer_vgg_state = []
        while not self.coord.should_stop() and _get_train_count() < MAX_GLOBAL_EP:
            EPI_COUNT = _add_train_count()
            # self.image_current ,self.vgg_current,self.image_terminal,self.vgg_terminal
            image_current ,vgg_current,image_terminal,vgg_terminal = self.env.reset_env()
            target_id = self.env.terminal_state_id
            ep_r = 0
            step_in_episode = 0
            a_lr = 0.00001
            while True:
                self.AC.load_weight(target_id=target_id)
                a,global_prob  = self.AC.glo_choose_action(image_current, image_terminal)
                # self.image_current ,self.vgg_current,reward,self.image_terminal,self.vgg_terminal,(currrent_id,next_id)
                image_current_next,vgg_current_next,r,done,info = self.env.take_action(a)
                _add_steps_count()
                ep_r += r
                buffer_s.append(image_current)
                buffer_s_next.append(image_current_next)
                buffer_a.append(a)
                buffer_t.append(image_terminal)
                buffer_r.append(r)
                buffer_vgg_state_next.append(vgg_current_next)
                buffer_vgg_state.append(vgg_current)
                if step_in_episode % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    buffer_v = self.AC.get_special_value(feed_dict={self.AC.state_vgg_feature: np.vstack(buffer_vgg_state)})
                    if done:
                        buffer_v[-1] = 0  # terminal
                    buffer_advantage = [0]*len(buffer_s)
                    buffer_v_next = self.AC.get_special_value(feed_dict={self.AC.state_vgg_feature: np.vstack(buffer_vgg_state_next)})
                    for i in range(len(buffer_r)):
                        v_next = buffer_v_next[i]
                        reward = buffer_r[i]
                        q = reward + GAMMA*v_next
                        advantage = q-buffer_v[i]
                        buffer_advantage[i] = advantage

                    buffer_s, buffer_a, buffer_t = np.array(buffer_s), np.array(buffer_a), np.array(buffer_t)
                    buffer_advantage = np.vstack(buffer_advantage)
                    feed_dict = {
                        self.AC.state_image: buffer_s,
                        self.AC.action: buffer_a,
                        self.AC.target_image: buffer_t,
                        self.AC.adv:buffer_advantage,
                        self.AC.learning_rate:a_lr,
                       }
                    # print(len(buffer_advantage),len(buffer_s))
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r, buffer_t,buffer_s_next = [], [], [], [],[]
                image_current = image_current_next
                step_in_episode += 1
                if done or step_in_episode >= MAX_STEP_IN_EPISODE:
                    if done:
                        roa = round((self.env.short_dist * 1.0 / step_in_episode), 4)
                    else:
                        roa = 0.000
                    _append_result_mean_list(roa,ep_r)

                    if EPI_COUNT%10 == 0:
                        roa_mean_train,reward_mean_train,length_train = _get_train_mean_roa_reward()
                        _reset_result_mean_list()
                        #roa_eva,reward_eva,lenght_eva = self.evaluate()
                        roa_eva,reward_eva,lenght_eva = 0,0,0
                        _append_reward_roa_show(
                                                reward_evaluate = reward_eva       ,
                                                roa_evaluate    = roa_eva          ,
                                                reward_train    = reward_mean_train,
                                                roa_train       = roa_mean_train
                                                )
                        # print("Train!     Epi:%6s || Glo_Roa:%6s  || Glo_Reward:%7s         Evaluate!  Epi:%6s || Roa_mean:%6s || Reward_mean:%7s "
                        #   %(EPI_COUNT, round(roa_mean_train, 3), round(reward_mean_train, 2),EPI_COUNT,round(roa_eva,4),round(reward_eva,3)))
                        print("%s Train %s targets!     Epi:%6s || Glo_Roa:%6s  || Glo_Reward:%7s "
                            %(SOFT_LOSS_TYPE,len(TARGET_ID_LIST) ,EPI_COUNT ,round(roa_mean_train, 3), round(reward_mean_train,2)))
                        # if reward_mean_train>9.0 and reward_mean_train > _get_max_reward() and roa_mean_train>0.5:
                        #     _update_max_reward(reward_mean_train)
                        #     path = ROOT_PATH + "/weight/"+str(len(TARGET_ID_LIST))+"_Targets_"+SOFT_LOSS_TYPE+"_.ckpt"
                        #     self.AC.global_AC.store(path)
                        #     print("Store Weight!")

                    break

    def evaluate(self):
        roa,reward,length = super().evaluate()
        return roa,reward,length


    def evaluate_generalization(self):
        env = load_thor_env(scene_name='bedroom_04', random_start=True, random_terminal=True,
                            whe_show=False, terminal_id=None, start_id=None, whe_use_image=False,
                            whe_flatten=False, num_of_frames=1)
        self.env = env





