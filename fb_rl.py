import collections
import sys
from mlp_regressor import MlpRegressor_P

class FB_RL:
    def __init__(self, nn_config_dict):
        self.reward_dict = collections.defaultdict(lambda :0)
        self.computed_reward_dict = collections.defaultdict(lambda: 0)
        self.action_dict = collections.defaultdict(lambda :'space')
        self.env_dict = collections.defaultdict(lambda :0)
        self.Q_dict = collections.defaultdict(lambda :0)
        self.step = 0

        # ----------------------------------------------------------------
        # ANN config
        # ----------------------------------------------------------------
        learning_rate_init = nn_config_dict['learning_rate_init']
        hidden_layer_sizes = nn_config_dict['hidden_layer_sizes']
        # ---------------------------------------------------------------


        # build regressor
        self.space_regressor = MlpRegressor_P(hidden_layer_sizes, learning_rate_init=learning_rate_init)
        self.idle_regressor = MlpRegressor_P(hidden_layer_sizes, learning_rate_init=learning_rate_init)
        #

    def compute_reward(self):
        max_step = max(list(self.reward_dict.keys()))
        for step, reward in self.reward_dict.items():
            print ("step: ", step)

            if step <= max_step - 5:

                next_step3_reward = self.reward_dict[step + 3]
                next_step4_reward = self.reward_dict[step + 4]
                next_step5_reward = self.reward_dict[step + 5]
                reward = 0.1*next_step3_reward + 0.1*next_step4_reward + 0.8*next_step5_reward

            elif step <= max_step - 4:

                next_step2_reward = self.reward_dict[step + 2]
                next_step3_reward = self.reward_dict[step + 3]
                next_step4_reward = self.reward_dict[step + 4]
                reward = 0.1*next_step2_reward + 0.1*next_step3_reward + 0.8*next_step4_reward

            elif step <= max_step - 3:
                next_step1_reward = self.reward_dict[step + 1]
                next_step2_reward = self.reward_dict[step + 2]
                next_step3_reward = self.reward_dict[step + 3]
                reward = 0.1*next_step1_reward + 0.1*next_step2_reward + 0.8*next_step3_reward

            elif step <= max_step - 2:
                next_step1_reward = self.reward_dict[step + 1]
                next_step2_reward = self.reward_dict[step + 2]
                reward = 0.2 * next_step1_reward + 0.8 * next_step2_reward

            # this is the final step
            elif step <= max_step - 1:
                reward = self.reward_dict[step + 1]
            elif step == max_step:
                pass
            else:
                print ("ERROR")
                sys.exit()

            self.computed_reward_dict[step] = reward

    def get_action_feature_reward(self):

        # space action
        space_feature_list_all = []
        space_value_list = []
        #

        # idle action
        idle_feature_list_all = []
        idle_value_list = []
        #

        for step, action in self.action_dict.items():
            if action == 'space':
                feature_list = self.env_dict[step]
                reward = self.computed_reward_dict[step]
                space_feature_list_all.append(feature_list)
                space_value_list.append(reward)
            elif action == 'idle':
                feature_list = self.env_dict[step]
                reward = self.computed_reward_dict[step]
                idle_feature_list_all.append(feature_list)
                idle_value_list.append(reward)
            else:
                print ("error")
                sys.exit()
        return space_feature_list_all, space_value_list, idle_feature_list_all, idle_value_list

    def get_action(self, feature_list):


        space_reward_value = self.space_regressor.regressor_dev(feature_list)
        idle_reward_value = self.idle_regressor.regressor_dev(feature_list)

        print ("space: {}, idle: {}".format(space_reward_value, idle_reward_value))

        if space_reward_value >= idle_reward_value:
            action = 'space'
        else:
            action = 'idle'

        return action

    def train_rl(self):

        # get the data to train
        space_feature_list_all, space_value_list, idle_feature_list_all, idle_value_list =\
            self.get_action_feature_reward()
        #
        print ("space_feature_list_all_len: ", len(space_feature_list_all))
        print ("space_feature_list_size: ", len(space_feature_list_all[0]))
        print ("space_value_list_size: ", len(space_value_list))
        print ("space_regressor training...")
        self.space_regressor.regressor_train(space_feature_list_all, space_value_list,
                                             save_clsfy_path = 'space_regressor')


        if idle_feature_list_all:
            print ("ild_feature_list_all_len: ", len(idle_feature_list_all))
            print ("idle_feature_list_all_size: ", len(idle_feature_list_all[0]))
            print ("idle_value_list_size: ", len(idle_value_list))
            print("idle_regressor training...")
            self.idle_regressor.regressor_train(idle_feature_list_all, idle_value_list,
                                                save_clsfy_path='idle_regressor')



