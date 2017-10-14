import collections
import sys
import os
import numpy as np
import random
import torch
from mlp_regressor import MlpRegressor_P
from torch.utils.data import TensorDataset
from wxx_cnn import CNN
import io
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')


class FB_RL:
    def __init__(self, nn_config_dict, alpha, is_CNN = False, is_CUDA = False):
        self.reward_dict = collections.defaultdict(lambda :0)
        self.computed_reward_dict = collections.defaultdict(lambda: 0)
        self.action_dict = collections.defaultdict(lambda :'space')
        self.env_dict = collections.defaultdict(lambda :0)
        self.Q_dict = collections.defaultdict(lambda :0)
        self.step = 0
        self.alpha = alpha # RL learning rate (1-a)*old_Q + a * (....)
        self.is_CUDA = is_CUDA
        # ----------------------------------------------------------------
        # ANN config
        # ----------------------------------------------------------------
        learning_rate_init = nn_config_dict['learning_rate_init']
        hidden_layer_sizes = nn_config_dict['hidden_layer_sizes']
        is_verbose = nn_config_dict['is_verbose']
        tol = nn_config_dict['tol']
        max_iter = nn_config_dict['max_iter']
        fig_wid = nn_config_dict['fig_wid']
        fig_len = nn_config_dict['fig_len']

        # ---------------------------------------------------------------
        if is_CNN:
            EPOCH = nn_config_dict['EPOCH']
            BATCH_SIZE = nn_config_dict['BATCH_SIZE']

            self.space_regressor = CNN(EPOCH, BATCH_SIZE, learning_rate_init, fig_wid, fig_len,verbose = is_verbose)
            self.idle_regressor = CNN(EPOCH, BATCH_SIZE, learning_rate_init,fig_wid, fig_len, verbose = is_verbose)
            if self.is_CUDA:
                self.space_regressor.cuda()
                self.idle_regressor.cuda()
            #

        else:

            # MLP
            self.space_regressor = MlpRegressor_P(hidden_layer_sizes, learning_rate_init=learning_rate_init,
                                                  verbose = is_verbose, tol = tol, max_iter = max_iter)
            self.idle_regressor = MlpRegressor_P(hidden_layer_sizes, learning_rate_init=learning_rate_init,
                                                 verbose=is_verbose, tol = tol, max_iter = max_iter)
            #

    def compute_reward(self):
        max_step = max(list(self.reward_dict.keys()))
        for step, reward in self.reward_dict.items():
            #print ("step: ", step)

            # if step <= max_step - 8:
            #     next_step3_reward = self.reward_dict[step + 3]
            #     next_step4_reward = self.reward_dict[step + 4]
            #     next_step5_reward = self.reward_dict[step + 5]
            #     next_step6_reward = self.reward_dict[step + 6]
            #     next_step7_reward = self.reward_dict[step + 7]
            #     next_step8_reward = self.reward_dict[step + 8]
            #     reward = 0.04 * next_step3_reward +\
            #              0.04 * next_step4_reward + \
            #              0.04 * next_step5_reward + \
            #              0.04 * next_step6_reward + \
            #              0.04 * next_step7_reward + \
            #              0.8 * next_step8_reward
            #
            # elif step <= max_step - 7:
            #     next_step3_reward = self.reward_dict[step + 3]
            #     next_step4_reward = self.reward_dict[step + 4]
            #     next_step5_reward = self.reward_dict[step + 5]
            #     next_step6_reward = self.reward_dict[step + 6]
            #     next_step7_reward = self.reward_dict[step + 7]
            #     reward = 0.05 * next_step3_reward + \
            #              0.05 * next_step4_reward + \
            #              0.05 * next_step5_reward + \
            #              0.05 * next_step6_reward + \
            #              0.8 * next_step7_reward

            if step <= max_step - 6:
                next_step3_reward = self.reward_dict[step + 3]
                next_step4_reward = self.reward_dict[step + 4]
                next_step5_reward = self.reward_dict[step + 5]
                next_step6_reward = self.reward_dict[step + 6]
                reward = 0.05 * next_step3_reward + \
                         0.05 * next_step4_reward + \
                         0.4 * next_step5_reward + \
                         0.5 * next_step6_reward

            elif step <= max_step - 5:
                next_step3_reward = self.reward_dict[step + 3]
                next_step4_reward = self.reward_dict[step + 4]
                next_step5_reward = self.reward_dict[step + 5]
                reward = 0.1*next_step3_reward + 0.2*next_step4_reward + 0.6*next_step5_reward

            elif step <= max_step - 4:

                next_step2_reward = self.reward_dict[step + 2]
                next_step3_reward = self.reward_dict[step + 3]
                next_step4_reward = self.reward_dict[step + 4]
                reward = 0.1*next_step2_reward + 0.1*next_step3_reward + 0.7*next_step4_reward

            elif step <= max_step - 3:
                next_step1_reward = self.reward_dict[step + 1]
                next_step2_reward = self.reward_dict[step + 2]
                next_step3_reward = self.reward_dict[step + 3]
                reward = 0.1*next_step1_reward + 0.1*next_step2_reward + 0.8*next_step3_reward

            elif step <= max_step - 2:
                next_step1_reward = self.reward_dict[step + 1]
                next_step2_reward = self.reward_dict[step + 2]
                reward = 0.1 * next_step1_reward + 0.9 * next_step2_reward

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

        #print ("self.action_dict.items(): ", self.action_dict.items())

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

        #print ("space_feature_list_all: ", space_feature_list_all)
        #print ("space_value_list: ", space_value_list)
        #sys.exit()


        return space_feature_list_all, space_value_list, idle_feature_list_all, idle_value_list

    def get_action(self, feature_list, img_shape, is_CNN = False, random_prob = 0.2, game = 'fb'):

        if is_CNN:
            feature_array = np.array([[np.array(feature_list).reshape(*img_shape)]])
            #feature_array = np.array([[x.reshape(*img_shape)] for x in feature_list])
            if self.is_CUDA:
                feature_tensor = torch.from_numpy(feature_array).float().cuda()
            else:
                feature_tensor = torch.from_numpy(feature_array).float()
            space_reward_value = self.space_regressor.regressor_dev(feature_tensor)
            idle_reward_value = self.idle_regressor.regressor_dev(feature_tensor)
        else:
            feature_array = np.array(feature_list)
            feature_array = self.data_convert_before_training(feature_array, single_sample = True)
            space_reward_value = self.space_regressor.regressor_dev(feature_array)
            idle_reward_value = self.idle_regressor.regressor_dev(feature_array)


        print ("space: {}, idle: {}".format(space_reward_value, idle_reward_value))

        if game == 'fb':
            action_list = ['space', 'idle']
        elif game == 'trex':
            action_list = ['space', 'idle','idle','idle']
        else:
            print ("Please type the right game!!")
            sys.exit()
        random_number = random.random()
        if random_number <= random_prob:
            #use_random_action = True
            action = random.sample(action_list, 1)[0]
            print ("Random prob now---{}, Random action---{}!!".format(random_prob, action))
        else:
            if space_reward_value >= idle_reward_value:
                action = 'space'
            else:
                action = 'idle'

        return action


    def data_convert_before_training(self, input_list, single_sample = False, single_feature = False):
        if single_sample:
            output_list = input_list.reshape(1,-1)
        elif single_feature:
            output_list = [x.reshape(-1,1) for x in input_list]
        else:
            print ("Make sure single sample or single feature is True")
            sys.exit()
        return output_list


    def train_rl(self, img_shape, file1_name, file2_name, is_CNN = False):

        # # get the data to train
        # space_feature_list_all, space_value_list, idle_feature_list_all, idle_value_list =\
        #     self.get_action_feature_reward()
        # #


        idle_feature_list_all = []

        with open (file1_name, 'r') as f1:
            f1_readlines = f1.readlines()
            space_feature_list_all = f1_readlines[::2]
            space_value_list = f1_readlines[1::2]
            space_feature_list_all = [np.array([int(x2) for x2 in x.strip().split(',')]) for x in space_feature_list_all]
            space_value_list = [float(x) for x in space_value_list]
        with open (file2_name, 'r') as f2:
            f2_readlines = f2.readlines()
            idle_feature_list_all = f2_readlines[::2]
            idle_value_list = f2_readlines[1::2]
            idle_feature_list_all = [np.array([int(x2) for x2 in x.strip().split(',')]) for x in idle_feature_list_all]
            idle_value_list = [float(x) for x in idle_value_list]


        # convert data format
        #space_feature_list_all = self.data_convert_before_training(space_feature_list_all, single_sample = True)
        #space_value_list = self.data_convert_before_training(space_value_list, single_feature = True)
        #idle_feature_list_all = self.data_convert_before_training(idle_feature_list_all, single_sample = True)
        #idle_value_list = self.data_convert_before_training(idle_value_list, single_feature = True)
        #

        # --------------------------------------------------------------------------------------------------------------
        # CNN, convert to tensor
        # --------------------------------------------------------------------------------------------------------------
        if is_CNN:
            print ("is_CNN: ", is_CNN)
            # space
            space_feature_array = np.array([[x.reshape(*img_shape)] for x in space_feature_list_all])
            space_value_array = np.array(space_value_list)
            space_feature_tensor = torch.from_numpy(space_feature_array).float()
            space_value_tensor = torch.from_numpy(space_value_array)
            space_action_dataset = TensorDataset(space_feature_tensor, space_value_tensor)
            #

            # idle
            idle_feature_array = np.array([[x.reshape(*img_shape)] for x in idle_feature_list_all])
            idle_value_array = np.array(idle_value_list)
            idle_feature_tensor = torch.from_numpy(idle_feature_array).float()
            idle_value_tensor = torch.from_numpy(idle_value_array)
            idle_action_dataset = TensorDataset(idle_feature_tensor, idle_value_tensor)
            #

            #print ("space_feature_tensor: {}".format(space_feature_tensor))
            #print ("space_value_tensor: {}".format(space_value_tensor))
            #print ("action_space_dataset: {}".format(action_space_dataset))
            # print ("space_feature_tensor: ", space_feature_tensor)
            # print ("space_value_tensor: ", space_value_tensor)


            self.space_regressor.regressor_train(space_action_dataset)
            self.idle_regressor.regressor_train(idle_action_dataset)


        # --------------------------------------------------------------------------------------------------------------
        else:
            print("space_feature_list_all_len: ", len(space_feature_list_all))
            print("space_feature_list_size: ", len(space_feature_list_all[0]))
            print("space_value_list_size: ", len(space_value_list))
            print("space_regressor training...")
            self.space_regressor.regressor_train(space_feature_list_all, space_value_list)

            if idle_feature_list_all:
                print ("idle_feature_list_all_len: ", len(idle_feature_list_all))
                print ("idle_feature_list_all_size: ", len(idle_feature_list_all[0]))
                print ("idle_value_list_size: ", len(idle_value_list))
                print ("idle_regressor training...")
                self.idle_regressor.regressor_train(idle_feature_list_all, idle_value_list)


    def save_Q_learning_data(self, file1_name, file2_name, space_kept_number = 500, idle_kept_number = 500):


        # get the data of last iteration
        space_feature_list_all, space_value_list, idle_feature_list_all, idle_value_list =\
            self.get_action_feature_reward()
        #


        # detect whether Q_learning file exist
        file1_name = file1_name
        file2_name = file2_name

        is_file1 = os.path.isfile(file1_name)
        is_file2 = os.path.isfile(file2_name)
        #


        # ---------------------------------------------------------------------------------------------------------
        # update for space and idle
        # ---------------------------------------------------------------------------------------------------------
        space_start_read_index = 0
        file_list = [is_file1, is_file2]
        file_name_list = [file1_name, file2_name]
        for i, is_file in enumerate(file_list):
            file_name = file_name_list[i]
            if file_name == file1_name:
                feature_list_all = space_feature_list_all
                value_list = space_value_list
            elif file_name == file2_name:
                feature_list_all = idle_feature_list_all
                value_list = idle_value_list

            if is_file:
                with open (file_name, 'r') as file1:
                    f_readlines_list = file1.readlines()
                    sample_feature_list = [tuple([int(x2) for x2 in x.strip().split(',')]) for x in f_readlines_list[::2]]
                    sample_value_list = [float(x.strip()) for x in f_readlines_list[1::2]]
                    sample_feature_set = set(sample_feature_list)

                    for i, space_feature in enumerate(feature_list_all):
                        space_feature_tuple = tuple(space_feature)
                        new_space_value = value_list[i]
                        if space_feature_tuple in sample_feature_set:
                            old_index = sample_feature_list.index(space_feature_tuple)
                            old_value = sample_value_list[old_index]

                            # RL update
                            update_space_value = (1-self.alpha) * old_value + self.alpha * new_space_value
                            #

                            # remove old value and features if new features overlap with the old one
                            sample_feature_list.pop(old_index)
                            sample_value_list.pop(old_index)
                            #

                        else:
                            update_space_value = new_space_value


                        sample_feature_list.append(space_feature_tuple)
                        sample_value_list.append(update_space_value)
            else:
                sample_feature_list = feature_list_all
                sample_value_list = value_list

            # write new files
            space_start_read_index = len(sample_feature_list) - space_kept_number
            if space_start_read_index < 0:
                space_start_read_index = 0

            #print ("-------------------------------------")
            #print ("space_start_read_index: ", space_start_read_index)
            #print ("sample_feature_list: ", sample_feature_list)
            #print ("sample_value_list: ", sample_value_list)


            with open (file_name, 'w') as file1:
                for i, feature in enumerate(sample_feature_list):
                    if i < space_start_read_index:
                        continue
                    value_str = str(sample_value_list[i])
                    feature_str = ','.join([str(x) for x in feature])
                    file1.write(feature_str + '\n')
                    file1.write(value_str + '\n')
            # ---------------------------------------------------------------------------------------------------------
        #sys.exit()


