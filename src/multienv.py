import numpy as np
import json, csv
import cv2
import pickle
from src.utils import Foo
# from tfcode.nav_utils import global2loc
import os, sys
from navi_env import Environment

class MultiEnv():

    def __init__(self, envs, navi):
        self.envs = [Environment(env, navi) for env in envs]
        # TODO: Manually add start position for every house

    def reset(self, rng, single_target=None, multi_target=None):
        positions = []
        for env in self.envs:
            positions.extend(
                env.reset(rng, single_target=single_target, multi_target=multi_target))
        return positions

    def get_common_data(self):
        # TODO target_goal_location into multi-locations
        pass

    def get_step_data(self):
        outs = []
        for env in self.envs:
            outs.append(env.get_step_data())
        ret = {}

        for i in range(3):
            ret['locsmap_{:d}'.format(i)] = \
                np.concatenate([out['locsmap_{:d}'.format(i)] for out in outs], axis=0)
            ret['onehot_semantic_{:d}'.format(i)] = \
                np.concatenate([out['onehot_semantic_{:d}'.format(i)] for out in outs], axis=0)
        ret['if_reach_goal'] = \
            np.concatenate([out['if_reach_goal'] for out in outs], axis=0)
        return ret

    def get_step_data_names(self):
        names = []
        names += ['locsmap_{:d}'.format(i) for i in range(3)]
        names += ['onehot_semantic_{:d}'.format(i) for i in range(3)]
        names += ['if_reach_goal']
        return names

    def step(self, action_indices):
        positions = []
        for i in range(len(self.envs)):
            positions.extend(
                self.envs[i].step(action_indices[4 * i : 4 * (i + 1)]))

        return positions

    def get_batch_gt_actions(self):
        outs = []
        for env in self.envs:
            outs.append(env.get_batch_gt_actions())
        ret = np.concatenate([out for out in outs], axis=0)
        return ret