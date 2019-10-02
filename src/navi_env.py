import numpy as np
import json, csv
import cv2
import pickle
from src.utils import Foo
# from tfcode.nav_utils import global2loc
import os, sys

CONFIG_PATH = "/media/z/Data/Object_Searching/code/NewMethods/config.json"
MetaDataColorMap = "/media/z/Data/Object_Searching/code/Environment/House3D/" + \
                   "House3D/metadata/colormap_coarse.csv"


# Reform Environment for batch data feeding
class Environment():

    def __init__(self, env, navi):
        """

        :param env: houseid
        :param navi: navigation parameters

        """
        self.env = env
        self.navi = navi

        cfg = json.load(open(CONFIG_PATH, 'rb'))
        code_dir = cfg['codeDir']
        self.dir = '%s/Environment/houses/%s' % (code_dir, self.env)
        self.actions = ['MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight', 'RotateLeft', 'RotateRight']

        self.positions = [None] * self.navi.batch_size
        self.lastActionSuccess = [True] * self.navi.batch_size

        # self.targets = [i[:-4] for i in os.listdir('%s/targets_image' % self.dir)]
        # TODO : notice!!! next time get targets and sort by char
        cat = json.load(open("%s/planner_data/targets_info_all.json" % self.dir, 'r'))
        self.targets = [i.lower() for i in cat.keys()]
        self._load_map()
        self._load_data()
        pass

    def _load_map(self):
        self.map = {}
        map_path = '%s/planner_data/map.txt' % self.dir
        with open(map_path, 'r') as f:
            for line in f:
                nums = line.split()
                if len(nums) == 2:
                    self.abs_start_pos = (float(nums[0]), float(nums[1]))
                else:
                    idx = int(nums[0])
                    pos = (int(nums[1]), int(nums[2]))
                    self.map[pos] = idx

    def _load_data(self):
        self.smap = pickle.load(open('%s/planner_data/localsmaps_31.pkl' % self.dir, 'rb'))
        self.fmap = pickle.load(open('%s/planner_data/fmap.pkl' % self.dir, 'rb'))
        self.target_locs = pickle.load(open('%s/planner_data/targetlocs.pkl' % self.dir, 'rb'))
        # self.start_locs = pickle.load(open('%s/planner_data/startlocs.pkl' % self.dir, 'rb'))
        self.gt_loc_aseq = pickle.load(open('%s/planner_data/states_aseq.pkl' % self.dir, 'rb'))
        self.locs = self.gt_loc_aseq.keys()

    def startatPos(self, position):
        return self.resetatPos(position)

    def resetatPos(self, position):
        assert len(position) == self.navi.batch_size, \
            "The reset or start positions should be list of size %d" % self.navi.batch_size
        self.positions = position
        self.lastActionSuccess = [True] * self.navi.batch_size
        self.reachgoal = [0. if (i[0], i[1]) == (j[0], j[1]) else 1.
                          for i, j in zip(self.positions, self.epi.target_locs)]
        return self.positions

    def reset(self, rng, single_target=None, multi_target=None):
        ## Get episode mission
        # Get target locations
        targets_idxs = np.arange(0, len(self.target_locs))
        rng.shuffle(targets_idxs)
        targets_idx = targets_idxs[:self.navi.batch_size]
        if single_target:
            # Single Target for all batch
            targets_idx = np.array([self.targets.index(single_target)] * self.navi.batch_size)
        if multi_target:
            assert len(multi_target) == self.navi.batch_size, \
                'number of multiple targets should be equal to batch size!'
            targets_idx = np.array([self.targets.index(tar) for tar in multi_target])

        epi_targets = [self.targets[target_idx] for target_idx in targets_idx]
        epi_target_locs = [list(self.target_locs[target]) for target in epi_targets]

        # Get starting locations
        # epi_start_locs_1 = []
        # epi_start_locs = [self.start_locs[epi_target] for epi_target in epi_targets]
        # for epi_start_loc in epi_start_locs:
        #     epi_start_locs_1.append(list(epi_start_loc[rng.randint(0, len(epi_start_loc))]))

        epi_start_locs_1 = [list(self.locs[rng.randint(len(self.locs))]) for _ in range(self.navi.batch_size)]
        print epi_start_locs_1
        # Get Distance between starting locs and target locs
        dist = []
        for epi_start_loc, target in zip(epi_start_locs_1, epi_targets):
            dist.append(len(self.gt_loc_aseq[tuple(epi_start_loc)][target]))

        # Set reach to zero, reachgoal acts like a mask
        self.reachgoal = [0. if i <= 1 else 1. for i in dist]

        self.epi = Foo(
            start_locs=epi_start_locs_1, target_locs=epi_target_locs,
            targets=epi_targets, dist_to_tar=dist
        )
        # Initialize Environment
        self.resetatPos(epi_start_locs_1)

        return epi_start_locs_1

    def get_batch_dist(self, positions=None):
        """
        since at the target location, the optimal action index is !-1!
        when len(self.gt_loc_aseq
        :param positions: postion in Batch
        :return: B X 1 distance to goal location
        """
        if positions:
            poses = positions
        else :
            poses = self.positions
        dist = []
        for pose, target in zip(poses, self.epi.targets):
            dist.append(len(self.gt_loc_aseq[tuple(pose)][target]))
        return dist

    # TODO: implement random move in batch
    # def random_move(self, max_steps=500):
    #     steps = np.random.randint(max_steps)
    #     for i in range(steps):
    #         a = np.random.randint(len(self.actions))
    #         self.step(a)
    #     return self.positions

    def get_batch_g_ids(self):
        batch_g_ids = [self.map[(x, y)] * 4 + orien for (x, y, orien) in self.positions]
        return batch_g_ids

    def get_state_locsmap(self, x, y, orien, sc):
        xysmap = self.smap[(x, y)]
        if sc == 11:
            xysmap = xysmap[10:21, 10:21, :]
        elif sc == 21:
            xysmap = xysmap[5:26, 5:26, :]
            xysmap = xysmap[::2, ::2, :]
        elif sc == 31:
            # TODO : Reconsider Down sample Strategy
            xysmap = xysmap[::3, ::3, :]
        else:
            sys.exit("the scale factor (sc) should be one of [11, 21, 31].")

        for _ in range(orien):
            xysmap = np.rot90(xysmap)
        return xysmap

    def get_batch_locsmaps(self, sc):
        batch_locsmaps = np.array([self.get_state_locsmap(x, y, orien, sc)
                                   for (x, y, orien) in self.positions])
        batch_locsmaps = np.expand_dims(batch_locsmaps, axis=1)
        return batch_locsmaps

    def get_state_image(self, mode, img_id):
        # img_id = self.get_batch_g_ids()
        image_path = '%s/%s/%08d.png' % (self.dir, mode, img_id)
        image = cv2.imread(image_path)
        return image[:, :, ::-1]

    def get_batch_img(self, mode):
        img_ids = self.get_batch_g_ids()
        return [self.get_state_image(mode, img_id) for img_id in img_ids]

    # TODO: return episode targets as batch if needed
    # def get_state_class_info(self):
    #     info_id = self.get_batch_g_ids()
    #     info_path = '%s/class_info/%08d.json' % (self.dir, info_id)
    #     info = json.load(open(info_path, 'r'))
    #     return info

    def success(self):
        return self.lastActionSuccess

    def step(self, action_indices):
        self.positions = self._get_current_position(action_indices)
        return self.positions

    def _get_current_position(self, action_indices):
        positions = []

        for i, action_idx in enumerate(action_indices):
            # The gt action index for reach goal is -1
            if (not self.reachgoal[i]) or action_idx == -1:
                positions.append(self.positions[i])
                assert self.positions[i][0]== self.epi.target_locs[i][0] and self.positions[i][1] == self.epi.target_locs[i][1], \
                    'the position is not target location however get -1 action index!!! ' + \
                    'If not using teach force, disable the line!'
            else:
                action_name = self.actions[action_idx]
                (x, y, orien) = self.positions[i]
                if action_name == 'MoveLeft':
                    if orien == 0:
                        x += 1
                    elif orien == 1:
                        y += 1
                    elif orien == 2:
                        x -= 1
                    elif orien == 3:
                        y -= 1
                elif action_name == 'MoveRight':
                    if orien == 0:
                        x -= 1
                    elif orien == 1:
                        y -= 1
                    elif orien == 2:
                        x += 1
                    elif orien == 3:
                        y += 1
                elif action_name == 'MoveAhead':
                    if orien == 0:
                        y += 1
                    elif orien == 1:
                        x -= 1
                    elif orien == 2:
                        y -= 1
                    elif orien == 3:
                        x += 1
                elif action_name == 'MoveBack':
                    if orien == 0:
                        y -= 1
                    elif orien == 1:
                        x += 1
                    elif orien == 2:
                        y += 1
                    elif orien == 3:
                        x -= 1
                elif action_name == 'RotateRight':
                    orien = (orien + 1) % 4
                elif action_name == 'RotateLeft':
                    orien = (orien + 3) % 4

                self.lastActionSuccess[i] = True
                if action_name in ['MoveLeft', 'MoveRight', 'MoveAhead', 'MoveBack']:
                    if (x, y) not in self.map:
                        self.lastActionSuccess[i] = False
                        (x, y, orien) = self.positions[i]

                target_loc = self.epi.target_locs[i]
                if x == target_loc[0] and y ==target_loc[1]:
                    self.reachgoal[i] = 0.

                positions.append((x, y, orien))

        return positions

    def get_common_data(self):
        # TODO: 0/1 free space map as batch
        maps = self.fmap
        # TODO: goal_locations in the environment as batch
        goal_locs = np.array([
            [target_loc[0], target_loc[1]]
            for target_loc in self.epi.target_locs
        ])
        goal_locs = np.expand_dims(goal_locs, axis=1)
        return vars(Foo(orig_maps=maps, goal_loc=goal_locs))

    def get_step_data(self):
        # Return Dict
        outs = {}

        # localsmap at different scales
        for i in range(len(self.navi.map_crop_sizes)):
            outs['locsmap_{:d}'.format(i)] = self.get_batch_locsmaps(self.navi.map_orig_sizes[i])

        # Get gt_dist_to_goal at current locations
        batch_dist = np.array(self.get_batch_dist())
        outs['gt_dist_to_goal'] = np.expand_dims(batch_dist, axis=1)[:, np.newaxis, :]
        # from list of tuple to array and expand axis
        loc_on_map = np.array([list(pos) for pos in self.positions])
        # Current locations as batch
        outs['loc_on_map'] = np.expand_dims(loc_on_map, axis=1)
        # Semantic onehot vector map
        outs['onehot_semantic'] = self.get_batch_onehot_semantic()
        # Output if one worker in the batch has reached the goal
        outs['if_reach_goal'] = np.expand_dims(np.expand_dims(np.array(self.reachgoal), axis=1), axis=1)
        return outs

    # gt means ground-truth which also means optimal
    def get_batch_gt_actions(self):
        batch_tar_aseq = [self.gt_loc_aseq[tuple(pos)] for pos in self.positions]
        batch_gt_actions = []
        for i, target in enumerate(self.epi.targets):
            batch_gt_actions.append(batch_tar_aseq[i][target][0])
        batch_onehot_gt_actions = np.zeros((self.navi.batch_size, len(self.actions)), dtype=np.float32)
        batch_onehot_gt_actions[np.arange(self.navi.batch_size), batch_gt_actions] = 1.
        batch_onehot_gt_actions = batch_onehot_gt_actions[:, np.newaxis, :]
        return batch_onehot_gt_actions

    def get_batch_onehot_semantic(self):
        """

        :return: B x H x W x (semantic onehot vector of map channel size)
        """
        batch_onehot_semantic = np.zeros((self.navi.batch_size, self.navi.map_crop_sizes[0],
                                          self.navi.map_crop_sizes[0], self.navi.map_channels), dtype=np.float32)
        for i, target in enumerate(self.epi.targets):
            batch_onehot_semantic[i,:,:, self.targets.index(target)] = 1.
        batch_onehot_semantic = np.expand_dims(batch_onehot_semantic, axis=1)
        return batch_onehot_semantic

    def get_step_data_names(self):
        names = []
        names += ['locsmap_{:d}'.format(i) for i in range(3)]
        names += ['loc_on_map', 'gt_dist_to_goal', 'onehot_semantic', 'if_reach_goal']
        return names