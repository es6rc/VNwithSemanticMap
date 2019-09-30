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


class gensmap():
    def __init__(self, env, sz=None, rnd_seed=1008):
        """
         :vars: {
            ...
            self.house: {'L_lo':L_lo, 'L_hi': L_hi, 'L_det': L_det,
                        'robotRad': robotRad, 'n_row': n_row, 'orgn_coor': origin_coor,
                        'smap': smap, 'fmap': fmap
                        }
                }
        """
        self.env = env
        cfg = json.load(open(CONFIG_PATH, 'r'))
        code_dir = cfg['codeDir']
        self.dir = '%s/Environment/houses/%s' % (code_dir, self.env)
        # print self.dir
        self.house = pickle.load(open('%s/planner_data/housemap.pkl' % self.dir, 'rb'))

        self.L_lo, self.L_hi, self.L_det = \
            self.house['L_lo'], self.house['L_hi'], self.house['L_det']
        self.robotRad = self.house['robotRad']
        self.translation_scale = 0.2
        # self.orgn_coor = self.house['orgn_coor']
        self.n_row = self.house['n_row']
        self.fmap = self.house['fmap']

        self._load_class2id()
        self.cat2idx_coarseid = self._load_cat()
        self.smap = self._load_smap()

        self._load_map()

        self.sz = sz
        self.rnd = np.random.RandomState(rnd_seed)

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

    def _load_class2id(self):
        self.coarse_class2id = {}
        self.coarse_colormap = {}
        with open(MetaDataColorMap) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader, None)
            myid = 0
            for row in reader:
                color = (int(row[1]), int(row[2]), int(row[3]))
                class_label = row[0].lower()
                if class_label not in self.coarse_class2id:
                    self.coarse_class2id[class_label] = myid
                    self.coarse_colormap[myid] = color
                    myid += 1
            # self.coarse_class2id['others'] = myid
        csvfile.close()

    def _load_cat(self):
        path = "%s/targets_info_all.json" % self.dir
        cat = json.load(open(path, 'r'))
        self.cat = [i.lower() for i in cat.keys()]
        cat2idx_coarseid = {k: (idx, self.coarse_class2id[k]) \
                            for idx, k in enumerate(self.cat)}
        self.bgcat = list(set(self.coarse_class2id.keys()) - set(self.cat))
        self.bgcat2idx_coarseid = {k: (idx, self.coarse_class2id[k]) \
                                   for idx, k in enumerate(self.bgcat)}
        return cat2idx_coarseid

    def _load_smap(self):
        smap_all = self.house['smap']
        sub_smap = np.zeros((self.n_row + 1, self.n_row + 1,
                             len(self.cat) + 1), dtype=np.int32)

        # Copy all the existing category to the smap at the corresponding idx channel
        for k, (idx, coarse_classid) in self.cat2idx_coarseid.items():
            sub_smap[..., idx] = smap_all[..., coarse_classid]

        # Plus all the non-existing category to the smap at the last channel
        # if >= 1, assign 1
        for k, (idx, coarse_classid) in self.bgcat2idx_coarseid.items():
            sub_smap[..., len(self.cat)] += smap_all[..., coarse_classid]
        sub_smap[..., len(self.cat)][sub_smap[..., len(self.cat)] >= 1] = 1

        return sub_smap

    def rescale(self, x1, y1, x2, y2, n_row=None):
        if n_row is None:
            n_row = self.n_row
        tiny = 1e-9
        tx1 = np.floor((x1 - self.L_lo) / self.L_det * n_row + tiny)
        ty1 = np.floor((y1 - self.L_lo) / self.L_det * n_row + tiny)
        tx2 = np.floor((x2 - self.L_lo) / self.L_det * n_row + tiny)
        ty2 = np.floor((y2 - self.L_lo) / self.L_det * n_row + tiny)
        return int(tx1), int(ty1), int(tx2), int(ty2)

    def to_coor(self, x, y):
        """
        Convert grid location to SUNCG dataset continuous coordinate
        (the grid center will be returned when shft is True)
        """
        grid_det = self.L_det / self.n_row
        tx, ty = x * grid_det + self.L_lo, y * grid_det + self.L_lo

        return tx, ty

    def to_grid(self, x, y, n_row=None):
        """
        Convert the true-scale coordinate in SUNCG dataset to grid location
        """
        if n_row is None:
            n_row = self.n_row
        tiny = 1e-9
        tx = np.floor((x - self.L_lo) / self.L_det * n_row + tiny)
        ty = np.floor((y - self.L_lo) / self.L_det * n_row + tiny)
        return int(tx), int(ty)

    def inside(self, x, y):
        return 0 <= x <= self.n_row and 0 <= y <= self.n_row

    def check_occupied_by_wall(self, cx, cy):
        # type: (float, float) -> bool
        # Return True if occupied by wall
        radius = self.robotRad
        x1, y1, x2, y2 = self.rescale(cx - radius, cy - radius, cx + radius, cy + radius)
        for xx in range(x1, x2 + 1):
            for yy in range(y1, y2 + 1):
                if self.fmap[xx, yy] == 1:
                    return False
                else:
                    return self.house['smap'][xx, yy, 74] == 1
        else:
            assert False, 'cx, cy should all be float number indicating real-world coordinates!'

    def collide_wall(self, pA, pB, num_samples=100):
        # type: ((float, float), (float, float), int) -> bool
        """
            Since the room has door object, the wall may not be blocking agent all the time,
            check free space map first for accessiblity and then check smap at wall channel.

        :param pA: starting points in real-world coordinates
        :param pB: destination points in real-world coordinates
        :param num_samples: interpolation points in between pA and pB
        :return: True if collide with wall object
        """
        ratio = 1.0 / num_samples
        for i in range(num_samples):
            xi = (pB[0] - pA[0]) * (i + 1) * ratio + pA[0]
            yi = (pB[1] - pA[1]) * (i + 1) * ratio + pA[1]
            try:
                occupied = self.check_occupied_by_wall(xi, yi)
                if occupied:
                    return True
            except IndexError:
                return True
        return False

    def check_smap_occupancy(self, cx, cy, threshold=1, if_random=True):
        radius = self.robotRad
        x1, y1, x2, y2 = self.rescale(cx - radius, cy - radius, cx + radius, cy + radius)
        if if_random:
            # Randomly sample a point in the area to represent the occupancy

            # When the check point is outside the map
            if x1 > 1000 or y1 > 1000 or x2 < 0 or y2 < 0:
                localsmap = np.zeros_like(self.smap[0, 0, :], dtype=np.float32)
            # When the check point is entirely within the map
            elif x2 <= 1000 and y2 <= 1000 and x1 >= 0 and y1 >= 0:
                xr = self.rnd.randint(x1, x2)
                yr = self.rnd.randint(y1, y2)
                localsmap = self.smap[xr, yr, :]
            else:
                x1 = max(x1, 0);  y1 = max(y1, 0)
                x2 = min(x2, self.n_row); y2 = min(y2, self.n_row)
                xr = self.rnd.randint(x1, x2)
                yr = self.rnd.randint(y1, y2)
                localsmap = self.smap[xr, yr, :]

        else:
            localsmap = self.smap[x1:x2 + 1, y1:y2 + 1, :]
            localsmap = localsmap.reshape((-1, localsmap.shape[-1]))
            localsmap = np.sum(localsmap, axis=0)
            localsmap[localsmap >= threshold] = 1
        return localsmap

    def GetSmapatLoc(self, cx, cy, sz, mode=2):
        # mode = 1 is visualize mode
        # mode = 2 is category mode
        halfsz = sz // 2  # also shift size

        x = np.array(range(-halfsz, halfsz + 1) * sz)  # -1, for the sake of orientation
        y = np.array([[i] * sz for i in range(-halfsz, halfsz + 1)]).flatten()  # -1, for the sake of orientation

        if mode == 2:
            locmap = np.zeros((sz, sz, self.smap.shape[-1]), dtype=np.float32)
        elif mode == 1:
            locmap = np.zeros((sz, sz, 3), dtype=np.uint8)
        else:
            sys.exit('mode should be either 1 or 2!!')

        for i in range(len(x)):
            xi = cx + self.translation_scale * x[i]
            yi = cy + self.translation_scale * y[i]
            if self.collide_wall([cx, cy], [xi, yi]):
                if mode == 1:
                    locmap[y[i] + halfsz, x[i] + halfsz, :] = 0.
                elif mode == 2:
                    locmap[x[i] + halfsz, y[i] + halfsz, :] = 0.

            else:
                if mode == 2:
                    locmap[x[i] + halfsz, y[i] + halfsz, :] = \
                        self.check_smap_occupancy(xi, yi).astype(np.float32)
                # elif mode == 1:
                #     x_pixel, y_pixel = self.to_grid(xi, yi)
                #     # smap_img is not available here.
                #     color = self.env.house.smap_img[y_pixel, x_pixel, :]
                #     locmap[y[i] + halfsz, x[i] + halfsz, :] = color[::-1]
        return locmap

    def GenSmap(self, debug=True):
        sz = self.sz
        if not sz:
            return None
        self.map = {}

        map_path = '%s/map.txt' % self.dir
        localsmaps = {}
        with open(map_path, 'r') as f:
            for line in f:
                nums = line.split()
                if len(nums) == 2:
                    self.abs_start_pos = (float(nums[0]), float(nums[1]))
                else:
                    idx = int(nums[0])
                    pos = (int(nums[1]), int(nums[2]))

                    cx = self.abs_start_pos[0] + self.translation_scale * pos[0]
                    cy = self.abs_start_pos[1] + self.translation_scale * pos[1]

                    localsmap = self.GetSmapatLoc(cx, cy, sz)
                    localsmaps[pos] = localsmap
                    if debug:
                        for j in range(localsmap.shape[-1]):
                            viewbycat = np.zeros((localsmap.shape[1], localsmap.shape[0]),
                                                 dtype=np.uint8)
                            viewbycat[localsmap[..., j] == 1] = 255
                            if np.sum(viewbycat) != 0:
                                for k, (catidx, coarseid) in self.cat2idx_coarseid.items():
                                    if catidx == j:
                                        print (viewbycat)
                                        print('class is %s' % k, 'class id is %d' % catidx)
                                        print('idx is %d' % (idx))
                                        cv2.imshow("debug", np.transpose(viewbycat))
                                key = cv2.waitKey(0)
                                if key == ord('q'):
                                    cv2.destroyAllWindows()

                    ### Since the size of generated semantic map is too big
                    ### We only save the semantic maps at orientation 0
                    # for orien in range(4):
                    #     # g_id = idx * 4 + orien
                    #     localsmaps[(int(nums[1]), int(nums[2]), orien)] = localsmap
                    #     if debug:
                    #         for j in range(localsmap.shape[-1]):
                    #             viewbycat = np.zeros((localsmap.shape[1], localsmap.shape[0]),
                    #                                  dtype=np.uint8)
                    #             viewbycat[localsmap[..., j] == 1] = 255
                    #             if np.sum(viewbycat) != 0:
                    #                 for k, (catidx, coarseid) in self.cat2idx_coarseid.items():
                    #                     if catidx == j:
                    #                         print (viewbycat)
                    #                         print('class is %s' % k, 'class id is %d' % catidx)
                    #                         print('idx is %d, orientation is %d' % (idx, orien))
                    #                         cv2.imshow("debug", np.transpose(viewbycat))
                    #                 key = cv2.waitKey(0)
                    #                 if key == ord('q'):
                    #                     cv2.destroyAllWindows()
                    #    localsmap = np.rot90(localsmap)
        return localsmaps

    # def _check_collision_fast(self, pA, pB, num_samples=5):
    #     ratio = 1.0 / num_samples
    #     for i in range(num_samples):
    #         p = (pB - pA) * (i + 1) * ratio + pA
    #         gx, gy = self.to_grid(p[0], p[2])
    #         if (not self.fmap[gx, gy]):
    #             return False
    #     return True
    #
    # def check_grid_occupy(self, cx,cy,gx,gy):
    #     # Return True if occupied
    #     grid_det = self.L_det / self.n_row
    #     for x in range(gx, gx + 2):
    #         for y in range(gy, gy + 2):
    #             rx, ry = x * grid_det + self.L_lo, y * grid_det + self.L_lo
    #             if (rx - cx) ** 2 + (ry - cy) ** 2 <= \
    #                     self.robotRad * self.robotRad:
    #                 return True
    #     return False
    #
    # def check_occupy(self, cx, cy):  # cx, cy are real coordinates
    #     radius = self.robotRad
    #     x1,y1,x2,y2=self.rescale(cx-radius,cy-radius,cx+radius,cy+radius)
    #     for xx in range(x1,x2+1):
    #         for yy in range(y1,y2+1):
    #             if (not self.inside(xx,yy) or self.fmap[xx,yy] == 1) \
    #                 and self.check_grid_occupy(cx,cy,xx,yy):
    #                 return False
    #     return True
    #
    # def _check_collision(self, pA, pB, num_samples=5):
    #     # if USE_FAST_COLLISION_CHECK:
    #     #     return self._check_collision_fast(pA, pB, FAST_COLLISION_CHECK_SAMPLES)
    #     # pA is always valid
    #     ratio = 1.0 / num_samples
    #     for i in range(num_samples):
    #         p = (pB - pA) * (i + 1) * ratio + pA
    #         if not self.house.check_occupy(p[0], p[2]):
    #             return False
    #     return True


def GenSmapinHouse(house_id, lmapszs, output_fmap=True):
    """
    Example:
    house_id = '5cf0e1e9493994e483e985c436b9d3bc'
    lmapszs = [11, 21, 31]
    GenSmapinHouse(house_id, lmapszs)

        :param lmapszs: local semantic map sizes, iterable input
        :type house_id: house id in string

        """
    for sz in lmapszs:
        assert  sz % 2, "Currently don't support even number size: %d" % sz

    for lmapsz in lmapszs:
        house = gensmap(env=house_id, sz=lmapsz)
        localmaps = house.GenSmap(debug=False)
        with open("%s/localsmaps_%d.pkl" % (house.dir, lmapsz), 'wb') as mfile:
            pickle.dump(localmaps, mfile)

#
if __name__ == '__main__':
    house_id = '5cf0e1e9493994e483e985c436b9d3bc'
    lmapszs = [31]
    GenSmapinHouse(house_id, lmapszs)

