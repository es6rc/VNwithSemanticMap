import numpy as np
import cv2
import json


def gen_2dmap(self, x=None, y=None, resolution=None):
    """
    Args:
        x, y: the agent's location. Will use the current camera position by default.
        resolution: (w, h) integer, same as the rendering by default.
    Returns:
        An RGB image of 2d localization, robot locates at (x, y)
    """
    if x is None:
        x, y = self.cam.pos.x, self.cam.pos.z
    if resolution is None:
        resolution = self.resolution
    house = self.house
    n_row = house.n_row

    # TODO: move cachedLocMap to House
    if self.cachedLocMap is None:
        locMap = np.zeros((n_row + 1, n_row + 1, 3), dtype=np.uint8)
        for i in range(n_row):  # w
            for j in range(n_row):  # h
                if house.obsMap[i, j] == 0:
                    locMap[j, i, :] = 255
                if house.canMove(i, j):
                    locMap[j, i, :2] = 200  # purple
        self.cachedLocMap = locMap.copy()
    else:
        locMap = self.cachedLocMap.copy()

    rad = house.robotRad / house.L_det * house.n_row
    x, y = house.to_grid(x, y)
    cv2.circle(locMap, (x, y), int(rad), (0, 0, 255), thickness=-1)
    locMap = cv2.resize(locMap, resolution)
    return locMap


def gen_2dfmap(self):
    house = self.house
    n_row = house.n_row

    locMap = np.zeros((n_row + 1, n_row + 1), dtype=np.uint8)
    for i in range(n_row):  # w
        for j in range(n_row):  # h
            if house.canMove(i, j):
                locMap[i, j] = 1  # free
    return locMap


def gen_2dsmap(self, x=None, y=None, resolution=None):
    with open(self.config) as jfile:
        cfg = json.load(jfile)
    org_smap, org_2dsmap = self.house.genSMap(cfg["modelCategoryFile"], cfg["colorFile"])


def collide_wall(self, pA, pB, num_samples=10):
    ratio = 1.0 / num_samples
    for i in range(num_samples):
        xi = (pB[0] - pA[0]) * (i + 1) * ratio + pA[0]
        yi = (pB[1] - pA[1]) * (i + 1) * ratio + pA[1]
        try:
            collide = self.house.check_occupied_by_obj(xi, yi, 'wall')
            if collide:
                return True
        except IndexError:
            return True
    return False