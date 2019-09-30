


### Add to House.__init__
# TODO: integrate this line to this class
'''
        with open("/media/z/Data/Object_Searching/code/Environment/House3D/tests/config.json") as jfile:
            cfg = json.load(jfile)

        if DebugMessages == True:
            print('Generate High Resolution Semantic Map  ...')
            ts = time.time()
        self.genSMap(cfg["modelCategoryFile"], cfg["colorFile"])
'''


def genSMap(self, MetaDataModelFile, MetaDataColorMap, dest=None, n_row=None):
    # type: (str, str, bool, np.array, int) -> np.array or None
    """

    :param MetaDataModelFile: Path to ModelCategoryMapping.csv
    :param MetaDataColorMap: Path to colormap_coarse.csv
    :param dest: save the SMap to certain value
    :param n_row: map size in grid world
    """
    # Generate dictionary of class labels and their ids
    self.coarse_class2id = {}
    self.color_map = {}
    with open(MetaDataColorMap) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader, None)
        myid = 0
        for row in reader:
            color = (int(row[1]), int(row[2]), int(row[3]))
            class_label = row[0].lower()
            if class_label not in self.coarse_class2id:
                self.coarse_class2id[class_label] = myid
                self.color_map[myid] = color
                myid += 1
        # self.coarse_class2id['others'] = myid
    csvfile.close()

    # Get ModelID - My Class Dictionary bounds
    target_match_class = "coarse_grained_class"

    modelid2myid = {}
    with open(MetaDataModelFile) as csvfile:
        reader = csv.DictReader(csvfile)
        for row1 in reader:
            modelid_a, class_a = row1['model_id'], row1[target_match_class]
            if row1[target_match_class].lower() in self.coarse_class2id:
                modelid2myid[modelid_a] = self.coarse_class2id[class_a.lower()]
            else:
                modelid2myid[modelid_a] = self.coarse_class2id['other']
    obsMap = dest if dest else self.obsMap
    if not n_row:
        n_row = self.n_row
    self.smap = np.zeros((self.n_row + 1, self.n_row + 1, len(self.coarse_class2id)), dtype=np.int8)

    print (self.color_map[74] == (4, 247, 87))  # Check if the color_map is correct
    #                                      # The wall colormap index is 74
    self.smap_img = np.zeros((self.n_row + 1, self.n_row + 1, 3), dtype=np.uint8)

    for i in range(n_row):  # w
        for j in range(n_row):  # h
            # Fill movable area with Color of Ground
            # obsMap[i, j] == 1: => obstacle
            if obsMap[i, j]:
                self.smap_img[j, i, :] = (99, 255, 172)  # Colored with ground's colormap
            else:
                self.smap_img[j, i, :] = list(self.color_map[40])  # Colored with floor's colormap

    # Fill all the wall objects
    for wall in self.all_walls:
        _x1, _z1, _y1 = wall['bbox']['min']
        _x2, _z2, _y2 = wall['bbox']['max']
        x1, y1, x2, y2 = self.rescale(_x1, _y1, _x2, _y2, n_row)
        self.smap_img[y1:(y2 + 1), x1:(x2 + 1), :] = list(self.color_map[74])

        # if _z1 < self.robotHei:
        fill_region(self.smap[..., 74], x1, y1, x2, y2, 1)

        # TODO VISIALIZE THE WALL SMAP
        # temp = self.smap[..., 74]

    # Overlay Objects upon others on 2D Top-Down Semantic Map
    for obj in self.all_obj:
        _x1, _, _y1 = obj['bbox']['min']
        _x2, _, _y2 = obj['bbox']['max']
        x1, y1, x2, y2 = self.rescale(_x1, _y1, _x2, _y2, n_row)
        objid = modelid2myid[obj["modelId"]]

        # Fill the 2d top-down Semantic Map
        self.smap_img[y1:(y2 + 1), x1:(x2 + 1), :] = list(self.color_map[objid])

        # Fill the Corresponding Semantic Channel and its area with ones
        fill_region(self.smap[..., objid], x1, y1, x2, y2, 1)
    # self.check_smap_occupy(42, 37, 1)
    return self.smap, self.smap_img


def check_smap_occupy(self, cx, cy, threshold):
    radius = self.robotRad
    x1, y1, x2, y2 = self.rescale(cx - radius, cy - radius, cx + radius, cy + radius)
    localsmap = self.smap[x1:x2 + 1, y1:y2 + 1, :]
    localsmap = localsmap.reshape((-1, localsmap.shape[-1]))
    localsmap = np.sum(localsmap, axis=0)
    localsmap[localsmap >= threshold] = 1
    return localsmap


def check_occupied_by_obj(self, cx, cy, objname):  # cx, cy are given in real positions
    radius = self.robotRad
    x1, y1, x2, y2 = self.rescale(cx - radius, cy - radius, cx + radius, cy + radius)
    for xx in range(x1, x2 + 1):
        for yy in range(y1, y2 + 1):
            if self.smap[xx, yy, self.coarse_class2id[objname]] == 1:
                return True
    return False
