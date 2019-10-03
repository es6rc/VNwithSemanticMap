import os, cv2
import pickle
import shutil

import numpy as np
from matplotlib import pyplot as plt
from src.navi_env import Environment
from src.utils import Foo
from preprocess.gensmap import gensmap


def get_colormap_class2id(houseid='5cf0e1e9493994e483e985c436b9d3bc'):
    houseinfo = gensmap(houseid)
    coarse_colormap = houseinfo.coarse_colormap
    coarse_class2id = houseinfo.coarse_class2id
    return coarse_colormap, coarse_class2id


def get_env_objs(houseid='5cf0e1e9493994e483e985c436b9d3bc', navi=None):
    # navi = Foo(batch_size=4, map_orig_sizes=[11, 21, 31],
    #            map_scales=[11, 21, 31], map_crop_sizes=[11] * 3, map_channels=26)
    env = Environment(houseid, navi)
    return env, env.targets


def position_attribute(pos):
    """
    Attribute to Xin Ye : http://www.public.asu.edu/~xinye1/
    """
    (x, y, orien) = pos
    if orien == 0:
        relative_orien = (0, 0.1)
    elif orien == 1:
        relative_orien = (-0.1, 0)
    elif orien == 2:
        relative_orien = (0, -0.1)
    else:
        relative_orien = (0.1, 0)
    return relative_orien


def get_map_location(map):
    """
    Attribute to Xin Ye : http://www.public.asu.edu/~xinye1/
    """
    x = []
    y = []
    for pos in map:
        x.append(pos[0])
        y.append(pos[1])
    return x, y


def show_trajectory(env, state_buffer, tar_loc, save_dir):
    """
    Attribute to Xin Ye : http://www.public.asu.edu/~xinye1/
    """
    fig, axes = plt.subplots()  # figsize=(5,4))
    plt.grid(True)
    plt.title('Top-down 2D Map')
    axes.invert_yaxis()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)

    axes.xaxis.set_ticks_position('bottom')
    axes.spines['bottom'].set_position(('data', 0))
    axes.yaxis.set_ticks_position('left')
    axes.spines['left'].set_position(('data', 0))

    map_x, map_y = get_map_location(env.map.keys())
    plt.scatter(map_x, map_y, marker='o', color='r', s=20)

    if env.env == 'real':
        target_x, target_y = [1], [1]
    else:
        target_x, target_y = tar_loc[0], tar_loc[1]
    plt.scatter(target_x, target_y, marker='o', color='r', s=50)

    prev_pos = state_buffer[0]
    (dx, dy) = position_attribute(prev_pos)
    pp, = axes.plot(prev_pos[0], prev_pos[1], marker='o', markerfacecolor='b', markersize=6)
    pa = axes.arrow(prev_pos[0], prev_pos[1], dx, dy, head_width=1, head_length=1, fc='b', ec='b')
    fig.savefig('%s/%02d.png' % (save_dir, 0), bbox_inches='tight', dpi=300)

    for i in range(len(state_buffer) - 1):
        pp.remove()
        pa.remove()
        curr_pos = state_buffer[i + 1]
        if curr_pos[0] != tar_loc[0] or curr_pos[1] != tar_loc[1]:
            axes.plot([prev_pos[0], curr_pos[0]], [prev_pos[1], curr_pos[1]], color='b', linestyle='-',
                      linewidth=2)
            (dx, dy) = position_attribute(curr_pos)
            pp, = axes.plot(curr_pos[0], curr_pos[1], marker='o', markerfacecolor='b', markersize=6)
            pa = axes.arrow(curr_pos[0], curr_pos[1], dx, dy, head_width=1, head_length=1, fc='b', ec='b')
            fig.savefig('%s/%02d.png' % (save_dir, i + 1), bbox_inches='tight', dpi=300)
            prev_pos = curr_pos
        else:
            break

def rot(values, orien):
    for _ in range(orien):
        values = np.rot90(values, -1)

    values = np.transpose(values, axes=(1, 0, 2))

    for _ in range(orien):
        values = np.rot90(values)

    return values


def visSmap(colormap, class2id, objs, locsmap, orien):
    """
    :param locsmap: 11 x 11 x Chnl
    :return: 11 x 11 x 3
    """
    locsmap_img = np.zeros([11, 11, 3], dtype=np.uint8)
    for objidx in range(locsmap.shape[-1] - 1):
        objcolor = colormap[class2id[objs[objidx]]]
        locsmap_img[locsmap[..., objidx] == 1] = list(objcolor)

    # plt.imshow(np.transpose(locsmap_img, axes=(1, 0, 2)))
    # plt.show()
    return rot(locsmap_img, orien)


def visFrValSmap(colormap, class2id, objs, fr, val, locsmap, loc, showplt=False):
    print(loc)
    x, y, orien = loc
    r_sc_0, r_sc_1, r_sc_2 = fr[2], fr[1], fr[0]
    r_sc_0 = np.mean(r_sc_0, axis=2, keepdims=True)
    r_sc_1 = np.mean(r_sc_1, axis=2, keepdims=True)
    r_sc_2 = np.mean(r_sc_2, axis=2, keepdims=True)
    r_sc_0_vis = rot(r_sc_0, orien)[..., 0]
    r_sc_1_vis = rot(r_sc_1, orien)[..., 0]
    r_sc_2_vis = rot(r_sc_2, orien)[..., 0]

    v_sc_0, v_sc_1, v_sc_2 = val[2], val[1], val[0]
    v_sc_0 = np.max(v_sc_0, axis=2, keepdims=True)
    v_sc_1 = np.max(v_sc_1, axis=2, keepdims=True)
    v_sc_2 = np.max(v_sc_2, axis=2, keepdims=True)
    v_sc_0_vis = rot(v_sc_0, orien)[..., 0]
    v_sc_1_vis = rot(v_sc_1, orien)[..., 0]
    v_sc_2_vis = rot(v_sc_2, orien)[..., 0]

    locsmap_0, locsmap_1, locsmap_2 = locsmap[0], locsmap[1], locsmap[2]
    vislocsmap_0 = visSmap(colormap, class2id, objs, locsmap_0, orien)
    vislocsmap_1 = visSmap(colormap, class2id, objs, locsmap_1, orien)
    vislocsmap_2 = visSmap(colormap, class2id, objs, locsmap_2, orien)

    fig, axs = plt.subplots(3, 3)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0., hspace=0.1)
    # plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
    axs[0, 0].imshow(r_sc_0_vis); axs[1, 0].imshow(r_sc_1_vis)
    axs[2, 0].imshow(r_sc_2_vis)
    axs[0, 1].imshow(v_sc_0_vis); axs[1, 1].imshow(v_sc_1_vis)
    axs[2, 1].imshow(v_sc_2_vis)
    axs[0, 2].imshow(vislocsmap_0); axs[1, 2].imshow(vislocsmap_1)
    axs[2, 2].imshow(vislocsmap_2)
    # plt.savefig('out.png', dpi=300)
    if showplt:
        plt.show()
        # exit(0)
    return plt


def visTrajFrValSmap(env, traj_record, savedir):

    # Get colormap & class2id & objects
    colormap, class2id = get_colormap_class2id(env.env)
    objs = env.targets

    # Generate Save Directory
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    targets = traj_record['target'][0]
    traget_locs = traj_record['targetloc']
    state_locs = np.array([np.squeeze(loc, axis=1) for loc in traj_record['loc']])
    fr = traj_record['fr']
    val = traj_record['value']
    localsmap = traj_record['localsmap']
    for i in range(len(traget_locs)):
        tar = targets[i]
        print(tar)
        tarloc = traget_locs[i]
        tar_state_loc = state_locs[:, i, :]

        # Output Trajctory on Grid World
        trajdir = '%s/%s/gridtraj' % (savedir, tar)
        if not os.path.isdir(trajdir):
            os.makedirs(trajdir)
        show_trajectory(env, tar_state_loc, tarloc, trajdir)

        # Output Fr & Value & Local Semantic Map
        tar_fr = np.array(fr)[:, :, i, ...]
        tar_val = np.array(val)[:, :, i, ...]
        tar_locsmap = np.array(localsmap)[:, :, i, 0, ...]

        for j in range(tar_state_loc.shape[0]):
            if tar_state_loc[j][0] == tarloc[0] and tar_state_loc[j][1] == tarloc[1]:
                print (j, tar_state_loc[j], tarloc)
                break

            else:
                plt = visFrValSmap(colormap, class2id, objs, tar_fr[j], tar_val[j],
                                   tar_locsmap[j], tar_state_loc[j], showplt=False)
                savepath = '%s/%s/frvalsmap' % (savedir, tar)
                if not os.path.isdir(savepath):
                    os.makedirs(savepath)
                plt.savefig('%s/%02d.png' % (savepath, j), dpi=300)


def getTrajFrValSmap(houseid, navi, savedir, trajpath):
    # sourcedir = '/media/z/Data/Object_Searching/code/NewMethods/M1000'
    # outputdir = sourcedir + '/tf_code/output'
    # testonbench = outputdir + '/planner_multitar/test_on_bench'
    # savedir = sourcedir + '/output'
    # traj = pickle.load(open('%s/fr_value_1234.pkl' % testonbench, 'rb'))
    # navi = Foo(batch_size=4, map_orig_sizes=[11, 21, 31],
    #            map_scales=[11, 21, 31], map_crop_sizes=[11] * 3, map_channels=26)
    # houseid = '5cf0e1e9493994e483e985c436b9d3bc'
    traj = pickle.load(open(trajpath, 'rb'))
    env = Environment(houseid, navi)
    visTrajFrValSmap(env, traj, savedir)


def testvis():
    navi = Foo(batch_size=4, map_orig_sizes=[11, 21, 31],
               map_scales=[11, 21, 31], map_crop_sizes=[11] * 3, map_channels=26)
    sourcedir = '/media/z/Data/Object_Searching/code/NewMethods/M1000'
    outputdir = sourcedir + '/tf_code/output'
    testonbench = outputdir + '/planner_multitar/test_on_bench'
    savedir = sourcedir + '/output/traj_123wo1'
    trajpath = testonbench + '/fr_value_123_wo1.pkl'
    getTrajFrValSmap('5cf0e1e9493994e483e985c436b9d3bc',navi, savedir, trajpath)


def resize_hconcat(img1, img2, dest_size):
    newimg1 = cv2.resize(img1, dest_size)
    newimg2 = cv2.resize(img2, dest_size)
    return cv2.hconcat([newimg1, newimg2])


def concatTraj_FrValSmap(pic_dir, ifdel=True):
    """
    Concatenate frvalsmap + gridtraj and save concatenated images
    pic_dir = sourcedir + '/output/traj'
    :param pic_dir: Source Root directory + picture relative path
    :return: None
    """
    for dir in os.listdir(pic_dir):
        tar_dir = os.path.join(pic_dir, dir)
        frvalsmap_dir = os.path.join(tar_dir, 'frvalsmap')
        gridtraj_dir = os.path.join(tar_dir, 'gridtraj')
        imgname_list = sorted(os.listdir(frvalsmap_dir))
        concat_dir = os.path.join(tar_dir, 'concat')
        if not os.path.exists(concat_dir):
            os.makedirs(concat_dir)

        for imgname in imgname_list:
            im1_path = os.path.join(frvalsmap_dir, imgname)
            im2_path = os.path.join(gridtraj_dir, imgname)
            im1 = cv2.imread(im1_path); im2 = cv2.imread(im2_path)
            concatimg = resize_hconcat(im1, im2, (1800, 1200))
            outpath = os.path.join(concat_dir, imgname)
            cv2.imwrite(outpath, concatimg)

        if ifdel:
            shutil.rmtree(frvalsmap_dir); shutil.rmtree(gridtraj_dir)


def testconcat():
    sourcedir = '/media/z/Data/Object_Searching/code/NewMethods/M1000'
    pic_dir = sourcedir + '/output/traj_123wo1'
    concatTraj_FrValSmap(pic_dir)