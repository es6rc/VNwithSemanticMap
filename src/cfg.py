from utils import Foo

import numpy as np
import logging
import tensorflow as tf


def setup_args_sum_ctrl():
    summary_args = Foo(display_interval=1, test_iters=26,
                       arop_full_summary_iters=14)

    control_args = Foo(train=False, test=False,
                       force_batchnorm_is_training_at_test=False,
                       reset_rng_seed=False, only_eval_when_done=False,
                       test_mode=None)
    return summary_args, control_args

def setup_smyctl(args, mode_str):
    mode, imset = mode_str.split('_')
    args.control.test_name = '{:s}_on_{:s}'.format(mode, imset)

    args.navi.batch_size = 4
    args.control.test = True
    args.solver.sample_action_type = 'sample'
    args.solver.sample_gt_prob_type = 'zero'

    args.summary.test_iters = 100
    args.control.only_eval_when_done = True
    args.control.reset_rng_seed = True
    args.control.test_mode = 'test'
    logging.error('args: %s', args)
    return args

def get_solver_vars(solver_str):
    if solver_str == '':
        vals = []
    else:
        vals = solver_str.split('_')
    ks = ['clip', 'dlw', 'long', 'typ', 'isdk', 'adam_eps', 'init_lr']
    ks = ks[:len(vals)]

    # Gradient clipping or not.
    if len(vals) == 0: ks.append('clip'); vals.append('noclip');
    # data loss weight.
    if len(vals) == 1: ks.append('dlw');  vals.append('dlw20')
    # how long to train for.
    if len(vals) == 2: ks.append('long');  vals.append('nolong')
    # Adam
    if len(vals) == 3: ks.append('typ');  vals.append('adam2')
    # reg loss wt
    if len(vals) == 4: ks.append('rlw');  vals.append('rlw1')
    # isd_k
    if len(vals) == 5: ks.append('isdk');  vals.append('isdk415')  # 415, inflexion at 2.5k.
    # adam eps
    if len(vals) == 6: ks.append('adam_eps');  vals.append('aeps1en8')
    # init lr
    if len(vals) == 7: ks.append('init_lr');  vals.append('lr1en3')

    assert (len(vals) == 8)

    vars = Foo()
    for k, v in zip(ks, vals):
        setattr(vars, k, v)
    # logging.info('solver_vars: %s', vars)
    return vars


def setup_args_solver(solver_str):
    solver = Foo(
        seed=0, learning_rate_decay=None, clip_gradient_norm=None, max_steps=None,
        initial_learning_rate=None, momentum=None, steps_per_decay=None,
        logdir=None, sync=False, adjust_lr_sync=True, wt_decay=0.0001,
        data_loss_wt=None, reg_loss_wt=None, freeze_conv=True, num_workers=1,
        rng_seed=0, ps_tasks=0, master='', typ=None, momentum2=None,
        adam_eps=None, sample_action_type='argmax', sample_action_combine_type='one_or_other',
        sample_gt_prob_type='inverse_sigmoid_decay', dagger_sample_bn_false=True)

    # Clobber with overrides from solver str.
    solver_vars = get_solver_vars(solver_str)

    solver.data_loss_wt = float(solver_vars.dlw[3:].replace('x', '.'))
    solver.adam_eps = float(solver_vars.adam_eps[4:].replace('x', '.').replace('n', '-'))
    solver.initial_learning_rate = float(solver_vars.init_lr[2:].replace('x', '.').replace('n', '-'))
    solver.reg_loss_wt = float(solver_vars.rlw[3:].replace('x', '.'))
    solver.isd_k = float(solver_vars.isdk[4:].replace('x', '.'))

    long = solver_vars.long
    if long == 'long':
        solver.steps_per_decay = 40000
        solver.max_steps = 120000
    elif long == 'long2':
        solver.steps_per_decay = 80000
        solver.max_steps = 120000
    elif long == 'nolong' or long == 'nol':
        solver.steps_per_decay = 4000
        solver.max_steps = 10000
    else:
        logging.fatal('solver_vars.long should be long, long2, nolong or nol.')
        assert False

    clip = solver_vars.clip
    if clip == 'noclip' or clip == 'nocl':
        solver.clip_gradient_norm = 0
    elif clip[:4] == 'clip':
        solver.clip_gradient_norm = float(clip[4:].replace('x', '.'))
    else:
        logging.fatal('Unknown solver_vars.clip: %s', clip)
        assert (False)

    typ = solver_vars.typ
    if typ == 'adam':
        solver.typ = 'adam'
        solver.momentum = 0.9
        solver.momentum2 = 0.999
        solver.learning_rate_decay = 1.0
    elif typ == 'adam2':
        solver.typ = 'adam'
        solver.momentum = 0.9
        solver.momentum2 = 0.999
        solver.learning_rate_decay = 0.1
    elif typ == 'sgd':
        solver.typ = 'sgd'
        solver.momentum = 0.99
        solver.momentum2 = None
        solver.learning_rate_decay = 0.1
    else:
        logging.fatal('Unknown solver_vars.typ: %s', typ)
        assert False

    return solver


def get_default_args_navi():
    outputs = Foo(
        images=True,
        rel_goal_loc=False,
        loc_on_map=True,
        gt_dist_to_goal=True,
        ego_maps=False,
        # ego_goal_imgs=False,
        egomotion=False)
        # visit_count=False,
        # analytical_counts=False,
        # node_ids=True,
        # readout_maps=False)

    # camera_param = Foo(
    #     width=225,
    #     height=225,
    #     z_near=0.05,
    #     z_far=20.0,
    #      fov=60.,
    #      modalities=['rgb'],
    #      img_channels=3)

    # data_augment = Foo(
    #     lr_flip=0,
    #     delta_angle=0.5,
    #     delta_xy=4,
    #     relight=True,
    #     relight_fast=False,
    #     structured=False)

    # class_map_names = ['chair', 'door', 'table']
    # semantic_task = Foo(class_map_names=class_map_names, pix_distance=16,
    #                     sampling='uniform')
    navi = Foo(
        max_dist=32,
        step_size=8,
        num_goals=1,
        num_steps=8,
        num_actions=6,
        n_ori=4,
        batch_size=4,
        # building_seed=0,
        # img_height=None,
        # img_width=None,
        # img_channels=None,
        # modalities=None,
        outputs=outputs,
        # camera_param=camera_param,
        map_scales=[1.],
        map_crop_sizes=[11],
        # rel_goal_loc_dim=4,
        # base_class='Building',
        # task='map+plan',
        # type='room_to_room_many',
        # data_augment=data_augment,
        # room_regex='^((?!hallway).)*$',
        # toy_problem=False,
        map_channels=26,
        # gt_coverage=False,
        # input_type='maps',
        # full_information=False,
        # aux_delta_thetas=[],
        # semantic_task=semantic_task,
        # num_history_frames=0,
        # node_ids_dim=1,
        # perturbs_dim=4,
        map_resize_method='linear_noantialiasing',
        # readout_maps_channels=1,
        # readout_maps_scales=[],
        # readout_maps_crop_sizes=[],
        # n_views=1,
        reward_time_penalty=0.1,
        reward_at_goal=1.,
        discount_factor=0.99,
        rejection_sampling_M=100,
        min_dist=None)

    return navi

def get_navtask_vars(navtask_str):
    if navtask_str == '':
        vals = []
    else:
        vals = navtask_str.split('_')

    ks_all = ['dataset_name', 'modality', 'task', 'history', 'max_dist',
              'num_steps', 'step_size', 'n_ori', 'aux_views', 'data_aug']
    ks = ks_all[:len(vals)]

    # All data or not.
    if len(vals) == 0: ks.append('dataset_name'); vals.append('sbpd')
    # modality
    if len(vals) == 1: ks.append('modality'); vals.append('rgb')
    # semantic task?
    if len(vals) == 2: ks.append('task'); vals.append('r2r')
    # number of history frames.
    if len(vals) == 3: ks.append('history'); vals.append('h0')
    # max steps
    if len(vals) == 4: ks.append('max_dist'); vals.append('32')
    # num steps
    if len(vals) == 5: ks.append('num_steps'); vals.append('45')
    # step size
    if len(vals) == 6: ks.append('step_size'); vals.append('8')
    # n_ori
    if len(vals) == 7: ks.append('n_ori'); vals.append('4')
    # Auxiliary views.
    if len(vals) == 8: ks.append('aux_views'); vals.append('nv0')
    # Normal data augmentation as opposed to structured data augmentation (if set
    # to straug.
    if len(vals) == 9: ks.append('data_aug'); vals.append('straug')

    assert (len(vals) == 10)
    for i in range(len(ks)):
        assert (ks[i] == ks_all[i])

    vars = Foo()
    for k, v in zip(ks, vals):
        setattr(vars, k, v)
    logging.error('navtask_vars: %s', vals)
    return vars


def setup_args_navi(navi_str):
    navi = get_default_args_navi()

    # Clobber with overrides from strings.
    navtask_vars = get_navtask_vars(navi_str)

    navi.n_ori = int(navtask_vars.n_ori)
    navi.max_dist = int(navtask_vars.max_dist)
    navi.num_steps = int(navtask_vars.num_steps)
    navi.step_size = int(navtask_vars.step_size)
    # navi.data_augment.delta_xy = int(navtask_vars.step_size) / 2.
    # n_aux_views_each = int(navtask_vars.aux_views[2])
    # aux_delta_thetas = np.concatenate((np.arange(n_aux_views_each) + 1,
    #                                    -1 - np.arange(n_aux_views_each)))
    # aux_delta_thetas = aux_delta_thetas * np.deg2rad(navi.camera_param.fov)
    # navi.aux_delta_thetas = aux_delta_thetas

    # if navtask_vars.data_aug == 'aug':
    #     navi.data_augment.structured = False
    # elif navtask_vars.data_aug == 'straug':
    #     navi.data_augment.structured = True
    # else:
    #     logging.fatal('Unknown navtask_vars.data_aug %s.', navtask_vars.data_aug)
    #     assert (False)

    navi.num_history_frames = int(navtask_vars.history[1:])
    # navi.n_views = 1 + navi.num_history_frames

    # navi.goal_channels = int(navtask_vars.n_ori)

    # if navtask_vars.task == 'ST':
    #     # Semantic task at hand.
    #     navi.smap_channels = \
    #         len(navi.semantic_task.class_map_names)
    #     navi.rel_goal_loc_dim = \
    #         len(navi.semantic_task.class_map_names)
    #     navi.type = 'to_nearest_obj_acc'
    # else:
    #     logging.fatal('navtask_vars.task: should be ST')
    #     assert (False)

    # if navtask_vars.modality == 'rgb':
    #     navi.camera_param.modalities = ['rgb']
    #     navi.camera_param.img_channels = 3
    # elif navtask_vars.modality == 'd':
    #     navi.camera_param.modalities = ['depth']
    #     navi.camera_param.img_channels = 2
    #
    # navi.img_height = navi.camera_param.height
    # navi.img_width = navi.camera_param.width
    # navi.modalities = navi.camera_param.modalities
    # navi.img_channels = navi.camera_param.img_channels
    # navi.img_fov = navi.camera_param.fov
    # TODO Add dataset here
    navi.dataset = None
    return navi


def get_default_arch_args():
    batch_norm_param = {'center': True, 'scale': True,
                        'activation_fn': tf.nn.relu}

    # mapper_arch_args = Foo(
    #     dim_reduce_neurons=64,
    #     fc_neurons=[1024, 1024],
    #     fc_out_size=8,
    #     fc_out_neurons=64,
    #     encoder='resnet_v2_50',
    #     deconv_neurons=[64, 32, 16, 8, 4, 2],
    #     deconv_strides=[2, 2, 2, 2, 2, 2],
    #     deconv_layers_per_block=2,
    #     deconv_kernel_size=4,
    #     fc_dropout=0.5,
    #     combine_type='wt_avg_logits',
    #     batch_norm_param=batch_norm_param)

    # readout_maps_arch_args = Foo(
    #     num_neurons=[],
    #     strides=[],
    #     kernel_size=None,
    #     layers_per_block=None)

    arch_args = Foo(
        vin_val_neurons=8, vin_action_neurons=8, vin_ks=3, vin_share_wts=False,
        pred_neurons=[64, 64], pred_batch_norm_param=batch_norm_param,
        conv_on_value_map=0, fr_neurons=16, fr_ver='v2', fr_inside_neurons=64,
        fr_stride=1, margin_crop_size=2, value_crop_size=5,
        vin_num_iters=8, isd_k=750., use_agent_loc=False, multi_scale=True,
        conv_ks=3, conv_neurons=16)
    # readout_maps=False, rom_arch=readout_maps_arch_args)

    return arch_args  # , mapper_arch_args


def process_arch(args, arch_str):
    if arch_str == 'Ssc':
        sc = 1. / args.navi.step_size
        args.arch.vin_num_iters = 8
        args.navi.map_scales = [sc]
        # max_dist = args.navi.max_dist * \
        #            args.navi.num_goals
        args.navi.map_crop_sizes = [11]
        args.navi.map_orig_sizes = [31]

        args.arch.fr_stride = 1
        args.arch.vin_action_neurons = 8
        args.arch.vin_val_neurons = 3
        args.arch.fr_inside_neurons = 32

        # args.mapper_arch.pad_map_with_zeros_each = [24]
        # args.mapper_arch.deconv_neurons = [64, 32, 16]
        # args.mapper_arch.deconv_strides = [1, 2, 1]

    elif (arch_str == 'Msc'):
        # Code for multi-scale planner.
        args.arch.vin_num_iters = 8
        args.arch.margin_crop_size = 3
        args.arch.value_crop_size = 5

        sc = 1. / args.navi.step_size
        # max_dist = args.navi.max_dist * args.navi.num_goals
        #
        # n_scales = np.log2(float(max_dist) / float(args.arch.vin_num_iters))
        # n_scales = int(np.ceil(n_scales) + 1)

        # args.navi.map_scales = \
        #     list(sc * (0.5 ** (np.arange(n_scales))[::-1]))
        # args.navi.map_crop_sizes = [11 for x in range(n_scales)]
        args.navi.map_crop_sizes = [11 for _ in range(3)]
        args.navi.map_orig_sizes = [11, 21, 31]
        args.arch.fr_stride = 1
        args.arch.vin_action_neurons = 8
        args.arch.vin_val_neurons = 3
        args.arch.fr_inside_neurons = 32

        # args.mapper_arch.pad_map_with_zeros_each = [0 for _ in range(n_scales)]
        # args.mapper_arch.deconv_neurons = [64 * n_scales, 32 * n_scales, 16 * n_scales]
        # args.mapper_arch.deconv_strides = [1, 2, 1]
    return args


def setup_args_arch(args, arch_str):
    # This function modifies args.
    # TODO construct arch for mapper
    # args.arch, args.mapper_arch = get_default_cmp_args()

    args.arch = get_default_arch_args()
    args = process_arch(args, arch_str)
    args.navi.outputs.ego_maps = False
    args.navi.outputs.ego_goal_imgs = False
    args.navi.outputs.egomotion = True

    return args
