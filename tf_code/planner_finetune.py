import tensorflow as tf
from tensorflow.contrib import slim

import numpy as np
import tf_utils
import nns
# import tfcode.nav_utils as nu
import smy_utils as smy
import worker

fr_v2 = nns.fr_v2
value_iteration_network = nns.value_iteration_network
fc_nn = nns.fc_network

setup_training = tf_utils.setup_training
compute_losses_multi_or = tf_utils.compute_losses_multi_or
_add_summaries = smy._add_summaries


def _inputs(problem):
    # Set up inputs.
    with tf.name_scope('inputs'):
        inputs = []
        inputs.append(('orig_maps', tf.float32,
                       (None, None)))
        inputs.append(('goal_loc', tf.float32,
                       (problem.batch_size, problem.num_goals, 2)))  # (batch, 1, (x, y))
        common_input_data, _ = tf_utils.setup_inputs(inputs)

        inputs = []
        for i in range(len(problem.map_crop_sizes)):
            inputs.append(('locsmap_{:d}'.format(i), tf.float32,
                           (problem.batch_size, None, problem.map_crop_sizes[i],
                            problem.map_crop_sizes[i], problem.map_channels)))
        inputs.append(('onehot_semantic', tf.float32,
                       (problem.batch_size, None, problem.map_crop_sizes[i],
                        problem.map_crop_sizes[i], problem.map_channels)))

        inputs.append(('stop_gt_act_step_number', tf.int32, (1, None, 1)))

        # For plotting result plot: Trajectory!
        inputs.append(('loc_on_map', tf.float32, (problem.batch_size, None, 3)))
        # minimal step from start point to goal location
        inputs.append(('gt_dist_to_goal', tf.float32, (problem.batch_size, None, 1)))
        # If reached goal location
        inputs.append(('if_reach_goal', tf.float32, (problem.batch_size, None, 1)))

        step_input_data, _ = tf_utils.setup_inputs(inputs)

        inputs = []
        # Added for policy gradient for fine tune
        inputs.append(('reward', tf.float32, (problem.batch_size, None, 1)))
        # Executed actions in one-hot form
        inputs.append(('executed_actions', tf.float32, (problem.batch_size, None, problem.num_actions)))
        # Optimal Actions
        inputs.append(('action', tf.float32, (problem.batch_size, None, problem.num_actions)))
        # Record final step status
        inputs.append(('final_step_if_reach', tf.float32, (problem.batch_size, 1)))
        train_data, _ = tf_utils.setup_inputs(inputs)
        train_data.update(step_input_data)
        train_data.update(common_input_data)
    return common_input_data, step_input_data, train_data


def setup_to_run(Z, args, is_training, batch_norm_is_training, summary_mode):
    # TODO realize single scale architecture
    assert (args.arch.multi_scale), 'removed support for old single scale code.'
    # Set up the Random seed.
    tf.compat.v1.set_random_seed(args.solver.seed)
    # Give the Problem
    navi = args.navi
    # Control the batch training
    batch_norm_param = {'center': True, 'scale': True,
                        'activation_fn': tf.nn.relu}
    batch_norm_is_training_op = \
        tf.compat.v1.placeholder_with_default(batch_norm_is_training, shape=[],
                                              name='batch_norm_is_training_op')
    batch_norm_param['is_training'] = batch_norm_is_training_op
    # Setup the inputs and intermediate output
    Z.input_tensors = {}
    Z.train_ops = {}
    Z.input_tensors['common'], Z.input_tensors['step'], Z.input_tensors['train'] = \
        _inputs(navi)
    Z.init_fn = None

    # num_steps = navi.num_steps
    map_crop_size_ops = []
    for map_crop_size in navi.map_crop_sizes:
        map_crop_size_ops.append(tf.constant(map_crop_size, dtype=tf.int32, shape=(2,)))

    ### Value vars for Planner
    Z.fr_ops = []
    Z.value_ops = []
    Z.fr_intermediate_ops = []
    Z.value_intermediate_ops = []
    Z.crop_value_ops = []
    Z.resize_crop_value_ops = []

    previous_value_op = None

    '''
        Inputs: 
            SemanticMap Input: B X H X W X Chns of Cats ( Refine Categories )
            ValueMap Input: B X H X W X ValueMap Chns (1 or 3)
            
        Record per step:
            Predicted Action:
            Optimal Action:
            Location if the action is taken: x, y in our coordinate system.
        
        Future Concern Output:
            fr_ops, value_ops
        
        Architecture parameters:
            Args.arch:
        
    '''
    for i in range(len(navi.map_crop_sizes)):
        map_crop_size = navi.map_crop_sizes[i]
        # The scale annotation are reversed here.
        with tf.variable_scope('scale_{:d}'.format(i)):

            sh = [-1, map_crop_size, map_crop_size, navi.map_channels]
            with tf.name_scope('CONV'):
                init_var = np.sqrt(2.0 / (3 ** 2) / args.arch.conv_neurons)

                x = Z.input_tensors['step']['locsmap_{:d}'.format(i)]
                # TODO args.arch.conv_neurons = 16 same as previous_value_op.shape[-1]
                x = slim.conv2d(tf.reshape(x, shape=sh), args.arch.conv_neurons, kernel_size=args.arch.conv_ks,
                                stride=1,
                                normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_param,
                                padding='SAME', scope='conv',
                                weights_regularizer=slim.l2_regularizer(args.solver.wt_decay),
                                weights_initializer=tf.random_normal_initializer(stddev=init_var))

            with tf.name_scope('concat'):
                # sh = [-1, map_crop_size, map_crop_size, navi.map_channels]

                onehot_semantic = tf.reshape(Z.input_tensors['step']['onehot_semantic'], shape=sh)
                to_concat = [x, onehot_semantic]
                if previous_value_op is not None:
                    to_concat.append(previous_value_op)
                x = tf.concat(to_concat, axis=3)

            # Resnet Conv Block

            # TODO args.arch.fr_neurons=16, fr_inside_neurons=32, (keep same as CMP)
            # TODO rename fr_neurons->fr_output_neurons,
            # TODO fr_inside_neurons->fr_bottleneck_base_depth

            fr_op, fr_intermediate_op = fr_v2(
                x, output_neurons=args.arch.fr_neurons,
                inside_neurons=args.arch.fr_inside_neurons,
                is_training=batch_norm_is_training_op, name='fr',
                wt_decay=args.solver.wt_decay, stride=args.arch.fr_stride)

            # TODO decide the architecture params
            if args.arch.vin_num_iters > 0:
                value_op, value_intermediate_op = value_iteration_network(
                    fr_op, num_iters=args.arch.vin_num_iters,
                    val_neurons=args.arch.vin_val_neurons,
                    action_neurons=args.arch.vin_action_neurons,
                    kernel_size=args.arch.vin_ks, share_wts=args.arch.vin_share_wts,
                    name='vin', wt_decay=args.solver.wt_decay)
            else:
                value_op = fr_op
                value_intermediate_op = []

            Z.value_intermediate_ops.append(value_intermediate_op)

            # Crop out and upsample the previous value map.
            # remove here is the margin that cropped out
            remove = args.arch.margin_crop_size
            if remove > 0:
                crop_value_op = value_op[:, remove:-remove, remove:-remove, :]
            else:
                crop_value_op = value_op

            crop_value_op = tf.reshape(crop_value_op, shape=[-1, args.arch.value_crop_size,
                                                             args.arch.value_crop_size,
                                                             args.arch.vin_val_neurons])
            if i < len(navi.map_crop_sizes) - 1:
                # Reshape it to shape of the next scale.
                previous_value_op = tf.compat.v1.image.resize_bilinear(crop_value_op,
                                                                       map_crop_size_ops[i + 1],
                                                                       align_corners=True)
                Z.resize_crop_value_ops.append(previous_value_op)

            Z.fr_ops.append(fr_op)
            Z.value_ops.append(value_op)
            Z.crop_value_ops.append(crop_value_op)
            Z.fr_intermediate_ops.append(fr_intermediate_op)

    Z.final_value_op = crop_value_op

    sh = [-1, args.arch.vin_val_neurons * (args.arch.value_crop_size ** 2)]
    Z.value_features_op = tf.reshape(Z.final_value_op, sh, name='reshape_value_op')

    # Determine what action to take.
    with tf.variable_scope('action_pred'):
        batch_norm_param = args.arch.pred_batch_norm_param
        if batch_norm_param is not None:
            batch_norm_param['is_training'] = batch_norm_is_training_op
        Z.action_logits_op, _ = fc_nn(
            Z.value_features_op, neurons=args.arch.pred_neurons,
            wt_decay=args.solver.wt_decay, name='pred', offset=0,
            num_pred=navi.num_actions,
            batch_norm_param=batch_norm_param)
        Z.action_prob_op = tf.nn.softmax(Z.action_logits_op)

    Z.train_ops['step'] = Z.action_prob_op
    Z.train_ops['common'] = [Z.input_tensors['common']['orig_maps'],
                             Z.input_tensors['common']['goal_loc']]
    Z.train_ops['batch_norm_is_training_op'] = batch_norm_is_training_op
    Z.loss_ops = []
    Z.loss_ops_names = []

    ewma_decay = 0.99 if is_training else 0.0
    weight = tf.ones_like(Z.input_tensors['train']['action'], dtype=tf.float32,
                          name='weight')

    # VS planner, change loss to policy gradient
    # Compute Loss
    Z.reg_loss_op, Z.data_loss_op, Z.total_loss_op, Z.acc_ops = \
        worker.compute_loss(Z.action_logits_op,
                            Z.input_tensors['train']['reward'],
                            Z.input_tensors['train']['executed_actions'],
                            Z.input_tensors['train']['action'],
                            Z.input_tensors['train']['final_step_if_reach'],
                            Z.input_tensors['train']['if_reach_goal'],
                            num_actions=navi.num_actions,
                            data_loss_wt=args.solver.data_loss_wt,
                            reg_loss_wt=args.solver.reg_loss_wt,
                            ewma_decay=ewma_decay)

    Z.loss_ops += [Z.reg_loss_op, Z.data_loss_op, Z.total_loss_op]
    Z.loss_ops_names += ['reg_loss', 'data_loss', 'total_loss']

    Z.lr_op, Z.global_step_op, Z.train_op, Z.should_stop_op, Z.optimizer, \
    Z.sync_optimizer = worker.setup_training(
        Z.total_loss_op,
        args.solver.initial_learning_rate,
        args.solver.steps_per_decay,
        args.solver.learning_rate_decay,
        args.solver.momentum,
        args.solver.max_steps,
        clip_gradient_norm=args.solver.clip_gradient_norm,
        typ=args.solver.typ, momentum2=args.solver.momentum2,
        adam_eps=args.solver.adam_eps)

    if args.solver.sample_gt_prob_type == 'inverse_sigmoid_decay':
        Z.sample_gt_prob_op = tf_utils.inverse_sigmoid_decay(args.arch.isd_k,
                                                             Z.global_step_op)
    elif args.solver.sample_gt_prob_type == 'teach_force':
        Z.sample_gt_prob_op = tf.constant(1., dtype=tf.float32)
    elif args.solver.sample_gt_prob_type == 'zero':
        Z.sample_gt_prob_op = tf.constant(-1.0, dtype=tf.float32)

    Z.sample_action_type = args.solver.sample_action_type
    Z.sample_action_combine_type = args.solver.sample_action_combine_type

    Z.summary_ops = {
        summary_mode: _add_summaries(Z, args, summary_mode,
                                     args.summary.arop_full_summary_iters)}

    Z.init_op = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())
    Z.saver_op = tf.train.Saver(keep_checkpoint_every_n_hours=4,
                                write_version=tf.train.SaverDef.V2)
    Z.batch_size = navi.batch_size
    return Z
