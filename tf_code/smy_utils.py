import os, numpy as np
import matplotlib.pyplot as plt
import logging
import tensorflow as tf

from src import utils
from src import file_utils as fu
import nav_utils as nu


def _vis(outputs, global_step, output_dir, metric_summary, N):
    # Plot the value map, goal for various maps to see what if the model is
    # learning anything useful.
    #
    # outputs is [values, goals, maps, occupancy, conf].
    #
    if N >= 0:
        outputs = outputs[:N]
    N = len(outputs)

    plt.set_cmap('jet')
    fig, axes = utils.subplot(plt, (N, outputs[0][0].shape[4] * 5), (5, 5))
    axes = axes.ravel()[::-1].tolist()
    for i in range(N):
        values, goals, maps, occupancy, conf = outputs[i]
        for j in [0]:
            for k in range(values.shape[4]):
                # Display something like the midpoint of the trajectory.
                id = np.int(values.shape[1] / 2)

                ax = axes.pop();
                ax.imshow(goals[j, id, :, :, k], origin='lower', interpolation='none')
                ax.set_axis_off();
                if i == 0: ax.set_title('goal')

                ax = axes.pop();
                ax.imshow(occupancy[j, id, :, :, k], origin='lower', interpolation='none')
                ax.set_axis_off();
                if i == 0: ax.set_title('occupancy')

                ax = axes.pop();
                ax.imshow(conf[j, id, :, :, k], origin='lower', interpolation='none',
                          vmin=0., vmax=1.)
                ax.set_axis_off();
                if i == 0: ax.set_title('conf')

                ax = axes.pop();
                ax.imshow(values[j, id, :, :, k], origin='lower', interpolation='none')
                ax.set_axis_off();
                if i == 0: ax.set_title('value')

                ax = axes.pop();
                ax.imshow(maps[j, id, :, :, k], origin='lower', interpolation='none')
                ax.set_axis_off();
                if i == 0: ax.set_title('incr map')

    file_name = os.path.join(output_dir, 'value_vis_{:d}.png'.format(global_step))
    with fu.fopen(file_name, 'w') as f:
        fig.savefig(f, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close(fig)


def _summary_vis(Z, batch_size, num_steps, arop_full_summary_iters):
    arop = [];
    arop_summary_iters = [];
    arop_eval_fns = [];
    vis_value_ops = [];
    vis_goal_ops = [];
    vis_map_ops = [];
    vis_occupancy_ops = [];
    vis_conf_ops = [];
    for i, val_op in enumerate(Z.value_ops):
        vis_value_op = tf.reduce_mean(tf.abs(val_op), axis=3, keep_dims=True)
        vis_value_ops.append(vis_value_op)

        ego_goal_imgs_i_op = Z.input_tensors['step']['ego_goal_imgs_{:d}'.format(i)]
        vis_goal_op = tf.reduce_max(ego_goal_imgs_i_op, 4, True)
        vis_goal_ops.append(vis_goal_op)

        vis_map_op = tf.reduce_mean(tf.abs(Z.ego_map_ops[i]), 4, True)
        vis_map_ops.append(vis_map_op)

    vis_goal_ops = tf.concat(vis_goal_ops, 4)
    vis_map_ops = tf.concat(vis_map_ops, 4)
    vis_value_ops = tf.concat(vis_value_ops, 3)
    vis_occupancy_ops = tf.concat(vis_occupancy_ops, 3)
    vis_conf_ops = tf.concat(vis_conf_ops, 3)

    sh = tf.unstack(tf.shape(vis_value_ops))[1:]
    vis_value_ops = tf.reshape(vis_value_ops, shape=[batch_size, -1] + sh)

    sh = tf.unstack(tf.shape(vis_conf_ops))[1:]
    vis_conf_ops = tf.reshape(vis_conf_ops, shape=[batch_size, -1] + sh)

    sh = tf.unstack(tf.shape(vis_occupancy_ops))[1:]
    vis_occupancy_ops = tf.reshape(vis_occupancy_ops, shape=[batch_size, -1] + sh)

    # Save memory, only return time steps that need to be visualized, factor of
    # 32 CPU memory saving.
    id = np.int(num_steps / 2)
    vis_goal_ops = tf.expand_dims(vis_goal_ops[:, id, :, :, :], axis=1)
    vis_map_ops = tf.expand_dims(vis_map_ops[:, id, :, :, :], axis=1)
    vis_value_ops = tf.expand_dims(vis_value_ops[:, id, :, :, :], axis=1)
    vis_conf_ops = tf.expand_dims(vis_conf_ops[:, id, :, :, :], axis=1)
    vis_occupancy_ops = tf.expand_dims(vis_occupancy_ops[:, id, :, :, :], axis=1)

    arop += [[vis_value_ops, vis_goal_ops, vis_map_ops, vis_occupancy_ops,
              vis_conf_ops]]
    arop_summary_iters += [arop_full_summary_iters]
    arop_eval_fns += [_vis]
    return arop, arop_summary_iters, arop_eval_fns


def _add_summaries(Z, args, summary_mode, arop_full_summary_iters):
    task_params = args.navi

    summarize_ops = [Z.lr_op, Z.global_step_op, Z.sample_gt_prob_op] + \
                    Z.loss_ops + Z.acc_ops
    summarize_names = ['lr', 'global_step', 'sample_gt_prob_op'] + \
                      Z.loss_ops_names + ['acc_{:d}'.format(i) for i in range(len(Z.acc_ops))]
    to_aggregate = [0, 0, 0] + [1] * len(Z.loss_ops_names) + [1] * len(Z.acc_ops)

    scope_name = 'summary'
    with tf.name_scope(scope_name):
        s_ops = nu.add_default_summaries(summary_mode, arop_full_summary_iters,
                                         summarize_ops, summarize_names,
                                         to_aggregate, Z.action_prob_op,
                                         Z.input_tensors, scope_name=scope_name)
        if summary_mode == 'val':
            arop, arop_summary_iters, arop_eval_fns = _summary_vis(
                Z, task_params.batch_size, task_params.num_steps,
                arop_full_summary_iters)
            s_ops.additional_return_ops += arop
            s_ops.arop_summary_iters += arop_summary_iters
            s_ops.arop_eval_fns += arop_eval_fns

    return s_ops


def get_default_summary_ops():
    return utils.Foo(summary_ops=None, print_summary_ops=None,
                     additional_return_ops=[], arop_summary_iters=[],
                     arop_eval_fns=[])


def simple_summaries(summarize_ops, summarize_names, mode, to_aggregate=False,
                     scope_name='summary'):
    if type(to_aggregate) != list:
        to_aggregate = [to_aggregate for _ in summarize_ops]

    summary_key = '{:s}_summaries'.format(mode)
    print_summary_key = '{:s}_print_summaries'.format(mode)
    prefix = ' [{:s}]: '.format(mode)

    # Default ops for things that dont need to be aggregated.
    if not np.all(to_aggregate):
        for op, name, to_agg in zip(summarize_ops, summarize_names, to_aggregate):
            if not to_agg:
                add_scalar_summary_op(op, name, summary_key, print_summary_key, prefix)
        summary_ops = tf.summary.merge_all(summary_key)
        print_summary_ops = tf.summary.merge_all(print_summary_key)
    else:
        summary_ops = tf.no_op()
        print_summary_ops = tf.no_op()

    # Default ops for things that dont need to be aggregated.
    if np.any(to_aggregate):
        additional_return_ops = [[summarize_ops[i]
                                  for i, x in enumerate(to_aggregate) if x]]
        arop_summary_iters = [-1]
        s_names = ['{:s}/{:s}'.format(scope_name, summarize_names[i])
                   for i, x in enumerate(to_aggregate) if x]
        fn = lambda outputs, global_step, output_dir, metric_summary, N: \
            accum_val_ops(outputs, s_names, global_step, output_dir, metric_summary,
                          N)
        arop_eval_fns = [fn]
    else:
        additional_return_ops = []
        arop_summary_iters = []
        arop_eval_fns = []
    return summary_ops, print_summary_ops, additional_return_ops, \
           arop_summary_iters, arop_eval_fns


def add_value_to_summary(metric_summary, tag, val, log=True, tag_str=None):
    """Adds a scalar summary to the summary object. Optionalhttps://wx2.qq.com/cgi-bin/mmwebwx-bin/webwxgetmsgimg?&MsgID=7245434972996279970&skey=%40crypt_376f7637_d35a4a4be6d9d9379a29cabe438fd0c2ly also logs to
  logging."""
    new_value = metric_summary.value.add()
    new_value.tag = tag
    new_value.simple_value = val
    if log:
        if tag_str is None:
            tag_str = tag + '%f'
        logging.info(tag_str, val)


def add_scalar_summary_op(tensor, name=None,
                          summary_key='summaries', print_summary_key='print_summaries', prefix=''):
    collections = []
    op = tf.summary.scalar(name, tensor, collections=collections)
    if summary_key != print_summary_key:
        tf.add_to_collection(summary_key, op)

    op = tf.Print(op, [tensor], '    {:-<25s}: '.format(name) + prefix)
    tf.add_to_collection(print_summary_key, op)
    return op


def accum_val_ops(outputs, names, global_step, output_dir, metric_summary, N):
    """Processes the collected outputs to compute AP for action prediction.

  Args:
    outputs        : List of scalar ops to summarize.
    names          : Name of the scalar ops.
    global_step    : global_step.
    output_dir     : where to store results.
    metric_summary : summary object to add summaries to.
    N              : number of outputs to process.
  """
    outs = []
    if N >= 0:
        outputs = outputs[:N]
    for i in range(len(outputs[0])):
        scalar = np.array(map(lambda x: x[i], outputs))
        assert (scalar.ndim == 1)
        add_value_to_summary(metric_summary, names[i], np.mean(scalar),
                             tag_str='{:>27s}:  [{:s}]: %f'.format(names[i], ''))
        outs.append(np.mean(scalar))
    return outs

