import numpy as np
import logging
import pickle
from src.utils import Foo
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.python.training import moving_averages

from src import utils

resnet_v2_50 = resnet_v2.resnet_v2_50


def setup_train_step_kwargs(Z, agent, logdir, rng_seed, is_chief, num_steps,
                            iters, train_display_interval,
                            dagger_sample_bn_false):
    # rng_data has 2 independent rngs, one for sampling episodes and one for
    # sampling perturbs (so that we can make results reproducible.
    train_step_kwargs = {'agent': agent, 'Z': Z, 'rng_data': [np.random.RandomState(rng_seed),
                                                          np.random.RandomState(rng_seed)],
                         'rng_action': np.random.RandomState(rng_seed)}

    if is_chief:
        train_step_kwargs['writer'] = tf.summary.FileWriter(logdir)  # , Z.tf_graph)
    else:
        train_step_kwargs['writer'] = None
    train_step_kwargs['iters'] = iters
    train_step_kwargs['train_display_interval'] = train_display_interval
    train_step_kwargs['num_steps'] = num_steps
    train_step_kwargs['logdir'] = logdir
    train_step_kwargs['dagger_sample_bn_false'] = dagger_sample_bn_false
    return train_step_kwargs


def train_step_fn(sess, train_op, global_step,
                  train_step_kwargs, mode='train'):
    Z = train_step_kwargs['Z']
    agent = train_step_kwargs['agent']
    rng_data = train_step_kwargs['rng_data']
    rng_action = train_step_kwargs['rng_action']
    writer = train_step_kwargs['writer']
    iters = train_step_kwargs['iters']
    num_steps = train_step_kwargs['num_steps']
    logdir = train_step_kwargs['logdir']
    dagger_sample_bn_false = train_step_kwargs['dagger_sample_bn_false']
    train_display_interval = train_step_kwargs['train_display_interval']
    if 'outputs' not in Z.train_ops:
        Z.train_ops['outputs'] = []

    s_ops = Z.summary_ops[mode]
    val_additional_ops = []

    # Print all variables here.
    # if True:
    #     v = tf.get_collection(tf.GraphKeys.VARIABLES)
    #     v_op = [_.value() for _ in v]
    #     v_op_value = sess.run(v_op)
    #
    #     filter = lambda x, y: 'Adam' in x.name
    #     # filter = lambda x, y: np.is_any_nan(y)
    #     ind = [i for i, (_, __) in enumerate(zip(v, v_op_value)) if filter(_, __)]
    #     v = [v[i] for i in ind]
    #     v_op_value = [v_op_value[i] for i in ind]
    #
    #     for i in range(len(v)):
    #         logging.info('XXXX: variable: %30s, is_any_nan: %5s, norm: %f.',
    #                      v[i].name, np.any(np.isnan(v_op_value[i])),
    #                      np.linalg.norm(v_op_value[i]))

    tt = utils.Timer()

    total_loss = should_stop = None

    # Test outputs
    total_cases = np.zeros((Z.batch_size))
    succ = np.zeros((Z.batch_size))

    testcheck = {}
    testcheck['iter'] = []
    testcheck['step'] = []
    testcheck['localsmap'] = []
    testcheck['target'] = []
    testcheck['loc'] = []
    testcheck['fr'] = []
    testcheck['value'] = []
    testcheck['excuted_actions'] = []
    testcheck['reachgoal'] = []
    cnt = 0
    for i in range(iters):
        tt.tic()

        # Initialize the agent.
        # init_env_state = agent.reset(rng_data[0], multi_target=['television', 'stand', 'desk', 'toilet'])
        init_env_state = agent.reset(rng_data[0], single_target='sofa')
        print(agent.epi.targets)

        # Get and process the common data.
        input = agent.get_common_data()
        feed_dict = prepare_feed_dict(Z.input_tensors['common'], input)
        if dagger_sample_bn_false:
            feed_dict[Z.train_ops['batch_norm_is_training_op']] = False
        common_data = sess.run(Z.train_ops['common'], feed_dict=feed_dict)

        states = []
        state_features = []
        state_target_actions = []
        executed_actions = []
        reachgoal = []
        rewards = []
        action_sample_wts = []
        states.append(init_env_state)

        for j in range(num_steps):
            f = agent.get_step_data()
            f['stop_gt_act_step_number'] = np.ones((1, 1, 1), dtype=np.int32) * j
            state_features.append(f)

            feed_dict = prepare_feed_dict(Z.input_tensors['step'], state_features[-1])  # Feed in latest state features
            optimal_action = agent.get_batch_gt_actions()
            for x, v in zip(Z.train_ops['common'], common_data):
                feed_dict[x] = v
            if dagger_sample_bn_false:
                feed_dict[Z.train_ops['batch_norm_is_training_op']] = False
            outs = sess.run([Z.train_ops['step'], Z.sample_gt_prob_op,
                             Z.fr_ops, Z.value_ops
                             ],
                            feed_dict=feed_dict)
            action_probs = outs[0]
            sample_gt_prob = outs[1]

            dic_optimal_actions = vars(Foo(action=optimal_action))
            state_target_actions.append(dic_optimal_actions)

            if j < num_steps - 1:
                # Sample from action_probs and optimal action.
                action, action_sample_wt = sample_action(
                    rng_action, action_probs, optimal_action, sample_gt_prob,
                    Z.sample_action_type, Z.sample_action_combine_type)
                # TODO get reward feedback
                # next_state, reward = agent.step(action)
                # locs = f['loc_on_map']

                if mode == 'test' and cnt < 10:

                    for btch in range(Z.batch_size):
                        target_loc = common_data[1][btch]; crnt_loc = f['loc_on_map'][btch]
                        xt = target_loc[0][0]; yt = target_loc[0][1]
                        xc = crnt_loc[0][0]; yc = crnt_loc[0][1]; orienc = crnt_loc[0][2]
                        # if abs(xt - xc) + abs(yt - yc) < 10 and orienc == 0:
                        if xc in range(-2, 3) and yc in range(0, 11) and orienc == 0 and \
                                agent.epi.targets[btch] == u'table_and_chair':
                            testcheck['iter'] += [[i]]
                            testcheck['step'] += [[j]]
                            testcheck['localsmap'] += [[f['locsmap_{:d}'.format(_)][btch][0] for _ in range(len(agent.navi.map_orig_sizes))]]
                            testcheck['target'] += [agent.epi.targets[btch]]
                            testcheck['loc'] += [[xc, yc, orienc]]
                            testcheck['fr'] += [[outs[2][sc][btch] for sc in range(3)]]
                            testcheck['value'] += [[outs[3][sc][btch] for sc in range(3)]]
                            testcheck['excuted_actions'] += [action[btch]]
                            # testcheck['reachgoal'] += reachgoal

                            cnt += 1

                if mode == 'test' and 6 <= cnt <= 10:
                    pickle.dump(testcheck, open('%s/fr_value_sofa.pkl' % (logdir), 'wb'))
                    cnt += 1

                # Step a batch of actions
                next_state = agent.step(action)
                reachgoal.append(agent.reachgoal)
                executed_actions.append(action)
                states.append(next_state)
                # rewards.append(reward)
                action_sample_wts.append(action_sample_wt)
                # net_state = dict(zip(Z.train_ops['state_names'], net_state))
                # net_state_to_input.append(net_state)

        # Concatenate things together for training.
        # rewards = np.array(rewards).T
        # action_sample_wts = np.array(action_sample_wts).T
        # executed_actions = np.array(executed_actions).T
        iter_final_state = state_features[-1]['if_reach_goal']
        assert iter_final_state.shape[0] == Z.batch_size
        succ += np.logical_xor(np.ones(Z.batch_size), iter_final_state[:,0,0])

        total_cases += 1
        print('success rate in the %dth iteration.' % i, np.divide(succ, total_cases))

        all_state_targets = concat_state_x(state_target_actions, ['action'])
        all_state_features = concat_state_x(state_features, agent.get_step_data_names() + ['stop_gt_act_step_number'])
        # all_state_net = concat_state_x(net_state_to_input,
        # Z.train_ops['state_names'])
        # all_step_data_cache = concat_state_x(step_data_cache,
        #                                      Z.train_ops['step_data_cache'])

        dict_train = dict(input)
        dict_train.update(all_state_features)
        dict_train.update(all_state_targets)

        # dict_train.update({     # 'rewards': rewards,
        #                    'action_sample_wts': action_sample_wts,
        #                    'executed_actions': executed_actions})
        feed_dict = prepare_feed_dict(Z.input_tensors['train'], dict_train)

        if mode == 'train':
            n_step = sess.run(global_step)
            print(n_step)
            if np.mod(n_step, train_display_interval) == 0:
                total_loss, np_global_step, summary, print_summary = sess.run(
                    [train_op, global_step, s_ops.summary_ops, s_ops.print_summary_ops],
                    feed_dict=feed_dict)
                logging.error("")
            else:
                total_loss, np_global_step, summary = sess.run(
                    [train_op, global_step, s_ops.summary_ops], feed_dict=feed_dict)

            if writer is not None and summary is not None:
                writer.add_summary(summary, np_global_step)

            should_stop = sess.run(Z.should_stop_op)

        if mode != 'train':
            arop = [[] for j in range(len(s_ops.additional_return_ops))]
            for j in range(len(s_ops.additional_return_ops)):
                if s_ops.arop_summary_iters[j] < 0 or i < s_ops.arop_summary_iters[j]:
                    arop[j] = s_ops.additional_return_ops[j]
            val = sess.run(arop, feed_dict=feed_dict)
            val_additional_ops.append(val)
            tt.toc(log_at=60, log_str='val timer {:d} / {:d}: '.format(i, iters),
                   type='time')

    if mode != 'train':
        # Write the default val summaries.
        summary, print_summary, np_global_step = sess.run(
            [s_ops.summary_ops, s_ops.print_summary_ops, global_step])
        if writer is not None and summary is not None:
            writer.add_summary(summary, np_global_step)

        # write custom validation ops
        val_summarys = []
        val_additional_ops = zip(*val_additional_ops)
        if len(s_ops.arop_eval_fns) > 0:
            val_metric_summary = tf.summary.Summary()
            for i in range(len(s_ops.arop_eval_fns)):
                val_summary = None
                if s_ops.arop_eval_fns[i] is not None:
                    val_summary = s_ops.arop_eval_fns[i](val_additional_ops[i],
                                                         np_global_step, logdir,
                                                         val_metric_summary,
                                                         s_ops.arop_summary_iters[i])
                val_summarys.append(val_summary)
            if writer is not None:
                writer.add_summary(val_metric_summary, np_global_step)

        # Return the additional val_ops
        total_loss = (val_additional_ops, val_summarys)
        should_stop = None

    return total_loss, should_stop


def setup_training(loss_op, initial_learning_rate, steps_per_decay,
                   learning_rate_decay, momentum, max_steps,
                   sync=False, adjust_lr_sync=True,
                   num_workers=1, replica_id=0, vars_to_optimize=None,
                   clip_gradient_norm=0, typ=None, momentum2=0.999,
                   adam_eps=1e-8):
    if sync and adjust_lr_sync:
        initial_learning_rate = initial_learning_rate * num_workers
        max_steps = np.int(max_steps / num_workers)
        steps_per_decay = np.int(steps_per_decay / num_workers)

    # Keep track of the number of batches seen so far
    global_step_op = slim.get_or_create_global_step()
    lr_op = tf.train.exponential_decay(initial_learning_rate,
                                       global_step_op, steps_per_decay, learning_rate_decay, staircase=True)
    if typ == 'sgd':
        optimizer = tf.train.MomentumOptimizer(lr_op, momentum)
    elif typ == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_op, beta1=momentum,
                                           beta2=momentum2, epsilon=adam_eps)

    if sync:

        sync_optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                        replicas_to_aggregate=num_workers,
                                                        replica_id=replica_id,
                                                        total_num_replicas=num_workers)
        train_op = slim.learning.create_train_op(loss_op, sync_optimizer,
                                                 variables_to_train=vars_to_optimize,
                                                 clip_gradient_norm=clip_gradient_norm)
    else:
        sync_optimizer = None
        train_op = slim.learning.create_train_op(loss_op, optimizer,
                                                 variables_to_train=vars_to_optimize,
                                                 clip_gradient_norm=clip_gradient_norm)
        should_stop_op = tf.greater_equal(global_step_op, max_steps)
    return lr_op, global_step_op, train_op, should_stop_op, optimizer, sync_optimizer


def compute_losses_multi_or(logits, actions_one_hot, if_reach_goal, weights=None,
                            num_actions=-1, data_loss_wt=1., reg_loss_wt=1.,
                            ewma_decay=0.99, reg_loss_op=None, batch_size=4):
    assert (num_actions > 0), 'num_actions must be specified and must be > 0.'

    with tf.name_scope('loss'):
        if_reach_goal = tf.squeeze(if_reach_goal, [-1])
        batch_total = tf.reduce_sum(if_reach_goal, reduction_indices=1)
        total = tf.reduce_sum(if_reach_goal)

        if weights is None:
            weights = tf.ones_like(actions_one_hot, dtype=tf.float32, name='weight')

        action_prob = tf.nn.softmax(logits) # (Bx40) x 6
        action_prob = tf.reshape(action_prob, shape=[batch_size, -1, num_actions]) # B x 40 x 6
        multip = tf.multiply(action_prob, actions_one_hot)  # B x 40 x 6
        action_prob = tf.reduce_sum(multip, reduction_indices=2) # B x 40

        example_loss = -tf.math.log(tf.maximum(tf.constant(1e-4), action_prob))  # B x 40
        example_loss = tf.multiply(example_loss, if_reach_goal)  # B x 40
        batch_loss = tf.reduce_sum(example_loss, reduction_indices=1) / \
                     tf.maximum(batch_total, tf.constant(1.))  # B. .If batch_total is zero all example_loss in that batch is all zeros.
        data_loss_op = tf.reduce_sum(batch_loss) / batch_size
        if reg_loss_op is None:
            if reg_loss_wt > 0:
                reg_loss_op = tf.add_n(tf.losses.get_regularization_losses())
            else:
                reg_loss_op = tf.constant(0.)

        if reg_loss_wt > 0:
            total_loss_op = data_loss_wt * data_loss_op + reg_loss_wt * reg_loss_op
        else:
            total_loss_op = data_loss_wt * data_loss_op

        is_correct = tf.cast(tf.greater(action_prob, 0.5, name='pred_class'), tf.float32)
        is_correct = tf.multiply(is_correct, if_reach_goal)
        acc_op = tf.reduce_sum(is_correct) / total

        ewma_acc_op = moving_averages.weighted_moving_average(
            acc_op, ewma_decay, weight=total, name='ewma_acc')

        acc_ops = [ewma_acc_op]

    return reg_loss_op, data_loss_op, total_loss_op, acc_ops


def sample_action(rng, action_probs, optimal_action, sample_gt_prob,
                  type='sample', combine_type='one_or_other'):
    """

    :param rng: np.rand.RandomState
    :param action_probs: action predictions
    :param optimal_action: onehot optimal actions: B x 1 x #actions
    :param sample_gt_prob: probability to sample
    :param type:
    :param combine_type:
    :return:
    """
    optimal_action = np.squeeze(optimal_action, axis=1)
    optimal_action_ = optimal_action / np.sum(optimal_action + 0., 1, keepdims=True)
    action_probs_ = action_probs / np.sum(action_probs + 0.001, 1, keepdims=True)
    batch_size = action_probs_.shape[0]

    action = np.zeros((batch_size), dtype=np.int32)
    action_sample_wt = np.zeros((batch_size), dtype=np.float32)

    sample_gt_prob_ = None
    if combine_type == 'add':
        sample_gt_prob_ = np.minimum(np.maximum(sample_gt_prob, 0.), 1.)

    for i in range(batch_size):
        distr_ = None
        if combine_type == 'one_or_other':
            sample_gt = rng.rand() < sample_gt_prob
            # Since this value is -1, rng.rand() creates number between [0,1)
            if sample_gt:
                distr_ = optimal_action_[i, :] * 1.
            else:
                distr_ = action_probs_[i, :] * 1.
        elif combine_type == 'add':
            distr_ = optimal_action_[i, :] * sample_gt_prob_ + \
                     (1. - sample_gt_prob_) * action_probs_[i, :]
            distr_ = distr_ / np.sum(distr_)

        if type == 'sample':
            action[i] = np.argmax(rng.multinomial(1, distr_, size=1))
        elif type == 'argmax':
            action[i] = np.argmax(distr_)
        action_sample_wt[i] = action_probs_[i, action[i]] / distr_[action[i]]
    return action, action_sample_wt

def setup_inputs(inputs):
    input_tensors = {}
    input_shapes = {}
    for (name, typ, sz) in inputs:
        _ = tf.compat.v1.placeholder(typ, shape=sz, name=name)
        input_tensors[name] = _
        input_shapes[name] = sz
    return input_tensors, input_shapes


def prepare_feed_dict(input_tensors, inputs):
    feed_dict = {}
    for n in input_tensors.keys():
        feed_dict[input_tensors[n]] = inputs[n].astype(input_tensors[n].dtype.as_numpy_dtype)
    return feed_dict


def concat_state_x(f, names):
    af = {}
    for k in names:
        af[k] = np.concatenate([x[k] for x in f], axis=1)
        # af[k] = np.swapaxes(af[k], 0, 1)
    return af


def inverse_sigmoid_decay(k, global_step_op):
    with tf.name_scope('inverse_sigmoid_decay'):
        k = tf.constant(k, dtype=tf.float32)
        tmp = k * tf.exp(-tf.cast(global_step_op, tf.float32) / k)
        tmp = tmp / (1. + tmp)
    return tmp


def add_value_to_summary(metric_summary, tag, val, log=True, tag_str=None):
    """Adds a scalar summary to the summary object. Optionally also logs to
  logging."""
    new_value = metric_summary.value.add()
    new_value.tag = tag
    new_value.simple_value = val
    if log:
        if tag_str is None:
            tag_str = tag + '%f'
        logging.info(tag_str, val)
