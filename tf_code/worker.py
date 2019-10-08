import numpy as np

import planner_finetune
from tf_utils import *
import src.utils as utils
# setup_to_run = planner_finetune.setup_to_run

gamma = 0.99


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.shape[0])):
        running_add = running_add * gamma + r[t][0]
        discounted_r[t][0] = running_add
    return discounted_r

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
    distrs = []
    for i in range(batch_size):
        distr_ = None
        if combine_type == 'one_or_other':
            sample_gt = rng.rand() < sample_gt_prob
            # When value is -1, rng.rand() creates number between [0,1)
            if sample_gt:
                distr_ = optimal_action_[i, :] * 1.
            else:
                distr_ = action_probs_[i, :] * 1.
        elif combine_type == 'add':
            distr_ = optimal_action_[i, :] * sample_gt_prob_ + \
                     (1. - sample_gt_prob_) * action_probs_[i, :]
            distr_ = distr_ / np.sum(distr_)
        distrs.append(distr_)
        if type == 'sample':
            action[i] = np.argmax(rng.multinomial(1, distr_, size=1))
        elif type == 'argmax':
            action[i] = np.argmax(distr_)
        action_sample_wt[i] = action_probs_[i, action[i]] / distr_[action[i]]
    return action, action_sample_wt, distrs


def compute_loss(logits, discounted_rewards, actions_one_hot, optimal_actions_onehot, final_step_if_reach,
                 if_reach_goal, num_actions=-1, data_loss_wt=1., reg_loss_wt=1.,
                 ewma_decay=0.99, reg_loss_op=None, batch_size=4):
    """

    :param logits: output of last fc before softmax
    :param discounted_rewards: discounted rewards (B x 39 x 1
    :param actions_one_hot: Executed action in one-hot form (B x 39 x len(actions)
    :param final_step_if_reach: Batch of `if_reach_goal` at the final state of each episode,
    :param if_reach_goal: `if_reach_goal` at each state of each batch. (B x 39 x 1)
    :param num_actions: int
    :param data_loss_wt: weight of data loss
    :param reg_loss_wt: weight of regulization loss
    :param ewma_decay:
    :param reg_loss_op:
    :param batch_size: int
    :return:
    """
    assert (num_actions > 0), 'num_actions must be specified and must be > 0.'

    with tf.name_scope('loss'):
        final_step_if_reach = tf.squeeze(final_step_if_reach, [-1])
        if_reach_goal = tf.squeeze(if_reach_goal, [-1])
        batch_total = tf.reduce_sum(if_reach_goal, reduction_indices=1)
        total = tf.reduce_sum(if_reach_goal)

        policy = tf.nn.softmax(logits)  # (Bx39) x 6
        policy = tf.reshape(policy, shape=[batch_size, -1, num_actions])  # B x 39 x 6
        action_prob = tf.reduce_sum(tf.multiply(policy, optimal_actions_onehot), reduction_indices=2)
        log_policy = tf.log(tf.clip_by_value(policy, 0.000001, 0.999999))
        multip = tf.multiply(log_policy, actions_one_hot)  # B x 39 x 6
        log_action_policy = tf.reduce_sum(multip, reduction_indices=2)  # B x 39
        # Get policy loss for each batch
        batch_policy_loss = tf.reduce_sum(tf.multiply(log_action_policy, tf.squeeze(discounted_rewards, [-1])), reduction_indices=1)  # B
        # make zero the unsuccessful trajs
        batch_policy_loss = batch_policy_loss * final_step_if_reach / tf.maximum(batch_total, tf.constant(1.))
        # Policy loss !!Notice the negative sign
        policy_loss = -tf.reduce_sum(batch_policy_loss / tf.reduce_sum(final_step_if_reach))  #

        # example_loss = -tf.math.log(tf.maximum(tf.constant(1e-4), log_action_policy))  # B x 39
        # example_loss = tf.multiply(example_loss, if_reach_goal)  # B x 39
        # # B. .If batch_total is zero all example_loss in that batch is all zeros.
        # batch_loss = tf.reduce_sum(example_loss, reduction_indices=1) / \
        #              tf.maximum(batch_total, tf.constant(1.))
        # data_loss_op = tf.reduce_sum(batch_loss) / batch_size
        if reg_loss_op is None:
            if reg_loss_wt > 0:
                reg_loss_op = tf.add_n(tf.losses.get_regularization_losses())
            else:
                reg_loss_op = tf.constant(0.)

        if reg_loss_wt > 0:
            total_loss_op = data_loss_wt * policy_loss + reg_loss_wt * reg_loss_op
        else:
            total_loss_op = data_loss_wt * policy_loss

        is_correct = tf.cast(tf.greater(action_prob, 0.5, name='pred_class'), tf.float32)
        is_correct = tf.multiply(is_correct, if_reach_goal)
        acc_op = tf.reduce_sum(is_correct) / total

        ewma_acc_op = moving_averages.weighted_moving_average(
            acc_op, ewma_decay, weight=total, name='ewma_acc')

        acc_ops = [ewma_acc_op]

    return reg_loss_op, policy_loss, total_loss_op, acc_ops

def setup_training(loss_op, initial_learning_rate, steps_per_decay,
                   learning_rate_decay, momentum, max_steps,
                   vars_to_optimize=None,
                   clip_gradient_norm=0, typ=None, momentum2=0.999,
                   adam_eps=1e-8):
    # Keep track of the number of batches seen so far
    global_step_op = slim.get_or_create_global_step()
    lr_op = tf.train.exponential_decay(initial_learning_rate,
                                       global_step_op, steps_per_decay, learning_rate_decay, staircase=True)
    if typ == 'sgd':
        optimizer = tf.train.MomentumOptimizer(lr_op, momentum)
    elif typ == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_op, beta1=momentum,
                                           beta2=momentum2, epsilon=adam_eps)
    else:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_op)


    sync_optimizer = None
    train_op = slim.learning.create_train_op(loss_op, optimizer,
                                             variables_to_train=vars_to_optimize,
                                             clip_gradient_norm=clip_gradient_norm)
    should_stop_op = tf.greater_equal(global_step_op, max_steps)

    return lr_op, global_step_op, train_op, should_stop_op, optimizer, sync_optimizer


def tune_fn(sess, train_op, global_step,
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
    testcheck['executed_actions'] = []
    testcheck['reachgoal'] = []
    testcheck['onehot'] = []
    testcheck['action_distr'] = []
    cnt = 0
    notwrite = True

    # iters = 1
    for i in range(iters):
        tt.tic()

        # Initialize the agent.
        init_env_state = agent.reset(rng_data[0], multi_target=['television', 'stand', 'desk', 'toilet'])
        # init_env_state = agent.reset(rng_data[0], single_target='sofa')
        # Given Fixed starting point
        # init_env_state = agent.startatPos([[-6, 23, 0]] * Z.batch_size)
        print(agent.epi.targets)

        # Obtain and Process the common data.
        input = agent.get_common_data()
        feed_dict = prepare_feed_dict(Z.input_tensors['common'], input)
        if dagger_sample_bn_false:
            feed_dict[Z.train_ops['batch_norm_is_training_op']] = False
        common_data = sess.run(Z.train_ops['common'], feed_dict=feed_dict)

        states = []
        state_features = []
        state_rewards = []
        state_actions = []
        executed_actions = []
        state_target_actions = []
        state_reach_goal = []
        action_sample_wts = []
        states.append(init_env_state)

        # num_steps = 1000
        for j in range(num_steps):
            f = agent.get_step_data()
            f['stop_gt_act_step_number'] = np.ones((1, 1, 1), dtype=np.int32) * j
            if j < num_steps - 1:
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

            dic_optimal_actions = vars(utils.Foo(action=optimal_action))
            state_target_actions.append(dic_optimal_actions)

            # There will be #num_steps of states visited,
            # but #num_steps-1 of actions executed
            if j < num_steps - 1:
                # Sample from action_probs and optimal action.
                action, action_sample_wt, distr = sample_action(
                    rng_action, action_probs, optimal_action, sample_gt_prob,
                    Z.sample_action_type, Z.sample_action_combine_type)
                # TODO get reward feedback

                # generate one-hot action
                onehot_action = np.zeros((Z.batch_size, len(agent.actions)), dtype=np.int32)
                onehot_action[np.arange(Z.batch_size), action] = 1
                onehot_action = onehot_action[:, np.newaxis, :]
                if mode == 'test' and notwrite:
                    testcheck['step'] += [[j]]
                    testcheck['localsmap'] += [
                        [f['locsmap_{:d}'.format(_)] for _ in range(len(agent.navi.map_orig_sizes))]]
                    testcheck['target'] = [agent.epi.targets]
                    testcheck['loc'] += [f['loc_on_map']]
                    testcheck['onehot'] += [[f['onehot_semantic']]]
                    testcheck['fr'] += [outs[2]]
                    testcheck['value'] += [outs[3]]
                    testcheck['executed_actions'] += [onehot_action]
                    testcheck['action_distr'] += [distr]
                    testcheck['targetloc'] = agent.epi.target_locs

                # Step a batch of actions
                state_reach_goal.append(vars(utils.Foo(if_reach_goal=
                                                       np.expand_dims(np.expand_dims(np.array(agent.reachgoal), axis=1), axis=1))))
                next_state = agent.step(action)
                state_rewards.append(agent.get_batch_rewards())
                state_actions.append(vars(utils.Foo(executed_actions=onehot_action)))

                executed_actions.append(action)
                states.append(next_state)
                action_sample_wts.append(action_sample_wt)

        if mode == 'test' and notwrite:
            # pickle.dump(testcheck, open('%s/fr_value_-1500_w1.pkl' % (logdir), 'wb'))
            notwrite = False

        iter_final_state = state_features[-1]['if_reach_goal']
        assert iter_final_state.shape[0] == Z.batch_size
        succ += np.logical_xor(np.ones(Z.batch_size), iter_final_state[:,0,0])

        total_cases += 1
        print('success rate in the %dth iteration.' % i, np.divide(succ, total_cases))

        all_state_rewards = concat_state_x(state_rewards, ['reward'])
        all_state_if_reach = concat_state_x(state_reach_goal, ['if_reach_goal'])
        allrewards = all_state_rewards['reward']
        if_reachs = all_state_if_reach['if_reach_goal']
        final_step_if_reach = np.ones((Z.batch_size, 1), dtype=np.float32)
        for i in range(Z.batch_size):
            # Check if at the final step, the agent reached target or not
            if allrewards[i, -1, 0] != 10. and np.sum(if_reachs[i]) == if_reachs.shape[1]:
                final_step_if_reach[i, 0] = 0.
        final_step_if_reach = {'final_step_if_reach': final_step_if_reach}
        allrewards = np.multiply(allrewards, if_reachs)
        for i in range(Z.batch_size):
            # Compute discounted rewards for each batch
            allrewards[i] = discount_rewards(allrewards[i])
        all_state_rewards['reward'] = allrewards
        all_state_targets = concat_state_x(state_target_actions, ['action'])
        all_state_features = concat_state_x(state_features, agent.get_step_data_names() + ['stop_gt_act_step_number'])
        all_state_executed_actions = concat_state_x(state_actions, ['executed_actions'])

        dict_train = dict(input)
        dict_train.update(all_state_features)
        dict_train.update(all_state_targets)
        dict_train.update(all_state_rewards)
        dict_train.update(all_state_executed_actions)
        dict_train.update(all_state_if_reach)
        dict_train.update(final_step_if_reach)
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

    return total_loss, should_stop
