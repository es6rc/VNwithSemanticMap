r""" Script to train and test the grid navigation agent.
Usage:
  1. Testing a model.
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 \
    PYTHONPATH='.' PYOPENGL_PLATFORM=egl python scripts/script_nav_agent_release.py \
    --config_name cmp.lmap_Msc.clip5.sbpd_d_r2r+bench_test \
    --logdir output/cmp.lmap_Msc.clip5.sbpd_d_r2r

  2. Training a model (locally).
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64 \
    PYTHONPATH='.' PYOPENGL_PLATFORM=egl python scripts/script_nav_agent_release.py \
    --config_name cmp.lmap_Msc.clip5.sbpd_d_r2r+train_train \
    --logdir output/cmp.lmap_Msc.clip5.sbpd_d_r2r_

  3. Training a model (distributed).
  # See https://www.tensorflow.org/deploy/distributed on how to setup distributed
  # training.
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 \
    PYTHONPATH='.' PYOPENGL_PLATFORM=egl python scripts/script_nav_agent_release.py \
    --config_name cmp.lmap_Msc.clip5.sbpd_d_r2r+train_train \
    --logdir output/cmp.lmap_Msc.clip5.sbpd_d_r2r_ \
    --ps_tasks $num_ps --master $master_name --task $worker_id
"""

import sys, os, numpy as np
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.framework import ops

os.environ['PYTHONPATH'] = '../'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/include"

import time, logging

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.platform import flags
from tensorflow.python.platform import app

from src import utils
from src import cfg
from src import file_utils as fu
from src import navi_env
import tf_utils
import planner

FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', 'output/planner_multitar', '')

# flags.DEFINE_string('config_name', 'Msc.clip5.house3d_rgb_r2r+train_train', '')
flags.DEFINE_string('config_name', 'Msc.clip5.house3d_rgb_r2r+test_bench', '')
# '<arch_str>.<solver_str>.<navi_str>+<mode>'
'''
    <arch_str>      Network Architecture Settings
    <solver_str>    Trainning Settings, Hparam
    <navi_str>      Interaction Settings with Environment
    <mode>          Train or Test
'''
flags.DEFINE_string('num_workers', '1', '')
flags.DEFINE_integer('solver_seed', 0, '')
flags.DEFINE_integer('task', 0, 'The Task ID. This value is used when training '
                                'with multiple workers to identify each worker.')
flags.DEFINE_string('sample_gt_action_prob_type', 'teach_force', '')

DELAY_START_ITERS = 20

def main(_):
    _launcher(FLAGS.config_name, FLAGS.logdir)


def _launcher(config_name, logdir):
    _args = _setup_args(config_name, logdir)

    fu.makedirs(_args.logdir)

    if _args.control.train:
        _train(_args)

    if _args.control.test:
        _test(_args)


def _setup_args(config_name, logdir):

    configs, mode_str = config_name.split('+')
    configs = configs.split('.')
    arch_str, solver_str, navi_str = configs[0], configs[1], configs[2]

    args = utils.Foo()
    args.logdir = logdir
    args.solver = cfg.setup_args_solver(solver_str)
    args.navi = cfg.setup_args_navi(navi_str)
    args = cfg.setup_args_arch(args, arch_str)
    args.summary, args.control = cfg.setup_args_sum_ctrl()

    args.solver.num_workers = FLAGS.num_workers
    args.solver.task = FLAGS.task
    args.solver.seed = FLAGS.solver_seed
    args.solver.sample_gt_prob_type = FLAGS.sample_gt_action_prob_type

    if mode_str == "test_bench":
        args = cfg.setup_smyctl(args, mode_str)
    elif mode_str == "train_train":
        args.control.train = True


    args.setup_to_run = planner.setup_to_run
    args.setup_train_step_kwargs = tf_utils.setup_train_step_kwargs
    return args


def _train(args):
    agent = navi_env.Environment('5cf0e1e9493994e483e985c436b9d3bc', args.navi)
    Z = utils.Foo()
    Z.tf_graph = tf.Graph()

    config = tf.ConfigProto()
    config.device_count['GPU'] = 1

    with Z.tf_graph.as_default():
        with tf.device(tf.train.replica_device_setter(args.solver.ps_tasks,
                                                      merge_devices=True)):
            with tf.container("planner"):
                Z = args.setup_to_run(Z, args, is_training=True,
                                      batch_norm_is_training=True,
                                      summary_mode='train')
                train_step_kwargs = args.setup_train_step_kwargs(
                    Z, agent, os.path.join(args.logdir, 'train'), rng_seed=args.solver.rng_seed,
                    is_chief=args.solver.rng_seed == 0,
                    num_steps=args.navi.num_steps * args.navi.num_goals, iters=1,
                    train_display_interval=args.summary.display_interval,
                    dagger_sample_bn_false=args.solver.dagger_sample_bn_false)

                delay_start = (args.solver.task * (args.solver.task + 1)) / 2 * DELAY_START_ITERS
                logging.info('delaying start for task %d by %d steps.',
                              args.solver.task, delay_start)
                
                additional_args = {}
                final_loss = slim.learning.train(
                    train_op=Z.train_op,
                    logdir=args.logdir,
                    is_chief=args.solver.task == 0,
                    number_of_steps=args.solver.max_steps,
                    train_step_fn=tf_utils.train_step_fn,
                    train_step_kwargs=train_step_kwargs,
                    master=args.solver.master,
                    global_step=Z.global_step_op,
                    init_op=Z.init_op,
                    init_fn=Z.init_fn,
                    sync_optimizer=Z.sync_optimizer,
                    saver=Z.saver_op,
                    save_summaries_secs=5000,
                    save_interval_secs=5000,
                    startup_delay_steps=delay_start,
                    summary_op=None, session_config=config, **additional_args)


def _test(args):
    # Give checkpoint directory
    container_name = ""
    checkpoint_dir = os.path.join(format(args.logdir))
    logging.error('Checkpoint_dir: %s', args.logdir)
    # Load Agent
    agent = navi_env.Environment('5cf0e1e9493994e483e985c436b9d3bc', args.navi)
    # Add Configure
    config = tf.compat.v1.ConfigProto()
    config.device_count['GPU'] = 1

    Z = utils.Foo()
    Z.tf_graph = tf.Graph()
    with Z.tf_graph.as_default():
        with tf.compat.v1.container(container_name):
            Z = args.setup_to_run(
                Z, args, is_training=False,
                batch_norm_is_training=args.control.force_batchnorm_is_training_at_test,
                summary_mode=args.control.test_mode)
            train_step_kwargs = args.setup_train_step_kwargs(
                Z, agent, os.path.join(args.logdir, args.control.test_name),
                rng_seed=1008, is_chief=True,
                num_steps=args.navi.num_steps * args.navi.num_goals,
                iters=args.summary.test_iters, train_display_interval=None,
                dagger_sample_bn_false=args.solver.dagger_sample_bn_false)

            saver = slim.learning.tf_saver.Saver(variables.get_variables_to_restore())

            sv = slim.learning.supervisor.Supervisor(
                graph=ops.get_default_graph(), logdir=None, init_op=Z.init_op,
                summary_op=None, summary_writer=None, global_step=None, saver=Z.saver_op)

            last_checkpoint = None
            # reported = False
            while True:
                last_checkpoint_ = None
                while last_checkpoint_ is None:
                    last_checkpoint_ = slim.evaluation.wait_for_new_checkpoint(
                        checkpoint_dir, last_checkpoint, seconds_to_sleep=10, timeout=60)
                if last_checkpoint_ is None:
                    break

                last_checkpoint = last_checkpoint_
                checkpoint_iter = int(os.path.basename(last_checkpoint).split('-')[1])

                logging.info('Starting evaluation at %s using checkpoint %s.',
                             time.strftime('%Y-%Z-%d-%H:%Z:%S', time.localtime()),
                             last_checkpoint)

                if (not args.control.only_eval_when_done or
                        checkpoint_iter >= args.solver.max_steps):
                    # start = time.time()
                    logging.info('Starting evaluation at %s using checkpoint %s.',
                                 time.strftime('%Y-%Z-%d-%H:%Z:%S', time.localtime()),
                                 last_checkpoint)

                    with sv.managed_session(args.solver.master, config=config,
                                            start_standard_services=False) as sess:
                        sess.run(Z.init_op)
                        sv.saver.restore(sess, last_checkpoint)
                        sv.start_queue_runners(sess)
                        if args.control.reset_rng_seed:
                            train_step_kwargs['rng_data'] = [np.random.RandomState(1008),
                                                             np.random.RandomState(1008)]
                            train_step_kwargs['rng_action'] = np.random.RandomState(1008)
                        vals, _ = tf_utils.train_step_fn(
                            sess, None, Z.global_step_op, train_step_kwargs,
                            mode=args.control.test_mode)
                        should_stop = False

                        if checkpoint_iter >= args.solver.max_steps:
                            should_stop = True

                        if should_stop:
                            break

if __name__ == '__main__':
    app.run()
