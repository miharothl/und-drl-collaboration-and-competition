import time

import torch
import numpy as np
from collections import deque
import sys

from drl import drl_logger
from drl.agent.i_agent import IAgent
from drl.env.i_environment import IEnvironment
from drl.experiment.configuration import Configuration
from drl.experiment.config.trainer_config import TrainerConfig
from drl.experiment.recorder import Recorder
from drl.experiment.train.trainer import Trainer


class MasterTrainer(Trainer):
    def __init__(self, cfg: Configuration, session_id, path_models='models'):

        super(MasterTrainer, self).__init__(cfg, session_id)

        reinforcement_learning_cfg = self.cfg.get_current_exp_cfg().reinforcement_learning_cfg

        if cfg.get_current_exp_cfg().reinforcement_learning_cfg.algorithm_type.startswith('ddpg')\
                or cfg.get_current_exp_cfg().reinforcement_learning_cfg.algorithm_type.startswith('maddpg'):
            self.eps = reinforcement_learning_cfg.ddpg_cfg.epsilon_start   # initialize epsilon
            self.eps_end = reinforcement_learning_cfg.ddpg_cfg.epsilon_end
            self.eps_decay = reinforcement_learning_cfg.ddpg_cfg.epsilon_decay
        else:
            self.eps = reinforcement_learning_cfg.dqn_cfg.epsilon_start   # initialize epsilon
            self.eps_end = reinforcement_learning_cfg.dqn_cfg.epsilon_end
            self.eps_decay = reinforcement_learning_cfg.dqn_cfg.epsilon_decay

    def train(self, agent: IAgent, env: IEnvironment, model_filename=None):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """

        trainer_cfg = self.cfg.get_current_exp_cfg().trainer_cfg

        reinforcement_learning_cfg = self.cfg.get_current_exp_cfg().reinforcement_learning_cfg

        scores_window = deque(maxlen=100)  # last 100 scores

        loss_window = deque(maxlen=100)
        pos_reward_ratio_window = deque(maxlen=100)
        neg_reward_ratio_window = deque(maxlen=100)

        # start with pre-trained model
        if (model_filename is not None):
            filename = self.select_model_filename(model_filename=model_filename)
            agent.current_model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
            agent.target_model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
            self.eps = 0.78

        epoch_recorder = Recorder(
            header=['epoch', 'avg_score', 'avg_val_score', 'epsilon', 'avg_loss', 'beta'],
            session_id=self.session_id,
            experiments_path=self.cfg.get_app_experiments_path(train_mode=True),
            model=None,
            log_prefix='epoch-',
            configuration=self.cfg.get_current_exp_cfg()
        )

        episode_recorder = Recorder(
            header=['step', 'episode', 'epoch', 'epoch step', 'epoch_episode', 'episode step', 'score', 'epsilon',
                    'beta',
                    'avg_pos_reward_ratio', 'avg_neg_reward_ratio', 'avg_loss'],
            session_id=self.session_id,
            experiments_path=self.cfg.get_app_experiments_path(train_mode=True),
            model=None,
            log_prefix='episode-',
            configuration=self.cfg.get_current_exp_cfg()
        )

        step = 0
        epoch = 0
        episode = 0

        while step < trainer_cfg.max_steps:

            epoch_step = 0

            ################################################################################
            # Training
            ################################################################################

            terminal = True
            epoch_episode = 0

            epoch_start_time = time.time()

            while (epoch_step < trainer_cfg.eval_frequency) and (step < trainer_cfg.max_steps):

                episode_start_time = time.time()

                for episode_step in range(trainer_cfg.max_episode_steps):

                    step_start_time = time.time()

                    if epoch_step >= trainer_cfg.eval_frequency:
                        break
                    elif step >= trainer_cfg.max_steps:
                        break

                    if terminal:
                        terminal = False

                        state, new_life = env.reset()

                        state = agent.pre_process(state)

                        score = 0
                        episode_score = np.zeros(self.cfg.get_current_exp_cfg().environment_cfg.num_agents)

                        epoch_episode += 1

                    action = agent.act(state, self.eps)

                    if new_life:
                        start_game_action = env.start_game_action()
                        action = start_game_action if start_game_action is not None else action

                    action += env.action_offset()

                    if trainer_cfg.human_flag:
                        env.render(mode='human')

                    next_state, reward, done, new_life = env.step(action)

                    next_state = agent.pre_process(next_state)

                    action -= env.action_offset()

                    pos_reward_ratio, neg_reward_ratio, loss, beta = agent.step(state, action, reward, next_state, done)

                    if loss is not None:
                        loss_window.append(loss)
                        pos_reward_ratio_window.append(pos_reward_ratio)
                        neg_reward_ratio_window.append(neg_reward_ratio)

                    step += 1
                    epoch_step += 1

                    state = next_state

                    episode_score += reward

                    if self.cfg.get_current_exp_cfg().environment_cfg.num_agents == 1:
                        if done:
                            break
                    else:
                        if np.any(done):
                            break

                    score = episode_score.mean(axis=0)

                    drl_logger.step(
                        "Train.",
                        extra={"params": {
                            "step": step,
                            "episode": episode,
                            "epoch": epoch,
                            "epoch_step": epoch_step,
                            "epoch_episode": epoch_episode,
                            "episode_step": episode_step,
                            "score": "{:.3f}".format(score),
                            "eps": "{:.3f}".format(self.eps),
                            "elapsed": "{:.3f}s".format(time.time() - step_start_time),
                        }})

                drl_logger.episode(
                    "Train.",
                    extra={"params": {
                        "step": step,
                        "episode": episode,
                        "epoch": epoch,
                        "epoch_step": epoch_step,
                        "epoch_episode": epoch_episode,
                        "episode_step": episode_step,
                        "score": "{:.3f}".format(score),
                        "eps": "{:.3f}".format(self.eps),
                        "elapsed": "{:.0f}s".format(time.time() - episode_start_time),
                    }})

                episode_recorder.record(
                    [step, episode, epoch, epoch_step, epoch_episode, episode_step, score, self.eps, beta,
                     np.mean(pos_reward_ratio_window) if len(pos_reward_ratio_window) > 0 else 0,
                     np.mean(neg_reward_ratio_window) if len(neg_reward_ratio_window) > 0 else 0,
                     np.mean(loss_window) if len(loss_window) > 0 else 0])

                episode += 1

                if step <= trainer_cfg.max_steps:
                    scores_window.append(score)  # save most recent score

                self.eps = max(self.eps_end, self.eps_decay * self.eps)  # decrease epsilon

                # sys.stdout.flush()

                episode_recorder.save()

                terminal = True

            ################################################################################
            # Validation
            ################################################################################

            val_step = 0

            val_scores_window = deque(maxlen=100)  # last 100 scores

            terminal = True
            epoch_val_episode = 0

            while val_step < trainer_cfg.eval_steps:

                episode_start_time = time.time()

                for episode_val_step in range(trainer_cfg.max_episode_steps):

                    step_start_time = time.time()

                    if val_step >= trainer_cfg.eval_steps:
                        break

                    if terminal:
                        terminal = False

                        state, new_life = env.reset()
                        state = agent.pre_process(state)
                        score = 0
                        episode_score = np.zeros(self.cfg.get_current_exp_cfg().environment_cfg.num_agents)
                        epoch_val_episode += 1

                    action = agent.act(state, eps=0)

                    if new_life:
                        start_game_action = env.start_game_action()
                        action = start_game_action if start_game_action is not None else action

                    action += env.action_offset()

                    if trainer_cfg.human_flag:
                        env.render(mode='human')

                    next_state, reward, done, new_life = env.step(action)

                    next_state = agent.pre_process(next_state)

                    val_step += 1

                    state = next_state

                    episode_score += reward

                    if self.cfg.get_current_exp_cfg().environment_cfg.num_agents == 1:
                        if done:
                            break
                    else:
                        if np.any(done):
                            break

                    score = episode_score.mean(axis=0)

                    drl_logger.step(
                        "Validate.",
                        extra={"params": {
                            "epoch": epoch,
                            "epoch_step": val_step,
                            "epoch_episode": epoch_val_episode,
                            "episode_step": episode_val_step,
                            "score": "{:.3f}".format(score),
                            "eps": "{:.3f}".format(self.eps),
                            "elapsed": "{:.3f}s".format(time.time() - step_start_time),
                        }})

                if val_step <= trainer_cfg.eval_steps:
                    val_scores_window.append(score)  # save most recent score

                drl_logger.episode(
                    "Validate.",
                    extra={"params": {
                        "epoch": epoch,
                        "epoch_step": val_step,
                        "epoch_episode": epoch_val_episode,
                        "episode_step": episode_val_step,
                        "score": "{:.3f}".format(score),
                        "eps": "{:.3f}".format(self.eps),
                        "elapsed": "{:.0f}s".format(time.time() - episode_start_time),
                    }})

                sys.stdout.flush()

                terminal = True

            drl_logger.epoch(
                "Epoch.",
                extra={"params": {
                    "epoch": epoch,
                    "mean score": np.mean(scores_window),
                    "mean val score": np.mean(val_scores_window),
                    "eps": "{:.3f}".format(self.eps),
                    "elapsed": "{:.0f}s".format(time.time() - epoch_start_time),
                }})

            epoch_recorder.record(
                [epoch, np.mean(scores_window), np.mean(val_scores_window), self.eps, np.mean(loss_window), beta])
            epoch_recorder.save()

            models = agent.get_models()

            for model in models:
                model_filename = self.get_model_filename(model.name, epoch, np.mean(scores_window), np.mean(val_scores_window), self.eps)
                torch.save(model.weights.state_dict(), model_filename)

            epoch += 1

        env.close()

        return scores_window
