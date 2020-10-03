import time

import torch
import matplotlib.pyplot as plt
import numpy as np

from drl import drl_logger
from drl.env.i_environment import IEnvironment
from drl.experiment.configuration import Configuration
from drl.experiment.recorder import Recorder
from drl.image import preprocess_image


class Player:
    def __init__(self, env: IEnvironment, agent, config: Configuration, session_id, path_models='models'):
        self.__env = env
        self.__agent = agent
        self.__config = config
        self.__session_id = session_id
        self.__path_models = path_models

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__env.close()

    def play(self, trained, mode, is_rgb, model_filename, num_episodes, num_steps):

        if is_rgb:
            return self.play_rgb(num_episodes=num_episodes, num_steps=num_steps, trained=trained, mode=mode, model_filename=model_filename)
        else:
            return self.play_classic(num_episodes=num_episodes, num_steps=num_steps, trained=trained, mode=mode, model_filename=model_filename)

    def play_classic( self, num_episodes=3, score_max=True, score_med=False, trained=True, mode='rgb_array',
                      model_filename=None,
                      num_steps=None):

        recorder = Recorder(header=['episode', 'step', 'action', 'reward', 'reward_total'],
                            session_id=self.__session_id,
                            experiments_path=self.__config.get_app_experiments_path(train_mode=False),
                            model=None,
                            configuration=self.__config.get_current_exp_cfg())

        if trained or (model_filename is not None):
            if self.__config.get_current_exp_cfg().reinforcement_learning_cfg.algorithm_type.startswith('dqn'):
                self.__agent.current_model.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))
            elif self.__config.get_current_exp_cfg().reinforcement_learning_cfg.algorithm_type.startswith('ddpg'):
                self.__agent.actor_current_model.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))

        play_episode = 0
        play_step = 0
        scores = []

        for i in range(num_episodes):
            reward_total = 0
            play_episode_step = 0
            state, new_life = self.__env.reset()
            self.__env.render(mode=mode)

            play_episode = i

            episode_start_time = time.time()

            if num_steps is None:

                terminal = True

                while True:

                    step_start_time = time.time()

                    if terminal:
                        terminal = False

                        state, new_life = self.__env.reset()

                        state = self.__agent.pre_process(state)
                        episode_score = np.zeros(self.__config.get_current_exp_cfg().environment_cfg.num_agents)

                    if trained:
                        action = self.__agent.act(state)
                    else:
                        action = self.__agent.act(state, eps=1.)

                    if new_life:
                        start_game_action = self.__env.start_game_action()
                        action = start_game_action if start_game_action is not None else action

                    action += self.__env.action_offset()

                    self.__env.render(mode=mode)

                    state, reward, done, info = self.__env.step(action)

                    action -= self.__env.action_offset()

                    episode_score += reward

                    score = episode_score.mean(axis=0)


                    # recorder.record([i, play_episode_step, action, reward, reward_total])


                    drl_logger.step(
                        "Play.",
                        extra={"params": {
                            "step": play_step,
                            "episode": play_episode,
                            "episode_step": play_episode_step,
                            "score": "{:.3f}".format(score),
                            "elapsed": "{:.3f}s".format(time.time() - step_start_time),
                        }})

                    if self.__config.get_current_exp_cfg().environment_cfg.num_agents == 1:
                        if done:
                            break
                    else:
                        if np.any(done):
                            break

                    play_episode_step += 1
                    play_step += 1

                drl_logger.episode(
                    "Play.",
                    extra={"params": {
                        "episode": play_episode,
                        "score": "{:.3f}".format(score),
                        "elapsed": "{:.3f}s".format(time.time() - episode_start_time),
                    }})

            else:
                for j in range(num_steps):
                    if trained:
                        action = self.__agent.act(state)
                    else:
                        action = self.__agent.act(state, eps=1.)

                    self.__env.render(mode=mode)
                    state, reward, done, _ = self.__env.step(action)
                    reward_total += reward
                    play_episode_step += 1

                    recorder.record([i, play_episode_step, action, reward, reward_total])

                    if done:
                        break

            scores.append([i, score, play_episode_step])

            recorder.save()

        np_scr = np.array(scores)
        mean = np_scr[:, 1].mean()

        return scores, mean

    def play_rgb(self, num_episodes=3, score_max=True, score_med=False, trained=True, mode='rgb_array', model_filename=None, num_steps=None):

        if trained or (model_filename is not None):
            filename = self.select_model_filename(score_max, score_med, model_filename=model_filename)
            self.__agent.current_model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

        for i in range(num_episodes):
            scores = []
            reward_total = 0
            steps_total = 0
            state, new_life = self.__env.reset()
            image = self.__env.render(mode=mode)

            image2 = preprocess_image(state)

            if num_steps is None:
                while True:
                    action = self.__agent.act(image2)
                    self.__env.render(mode=mode)
                    state, reward, done, _ = self.__env.step(action)
                    reward_total += reward
                    steps_total += 1

                    if done:
                        break
            else:
                for j in range(num_steps):
                    action = self.__agent.act(image2)
                    self.__env.render(mode=mode)
                    state, reward, done, _ = self.__env.step(action)
                    reward_total += reward
                    steps_total += j

                    if done:
                        break
            scores.append([[i, reward_total, steps_total]])

        return scores
