import torch
from typing import Tuple

from drl.experiment.configuration import Configuration
from drl.model.ddpg_model import ActorPendulum, CriticPendulum
from drl.model.model_dqn import DqnDueling2Hidden, Dqn2Hidden, Dqn3Hidden, Dqn4Hidden
from drl.model.model_dqn_rgb import QNetwork2a


class ModelFactory:

    def __init__(self):
        pass

    @staticmethod
    def create(seed, device, cfg: Configuration) -> Tuple[torch.nn.Module, torch.nn.Module]:

        if cfg.get_current_exp_cfg().reinforcement_learning_cfg.algorithm_type.startswith("dqn"):

            if cfg.get_current_exp_cfg().agent_cfg.state_rgb is True:
                network_type = 'rgb'
            else:
                network_type = 'classic'

            dueling = False
            if cfg.get_current_exp_cfg().reinforcement_learning_cfg.algorithm_type == 'dqn_dueling':
                dueling = True

            fc_units = cfg.get_current_exp_cfg().reinforcement_learning_cfg.dqn_cfg.model_cfg.hidden_layers
            num_frames = cfg.get_current_exp_cfg().agent_cfg.num_frames
            state_size = cfg.get_current_exp_cfg().agent_cfg.state_size
            action_size = cfg.get_current_exp_cfg().agent_cfg.action_size
            seed = seed
            device = device

            if network_type == "classic":

                if len(fc_units) == 2:

                    if dueling:
                        current_model = DqnDueling2Hidden(
                            state_size * num_frames, action_size, seed,
                            fc1_units=fc_units[0],
                            fc2_units=fc_units[1]
                        ).to(device)
                        target_model = DqnDueling2Hidden(
                            state_size * num_frames, action_size, seed,
                            fc1_units=fc_units[0],
                            fc2_units=fc_units[1]
                        ).to(device)
                    else:
                        current_model = Dqn2Hidden(
                            state_size * num_frames, action_size, seed,
                            fc1_units=fc_units[0],
                            fc2_units=fc_units[1]
                        ).to(device)
                        target_model = Dqn2Hidden(
                            state_size * num_frames, action_size, seed,
                            fc1_units=fc_units[0],
                            fc2_units=fc_units[1]
                        ).to(device)
                elif len(fc_units) == 3:
                    current_model = Dqn3Hidden(
                        state_size * num_frames, action_size, seed,
                        fc1_units=fc_units[0],
                        fc2_units=fc_units[1],
                        fc3_units=fc_units[2]).to(device)

                    target_model = Dqn3Hidden(
                        state_size * num_frames, action_size, seed,
                        fc1_units=fc_units[0],
                        fc2_units=fc_units[1],
                        fc3_units=fc_units[2]).to(device)
                elif len(fc_units) == 4:
                    current_model = Dqn4Hidden(
                        state_size * num_frames, action_size, seed,
                        fc1_units=fc_units[0],
                        fc2_units=fc_units[1],
                        fc3_units=fc_units[2],
                        fc4_units=fc_units[3]).to(device)

                    target_model = Dqn4Hidden(
                        state_size * num_frames, action_size, seed,
                        fc1_units=fc_units[0],
                        fc2_units=fc_units[1],
                        fc3_units=fc_units[2],
                        fc4_units=fc_units[3]).to(device)

                return current_model, target_model

            if network_type == 'rgb':
                current_model = QNetwork2a(state_size[0], state_size[1], num_frames, action_size, seed).to(device)
                target_model = QNetwork2a(state_size[0], state_size[1], num_frames, action_size, seed).to(device)

                return current_model, target_model


        if cfg.get_current_exp_cfg().reinforcement_learning_cfg.algorithm_type.startswith("ddpg"):

            fc_units_actor = cfg.get_current_exp_cfg().reinforcement_learning_cfg.ddpg_cfg.actor_model_cfg.hidden_layers
            fc_units_critic = cfg.get_current_exp_cfg().reinforcement_learning_cfg.ddpg_cfg.critic_model_cfg.hidden_layers
            state_size = cfg.get_current_exp_cfg().agent_cfg.state_size
            action_size = cfg.get_current_exp_cfg().agent_cfg.action_size
            seed = seed
            device = device

            # Actor Network (w/ Target Network)
            actor_local = ActorPendulum(
                state_size, action_size, seed,
                fc1_units=fc_units_actor[0],
                fc2_units=fc_units_actor[1]).to(device)
            actor_target = ActorPendulum(
                state_size, action_size, seed,
                fc1_units=fc_units_actor[0],
                fc2_units=fc_units_actor[1]).to(device)

            # Critic Network (w/ Target Network)
            critic_local = CriticPendulum(
                state_size, action_size, seed,
                fcs1_units=fc_units_critic[0],
                fc2_units=fc_units_critic[1]).to(device)
            critic_target = CriticPendulum(
                state_size, action_size, seed,
                fcs1_units=fc_units_critic[0],
                fc2_units=fc_units_critic[1]).to(device)

            return actor_local, actor_target, critic_local, critic_target

