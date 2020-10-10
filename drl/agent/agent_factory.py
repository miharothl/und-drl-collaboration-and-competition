from drl.agent.ddpg_agent import DdpgAgent
from drl.agent.dqn_agent import DqnAgent
from drl.agent.i_agent import IAgent
from drl.agent.maddpg_agent import MaddpgAgent
from drl.experiment.configuration import Configuration


class AgentFactory:

    def __init__(self):
        pass

    @staticmethod
    def create(config: Configuration, seed=0) -> IAgent:

        algorithm_type = config.get_current_exp_cfg().reinforcement_learning_cfg.algorithm_type

        if algorithm_type.startswith('dqn'):
            agent = DqnAgent(seed=seed, cfg=config)
        elif algorithm_type.startswith('ddpg'):
            agent = DdpgAgent(seed=seed, cfg=config)
        elif algorithm_type.startswith('maddpg'):
            agent = MaddpgAgent(seed=seed, cfg=config)
        else:
            raise Exception("Agent for algorighm '{}' type not supported".format(algorithm_type))

        return agent
