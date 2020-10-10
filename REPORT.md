[//]: # (Image References)


[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"
[image3]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

# Goal

The projects aim is to train two agents that control rackets and bounce a ball over the a net. If an agent hits the ball over the net,
it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.
The goal of each agent is to keep the ball in play.

![Trained Agent][image1]

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives
its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5
(over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

# Approach

I started with the DDPG [3] algorithm that I developed for Continuous Control[1] project. I adjusted the configuration
to use 2 agents and I was able to solve the environment by achieving a score 0.5+ over 100 consecutive episodes.

I tired to improve the learning process by implementing:
- PER [5]
- MADDPG [6] (in progress on branch: try-maddpg)

The steps that I followed to solve this environment:

1. Evaluate the state and action space of the environment
2. Establish a baseline using a random action policy
3. Implement the learning algorithm
4. Run experiments and select the best agent

## 1. Evaluate State & Action Space of the Environment

The action-space has 2 continuous dimensions corresponding to the, corresponding to movement toward (or away from) the net, and jumping. 
The state-space is continuous; it has 8 dimensions corresponding to the position and velocity of the ball and racket. Each agent receives
its own, local observation.

## 2. Establish Baseline Using Random Action Policy

Before starting the deep reinforcement learning process, its good to understand the environment. Controlling the 
rackets with an agent where actions have randomly selected achieve scores averaging 0.01 over 100 consecutive episodes.
 
## 3. Implement Learning Algorithm

The
[agent](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/agent/ddpg_agent.py)
and 
[environment](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/env/unity_multiple_env.py)
are created according to the provided
[configuration](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/experiment/configuration.py)
.
[Recorder](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/experiment/recorder.py)
records the experiment and store the results for later
[analysis](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/rlab-analysis.ipynb)
.

The agent interacts with the environment in the
[training loop](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/experiment/train/master_trainer.py)
.
In the exploration phase (higher *Epsilon*) of the training
agent's actions are mostly random, created using 
[Ornstein-Uhlenbeck noise generator](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/agent/tools/ou_noise.py)
. Actions, environment states, dones, and rewards tuples, are stored in the experience
replay buffer. The *Buffer Size* parameter determines the size of the buffer.

DDPG [3] is using 
[actor and critic](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/model/ddpg_model.py)
neural networks. Both have current, and target model with identical architecture used to stabilize the DDPG learning process.
During the learning process, weights of the target network are fixed (or updated more slowly based on parameter *Tau*).

The loss function is defined as the mean square error of critic's temporal difference error, the difference between the expected
and estimated q values. Adam optimizer minimizes the loss function performing the gradient descent and backpropagation algorithm
using the specified *Learning Rate*.

Learning is performed *Num Updates* times on every *Update Every* steps, when *Batch Size* of actions, states, dones and rewards tuples are
sampled from the
[replay buffer](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/agent/tools/replay_buffer.py) [2]
either randomly or in case of prioritized experience replay, based on their importance,
determined by the temporal difference error. Prioritized experience replay requires
[segment trees](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/agent/tools/segment_tree.py) [2]
.

During the exploitation phase of the training (lower *Epsilon*) the noise added to the actions is proportionally scaled down (*epsilon end*)
and mostly based on the estimated policies calculated by the current actor neural network.

## 4. Run Experiments and Select Best Agent

[Training](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/rlab-collaboration-and-competition.ipynb)
is done using the epochs, consisting of training episodes where epsilon greedy agent is used,
and validation episodes using only actions predicted by the trained agent.
 
I used the following training hyper-parameters:

|Hyper Parameter            |Value                    |
|:---                       |:---                     |
|Max Steps                  |150000                   |
|Max Episode Steps          |1000                     |
|Evaluation Frequency       |10000  (max 10 episodes) |
|Evaluation Steps           |2000   (max 2 episodes)  |
|Epsilon Start              |1.5 (rounded to 1.       |
|Epsilon End                |0.1                      |
|Epsilon Decay              |0.998                    |
|Actor Hidden Layers Units  |[256, 128]               |
|Critic Hidden Layers Units |[256, 128]               |
|Gamma                      |0.99                     |
|Tau                        |0.001                    |
|Learning Rate Actor        |0.0001                   |
|Learning Rate Critic       |0.0003                   |
|Update Every               |2                        |
|Num Updates                |4                        |
|Replay Buffer Size         |100000                   |
|Batch Size                 |128                      |
|Use Prioritized Replay     |True                     |
|Prioritized Replay Alpha   |0.6                      |
|Prioritized Replay Beta0   |0.4                      |
|Prioritized Replay eps     |1e-06                    |

The first version of an agent that can solve the environment with scores 0.5+ is obtained in over 100 episodes is 
trained in the Epoch 3 after playing 602 episodes.

![Training Score][image4]
![Training Epsilon][image5]

Using PER didn't improve the training process.

```
2020-10-10 09:58:18,668 - drl - EPISODE - Train. - {'step': 39341, 'episode': 600, 'epoch': 3, 'epoch_step': 9341, 'epoch_episode': 20, 'episode_step': 449, 'score': '1.150', 'eps': '0.451', 'elapsed': '50s'}
2020-10-10 09:58:43,526 - drl - EPISODE - Train. - {'step': 39564, 'episode': 601, 'epoch': 3, 'epoch_step': 9564, 'epoch_episode': 21, 'episode_step': 222, 'score': '0.550', 'eps': '0.450', 'elapsed': '25s'}
2020-10-10 09:59:31,814 - drl - EPISODE - Train. - {'step': 40000, 'episode': 602, 'epoch': 3, 'epoch_step': 10000, 'epoch_episode': 22, 'episode_step': 436, 'score': '1.100', 'eps': '0.449', 'elapsed': '48s'}
2020-10-10 10:00:12,679 - drl - EPISODE - Validate. - {'epoch': 3, 'epoch_step': 411, 'epoch_episode': 1, 'episode_step': 410, 'score': '1.100', 'eps': '0.449', 'elapsed': '41s'}
2020-10-10 10:01:52,111 - drl - EPISODE - Validate. - {'epoch': 3, 'epoch_step': 1411, 'epoch_episode': 2, 'episode_step': 999, 'score': '2.600', 'eps': '0.449', 'elapsed': '99s'}
2020-10-10 10:01:52,525 - drl - EPISODE - Validate. - {'epoch': 3, 'epoch_step': 1415, 'epoch_episode': 3, 'episode_step': 3, 'score': '0.000', 'eps': '0.449', 'elapsed': '0s'}
2020-10-10 10:02:50,690 - drl - EPISODE - Validate. - {'epoch': 3, 'epoch_step': 2000, 'epoch_episode': 4, 'episode_step': 585, 'score': '1.500', 'eps': '0.449', 'elapsed': '58s'}
2020-10-10 10:02:50,691 - drl - EPOCH - Epoch. - {'epoch': 3, 'mean score': 0.5895000087842345, 'mean val score': 1.3000000193715096, 'eps': '0.449', 'elapsed': '1305s'}
```

The best agent is trained in Epoch 6 after playing 674 episodes and can achieve a score **2.5** over 100 consecutive episodes.

```

Average score over 100 episodes is 2.5
```

# Future Work

Deep reinforcement learning is a fascinating and exciting topic. I'll continue to improve my reinforcement learning
laboratory by applying
 * MADDPG to the Soccer environment.

![Soccer][image3]

# References
  - [1] [Continuous Control](https://github.com/miharothl/DRLND-Continuous-Control)
  - [2] [Open AI Baselines](https://github.com/openai/baselines)
  - [3] [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)
  - [4] [Understanding Prioritized Experience Replay](https://danieltakeshi.github.io/2019/07/14/per/)
  - [5] [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
  - [6] [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)
