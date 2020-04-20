[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Project 2: Continuous Control

## Introduction

This project implements an agent to control
simulated 2-linked arms to track given targets. The implementation
is based on the DDPG algorithm from the paper [**Continuous control with deep reinforcement learning**](https://arxiv.org/pdf/1509.02971.pdf)
by Lillicrap, et. al.. 
A trained agent, which continuously control many 2-linked arms to track simultaneously many green objects, is demonstrated below.

![Trained Agent][image1]

The environment chosen for the project is a **modified version of the Reacher Environment** 
from the Unity ML-Agents toolkit. The original version can be found [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher),
and the version we will work with consists of a custom build provided by Udacity 
with the following descriptions:

* A group of 20 agents consisting of 2-link arms with 2 torque-actuated joints. The
  objective of all agents is to reach a goal, defined by a green sphere, and track
  it with the end-effector of the arm, defined by a small blue sphere at the end of
  the last link.

* Each agent receives an **observation** consisting of a 33-dimensional vector
  with measurements like relative position and orientations of the links, relative
  position of the goal and its speed, etc

* Each agent moves its arm around by applying **actions** consisting of 4 torques
  applied to each of the 2 actuated joints (2 torques per joint).

* Each agent gets a **reward** of +0.1 each step its end effector is within the limits
  of the goal. The environment is considered solved once the agent gets an average
  reward of +30 over 100 episodes.

* The task is **episodic** with a maximum of 1000 steps per episode.

## Installations

### Downloading the environment

Download the environment from one of the links below. You need only select the environment that matches your operating system (this project uses 20 agents version of Reacher environment):

Platform | Link
-------- | -----
Linux             | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
Mac OSX           | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
Windows (32-bit)  | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
Windows (64-bit)  | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

Unzip (or decompress) the downloaded file and store the path of the executable as we will need the path to input on `Continuous_Control.ipynb`. 

Note that this environment is compatible only with an older version of the ML-Agents toolkit (version 0.4.0), the next setup section will take care of this.

### Resolving dependencies

Please follow the instructions on [DRLND Github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install the required libraries.

Additionally, please also install `torchsummary` to visualize the model descriptions. Open your Anaconda environment that you just create (`dlrnd`) and type:

```bash
pip install torchsummary
```

## Usages

Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent or load the trained model and visualize the agents tracking the green objects!

## References

These are some code-references and papers I used while implementing my DDPG agent:

* [*Continuous control through deep reinforcement learning* paper by Lillicrap et. al.](https://arxiv.org/pdf/1509.02971.pdf)
* [*Deterministic Policy Gradients Algorithms* paper by Silver et. al.](http://proceedings.mlr.press/v32/silver14.pdf)
* [DDPG implementation from Udacity](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)