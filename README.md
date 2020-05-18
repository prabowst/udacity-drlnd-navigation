# Udacity DRLND - Navigation

This project entails a submisison for Udacity's Deep Reinforcement Learning Nanodegree Program: Project 1 - Navigation.

The project aims to train an agent in navigating a large square world, collecting yellow bananas and avoiding purple bananas.

### Table of Contents 

1. [Project Description](#description)
2. [Requirements](#requirements)
3. [Files](#files)
4. [Project Results](#results)
5. [Licensing and Acknowledgements](#licensing)

### Project Description<a name="description"></a>

This project entails training an agent to navigate and collect bananas in a large square world. The agent is given rewards as follows:

1. +1 for collecting yellow bananas
2. -1 for collecting purple bananas

The agent is expected to obtain as many rewards as possible.  

The state space has 37 dimensions which contain the agent's velocity and ray-based perceptions of objects surrounding the agent's forward direction.

The agent can navigate through the world using the following four discrete actions:

1. 0 - move forward
2. 1 - move backward
3. 2 - turn left
4. 3 - turn right

The agent is considered to solve the environment once it attains an **average score of at least 13.0 over the past 100 consecutive episodes**.

### Requirements<a name="requirements"></a>

There are several requirements in order to run this project. 

- Configure Python3.6 and PyTorch environment as described in the [Udacity repository](https://github.com/udacity/deep-reinforcement-learning#dependencies).
- Install the Unity environment following the requirement [steps](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md).
- Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.
- Place the environment in the preferred path.
