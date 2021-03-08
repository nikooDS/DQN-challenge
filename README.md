# DQN-challenge
## What I learned in this challenge
- There are two type of enviroments

    - Episodic
        - List of states s, actions u, rewards r, and of course new states s'
    - Continuous
        - No terminal State

- Two ways of learning

    - Monte Carlo Approach

        - Collecting rewards after the end of episode

        - Calculate max expected future reward
        - gets better by iteration
            - V(st) <- V(st) + α(R(t) - V(st))
            - max expected future reward starting from this state <-- former estimation + learning rate * (return - estimation of reward)
            - Problem with this approach: we calculate rewards at the end of every episode, we average all actions, even if some bad actions took place this will result in averaging them as good actions if the end result (as per episode) was good.

    - Temporal Difference

        - Estimate the reward at each step, gets better each step
        - V(st) <- V(st) + α(R(t+1) + γV(S[t+1]) - V(st))


- In this project, 

    - Use a Tensorflow model using both convolutional and dense layers
    - Learn how to connect our agent to OpenAI's Gym interface
    - Implement a memory class to allow the agent to use past experience during training in experience replay
    - Implement a loss function
    - Implement a training loop

## Running the experiment
- For running this experiment just need to run the DQNdemo.py