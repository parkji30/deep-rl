# deep-rl

Learning RL through games. 

The end game goal is to train a RL model on Starcraft Broodwar.

# Space Invaders

I actually decided to write this by hand because I wanted to learn DQL RL. I gotta say, not relying on AI (except for conceptual questions) really gets the knowledge the stick.
- I'm currently on Vanilla DQN. It's so bad. I can't get it to converge. (Mar 9, 2026)
- <img width="541" height="190" alt="image" src="https://github.com/user-attachments/assets/8790fdef-69fa-4911-8103-309ab82d47ce" />

- Okay update, turns out my model was learning. I was just plotting the wrong metric (loss). This is useless because Q-Values are always moving targets. The ground truth is always changing. It's better to plot reward per episode (Temporal Difference in our case, i.e. one step at a time). Here's the reward plot below.
