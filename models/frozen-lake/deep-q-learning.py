import numpy as np
import gymnasium
from qnn import QNN
from torch.optim import AdamW
from torch.nn import MSELoss
import torch
from tqdm import tqdm


def neural_training_loop(
    env,
    model, 
    steps=100, 
    episodes=10000,
    learning_rate=1e-4,
    gamma=0.95, # needed for bellman.
    loss_fn=MSELoss()
):
    state, info = env.reset()
    print("The environment state: {}".format(info))

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    loss = 1000
    pbar = tqdm(range(episodes), desc="Training Loss")
    for episode in pbar:
        
        for step in range(steps):
            epsilon = max(0.01, 1.0 - episode / (episodes * 0.5))
            # Explore
            if np.random.random() < epsilon:
                action = np.random.randint(4)
            # Exploit
            else:
                with torch.no_grad():
                    action = torch.argmax(model(torch.tensor([state]))).item()

            new_state, reward, terminated, truncated, info = env.step(action)

            # 1) Zero out the grad ( or else you do grad accumulation )
            optimizer.zero_grad()

            # 2) forward pass
            # This will be a 4 dim vector representing actions
            Q_curr = model(torch.tensor([state])).squeeze(0) 
            with torch.no_grad():
                if terminated or truncated:
                    Q_target = torch.tensor(reward, dtype=torch.float32)
                else:
                    Q_target = reward + gamma * torch.max(model(torch.tensor([new_state])))

            # 3) loss
            # Bellman loss
            loss = loss_fn(Q_curr[action], Q_target)

            # 4) backward pass
            loss.backward()

            # 5) Update weights
            optimizer.step()

            # update state
            state = new_state

            # Show Progress
            pbar.set_postfix(loss=loss.item())

            if terminated or truncated:
                # After termination
                state, info = env.reset()
                break
    return model


if __name__ == "__main__":
    num_training_episodes = 1000  # Num of training episodes
    lr = 1e-4  # learning rate
    steps = 100  # num of steps
    gamma = 0.95 # Discount factor for the bellman equation 
    env_id = "FrozenLake-v1"

    model = QNN()

    # RL World
    # Frozen lake has 4 Actions
    # 16 observable space
    env = gymnasium.make(
        "FrozenLake-v1", 
        map_name="8x8", 
        is_slippery=False, 
        render_mode="rgb_array"
    )

    trained_QNN = neural_training_loop(
        env=env,
        model=model,
        steps=10000,
        episodes=num_training_episodes,
        learning_rate=lr,
        gamma=gamma
    )

    # Print final Q-table
    print("\n" + "=" * 50)
    print("FINAL Q-Network")
    print("=" * 50)
    print("Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP")
    print("-" * 50)
    for state in range(64):
        if state % 4 == 0:
            with torch.no_grad():
                q_values = trained_QNN(torch.tensor([state])).squeeze(0)

            best_action = torch.argmax(q_values).item()
            action_names = ["LEFT", "DOWN", "RIGHT", "UP"]
            print("State {:2d}: {} -> Best: {}".format(state, q_values.numpy(), action_names[best_action]))
