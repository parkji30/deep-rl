import numpy as np
import gymnasium
from qnn import QNN
from torch.optim import AdamW

def create_q_table(state_space: int, action_space: int):
    # Qtable = np.zeros((state_space, action_space))
    Qtable = np.random.uniform(low=0, high=1, size=(state_space, action_space))
    return Qtable


def greedy_policy(q_table, state):
    # policies are the brain of the RL agent.
    # The tell us what action to take.

    # our Qtable is a matix of
    # row: state
    # column: action
    action = np.argmax(q_table[state][:])
    # return the index of the best action to take.
    return action


def epsilon_greedy_policy(q_table, state, epsilon, env):
    # This function is used to create the q_table
    # following the off-policy method (Different policies from train/inference)

    random_num = np.random.uniform(0, 1)

    # if random_num > episilon > exploitation
    if random_num > epsilon:
        action = greedy_policy(q_table, state)

    # otherwise exploration
    else:
        action = env.action_space.sample()

    return action

def neural_training_loop(
    env,
    model, 
    steps=10000, 
    learning_rate=1e-4,
):

    state, info = env.info()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for step in steps:
        initial_action = np.randint(4)

        new_state, reward, terminated_info, truncated_info = env.step(initial_action)

        actions = model()


def training_loop(
    env,
    num_training_episode: int,
    learning_rate: float,
    gamma: float,
    min_epsilon: float,
    max_epsilon: float,
    decay_rate: float,
    max_steps: int,
    Qtable,
):
    # Track successes for calculating success rate
    successes = 0

    for episode in range(num_training_episode):
        # Reduce epsilon ( we need less and less exploration )
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        # reset env
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False

        for step in range(max_steps):
            # choose an action
            action = epsilon_greedy_policy(Qtable, state, epsilon, env=env)

            # Take Action at Observation Rt+1, st+1
            # Take the action (a) and observe the outcome state(s) and reward (r)
            new_state, reward, terminated, truncated, info = env.step(action)

            # Update Q(s, a) := Q(s, a) +  lr [R(s,a) + gamma * max Q(s', a') - Q(s, a)]
            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )

            if terminated or truncated:
                break

            state = new_state

        # Track if we reached the goal (reward=1)
        if reward == 1.0:
            successes += 1

        # Print progress every 1000 episodes
        if (episode + 1) % 1000 == 0:
            success_rate = successes / (episode + 1) * 100
            non_zero_q = np.count_nonzero(Qtable)
            max_q = np.max(Qtable)
            print(f"\n=== Episode {episode + 1}/{num_training_episode} ===")
            print(f"Epsilon: {epsilon:.4f}")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Non-zero Q-values: {non_zero_q}/64")
            print(f"Max Q-value: {max_q:.4f}")
            print(f"Q-table for state 0 (start): {Qtable[0]}")

    return Qtable


if __name__ == "__main__":
    n_training_episodes = int(500000)  # Num of training episodes
    lr = 0.9  # learning rate
    n_evals = 100  # num of testing episodes

    env_id = "FrozenLake-v1"
    max_steps = 99  # max steps per episodes
    gamma = 0.95
    eval_seed = []

    # Exploration parameters
    max_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 5e-5

    # RL World
    # Frozen lake has 4 Actions
    # 16 observable space
    env = gymnasium.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="rgb_array")

    q_table_frozenlake = create_q_table(state_space=env.observation_space.n, action_space=env.action_space.n)
    print(q_table_frozenlake.shape)

    trained_Q_table = training_loop(
        env=env,
        num_training_episode=n_training_episodes,
        learning_rate=lr,
        gamma=gamma,
        min_epsilon=min_epsilon,
        max_epsilon=max_epsilon,
        decay_rate=decay_rate,
        max_steps=max_steps,
        Qtable=q_table_frozenlake,
    )

    # Print final Q-table
    print("\n" + "=" * 50)
    print("FINAL Q-TABLE")
    print("=" * 50)
    print("Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP")
    print("-" * 50)
    for state in range(16):
        best_action = np.argmax(trained_Q_table[state])
        action_names = ["LEFT", "DOWN", "RIGHT", "UP"]
        print(f"State {state:2d}: {trained_Q_table[state]} -> Best: {action_names[best_action]}")
