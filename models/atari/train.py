import os
import random

import ale_py
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW

from data import ReplayBuffer, Transition
from model import DeepQNetwork


gym.register_envs(ale_py)

NUM_FRAMES = 4
IMG_HEIGHT = 84
IMG_WIDTH = 84
ACTION_SPACE = 6

ENV_ID = "ALE/SpaceInvaders-v5"
EPISODES = 10000
BUFFER_CAPACITY = 30000
LEARNING_STARTS = 5000
REWARD_MA_WINDOW = 50
PLOT_EVERY = 20
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 100000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 5000
LEARNING_RATE = 1e-5
SEED = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "runs")
PLOT_PATH = os.path.join(OUTPUT_DIR, "script_training_progress.png")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "predictor_model.pt")


def make_env(env_id):
    env = gym.make(env_id, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env=env,
        screen_size=IMG_HEIGHT,
        grayscale_obs=True,
        frame_skip=4,
        terminal_on_life_loss=False,
        scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=NUM_FRAMES)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def train_step(replay_buffer, batch_size, optimizer, predictor_model, target_model, gamma, loss_func):
    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    predicted_q_values = predictor_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN 
        next_actions = predictor_model(next_states).argmax(dim=1)
        next_q_values = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # Vanilla DQN
        # next_q_values = target_model(next_states).max(dim=1).values


    target_q_values = rewards + gamma * next_q_values * (1 - dones)
    loss = loss_func(predicted_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def huber_loss(pred, target, delta=1.0):
    diff = target - pred
    abs_diff = torch.abs(diff)
    quadratic = 0.5 * diff**2
    linear = delta * (abs_diff - 0.5 * delta)
    loss = torch.where(abs_diff <= delta, quadratic, linear)
    return loss.mean()


def moving_average(values, window):
    averages = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_values = values[start : idx + 1]
        averages.append(sum(window_values) / len(window_values))
    return averages


def save_training_plot(rewards, avg_losses, epsilon, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    reward_ax, loss_ax = axes

    reward_ma = moving_average(rewards, REWARD_MA_WINDOW)
    reward_ax.plot(rewards, alpha=0.35, color="tab:blue", linewidth=0.8, label="Reward")
    reward_ax.plot(reward_ma, color="tab:orange", linewidth=2.0, label=f"{REWARD_MA_WINDOW}-ep MA")
    reward_ax.set_ylabel("Reward")
    reward_ax.legend(loc="upper left")
    reward_ax.set_title(
        "Episode {episode} | reward {reward:.0f} | max {max_reward:.0f} | epsilon {epsilon:.3f}".format(
            episode=len(rewards) - 1,
            reward=rewards[-1],
            max_reward=max(rewards),
            epsilon=epsilon,
        )
    )

    loss_ax.plot(avg_losses, alpha=0.8, color="tab:red", linewidth=0.8)
    loss_ax.set_xlabel("Episode")
    loss_ax.set_ylabel("Avg Loss")
    loss_ax.set_title("Average training loss per episode")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE.startswith("cuda"):
        torch.cuda.manual_seed_all(SEED)

    env = make_env(ENV_ID)
    env.reset(seed=SEED)

    predictor_model = DeepQNetwork(
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        action_space=ACTION_SPACE,
        num_frames=NUM_FRAMES,
    ).to(DEVICE)

    target_model = DeepQNetwork(
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        action_space=ACTION_SPACE,
        num_frames=NUM_FRAMES,
    ).to(DEVICE)

    target_model.load_state_dict(predictor_model.state_dict())
    target_model.eval()

    replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)
    optimizer = AdamW(params=predictor_model.parameters(), lr=LEARNING_RATE)

    step_counter = 0
    episode_rewards = []
    episode_avg_losses = []

    try:
        for episode in range(EPISODES):
            state, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_losses = []

            while not terminated and not truncated:
                state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE) / 255.0
                progress = min(step_counter / EPSILON_DECAY_STEPS, 1.0)
                epsilon = EPSILON_START + progress * (EPSILON_END - EPSILON_START)

                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        predicted_q_values = predictor_model(state_t.unsqueeze(0))
                    action = predicted_q_values.argmax(dim=1).item()

                next_state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                done = terminated or truncated
                replay_buffer.push(
                    Transition(
                        torch.from_numpy(state).to(device=DEVICE, dtype=torch.float32),
                        action,
                        reward,
                        torch.from_numpy(next_state).to(device=DEVICE, dtype=torch.float32),
                        done,
                    )
                )

                if step_counter >= LEARNING_STARTS:
                    loss_value = train_step(
                        replay_buffer=replay_buffer,
                        batch_size=BATCH_SIZE,
                        optimizer=optimizer,
                        predictor_model=predictor_model,
                        target_model=target_model,
                        gamma=GAMMA,
                        loss_func=huber_loss,
                    )
                    if loss_value is not None:
                        episode_losses.append(loss_value)

                state = next_state
                step_counter += 1

                if step_counter % TARGET_UPDATE_FREQ == 0:
                    target_model.load_state_dict(predictor_model.state_dict())

            episode_rewards.append(episode_reward)
            if episode_losses:
                episode_avg_losses.append(sum(episode_losses) / len(episode_losses))
            else:
                episode_avg_losses.append(float("nan"))

            if episode % PLOT_EVERY == 0 and episode_rewards:
                save_training_plot(
                    rewards=episode_rewards,
                    avg_losses=episode_avg_losses,
                    epsilon=epsilon,
                    output_path=PLOT_PATH,
                )
                torch.save(predictor_model.state_dict(), CHECKPOINT_PATH)

                print(
                    "episode={episode} reward={reward:.2f} epsilon={epsilon:.3f} saved={plot_path}".format(
                        episode=episode,
                        reward=episode_reward,
                        epsilon=epsilon,
                        plot_path=PLOT_PATH,
                    ),
                    flush=True,
                )
    finally:
        if episode_rewards:
            final_progress = min(step_counter / EPSILON_DECAY_STEPS, 1.0)
            final_epsilon = EPSILON_START + final_progress * (EPSILON_END - EPSILON_START)
            save_training_plot(
                rewards=episode_rewards,
                avg_losses=episode_avg_losses,
                epsilon=final_epsilon,
                output_path=PLOT_PATH,
            )
        torch.save(predictor_model.state_dict(), CHECKPOINT_PATH)
        env.close()


if __name__ == "__main__":
    main()
