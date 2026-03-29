import os
import random

import ale_py
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam

from data import ReplayBuffer, Transition
from model import DeepQNetwork


gym.register_envs(ale_py)

NUM_FRAMES = 4
IMG_HEIGHT = 84
IMG_WIDTH = 84
ACTION_SPACE = 6

ENV_ID = "ALE/SpaceInvaders-v5"
EPISODES = 100000
BUFFER_CAPACITY = 50000
LEARNING_STARTS = 20000
PLOT_EVERY = 50
REWARD_SMOOTHING_WINDOW = 20
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.10
EPSILON_DECAY_STEPS = 7500000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 5000
LEARNING_RATE = 6.25e-5
SEED = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Eval Variables
EVAL_EVERY = 100
EVAL_EPISODES = 20 
EVAL_EPSILON = 0.01

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


def sample_rewards_every_n_episodes(rewards, every_n_episodes):
    sampled_episodes = []
    sampled_rewards = []

    for idx, reward in enumerate(rewards, start=1):
        if idx % every_n_episodes == 0:
            sampled_episodes.append(idx)
            sampled_rewards.append(reward)

    return sampled_episodes, sampled_rewards


def moving_average(values, window_size):
    if not values:
        return np.array([])

    window_size = min(window_size, len(values))
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return np.convolve(np.asarray(values, dtype=np.float32), kernel, mode="valid")


def save_training_plot(rewards, avg_losses, epsilon, output_path, eval_rewards=None):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    reward_ax, loss_ax = axes

    sampled_episodes, sampled_rewards = sample_rewards_every_n_episodes(rewards, PLOT_EVERY)
    reward_ax.plot(
        sampled_episodes,
        sampled_rewards,
        alpha=0.8,
        color="tab:blue",
        linewidth=1.2,
        label=f"Reward every {PLOT_EVERY} eps",
    )
    if sampled_rewards:
        smoothed_rewards = moving_average(sampled_rewards, REWARD_SMOOTHING_WINDOW)
        smoothed_episodes = sampled_episodes[len(sampled_episodes) - len(smoothed_rewards) :]
        reward_ax.plot(
            smoothed_episodes,
            smoothed_rewards,
            color="tab:orange",
            linewidth=2.0,
            label=f"Smoothed ({REWARD_SMOOTHING_WINDOW * PLOT_EVERY} eps MA)",
        )
    if eval_rewards:
        eval_episodes = [episode for episode, reward in eval_rewards]
        eval_values = [reward for episode, reward in eval_rewards]
        reward_ax.plot(
            eval_episodes,
            eval_values,
            color="tab:green",
            linewidth=1.5,
            marker="o",
            markersize=3,
            label=f"Eval reward ({EVAL_EPISODES} eps @ eps={EVAL_EPSILON:.2f})",
        )
    reward_ax.set_ylabel("Reward")
    reward_ax.legend(loc="upper left")

    latest_eval = f"{eval_rewards[-1][1]:.0f}" if eval_rewards else "n/a"

    if sampled_rewards:
        reward_ax.set_title(
            (
                "Episode {episode} | reward at ep {sampled_episode}: {sampled_reward:.0f} | "
                "max {max_reward:.0f} | eval {eval_reward} | epsilon {epsilon:.3f}"
            ).format(
                episode=len(rewards),
                sampled_episode=sampled_episodes[-1],
                sampled_reward=sampled_rewards[-1],
                max_reward=max(rewards),
                eval_reward=latest_eval,
                epsilon=epsilon,
            )
        )
    else:
        reward_ax.set_title(
            "Episode {episode} | waiting for episode {plot_every} | max {max_reward:.0f} | eval {eval_reward} | epsilon {epsilon:.3f}".format(
                episode=len(rewards),
                plot_every=PLOT_EVERY,
                max_reward=max(rewards),
                eval_reward=latest_eval,
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


def evaluate_policy(env, model, eval_episodes, epsilon):
    # boolean flag
    was_training = model.training

    # Inference mode
    # disables batch and dropout
    model.eval()

    episode_rewards = []
    try:
        for _ in range(eval_episodes):
            # Reset the environment to it's starting state.
            state, info = env.reset()

            terminated = False
            truncated = False

            eval_episode_reward = 0.0
            
            while not terminated and not truncated:

                # we want to take a random action
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else: # rely on the trained_policy
                    # Distable gradient computation
                    # saves memory and speeds inference.
                    with torch.no_grad():
                        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE) / 255.0
                        predicted_q_values = model(state_tensor.unsqueeze(0))

                    # model will predict 6 q values. 
                    # pick the one with highest == best action
                    action = predicted_q_values.argmax(dim=1).item()

                state, reward, terminated, truncated, info = env.step(action)
                eval_episode_reward += reward

            episode_rewards.append(eval_episode_reward)

    # restore model training.
    finally:
        if was_training: model.train()

    return np.mean(episode_rewards)


def main():
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE.startswith("cuda"):
        torch.cuda.manual_seed_all(SEED)

    env = make_env(ENV_ID)
    eval_env = make_env(ENV_ID)
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
    optimizer = Adam(params=predictor_model.parameters(), lr=LEARNING_RATE)

    step_counter = 0
    episode_rewards = []
    eval_rewards = []
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
                clipped_reward = float(np.sign(reward))

                done = terminated or truncated
                replay_buffer.push(
                    Transition(
                        torch.from_numpy(state).to(device=DEVICE, dtype=torch.float32),
                        action,
                        clipped_reward,
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

            eval_reward = None
            if (episode + 1 ) % EVAL_EVERY == 0:
                # We get average eval reward)
                eval_reward = evaluate_policy(
                    env=eval_env,
                    model=predictor_model,
                    eval_episodes=EVAL_EPISODES,
                    epsilon=EVAL_EPSILON
                )
                eval_rewards.append((episode + 1, eval_reward))

            if (episode + 1) % PLOT_EVERY == 0 and episode_rewards:
                save_training_plot(
                    rewards=episode_rewards,
                    avg_losses=episode_avg_losses,
                    epsilon=epsilon,
                    output_path=PLOT_PATH,
                    eval_rewards=eval_rewards,
                )
                torch.save(predictor_model.state_dict(), CHECKPOINT_PATH)

                print(
                    "episode={episode} reward={reward:.2f} epsilon={epsilon:.3f} saved={plot_path}".format(
                        episode=episode + 1,
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
                eval_rewards=eval_rewards,
            )
        torch.save(predictor_model.state_dict(), CHECKPOINT_PATH)
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
