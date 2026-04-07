import argparse
import os
import random

import numpy as np
import torch

NUM_FRAMES = 4
IMG_HEIGHT = 84
IMG_WIDTH = 84
DEFAULT_ENV_ID = "ALE/SpaceInvaders-v5"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "runs", "predictor_model.pt")
DEFAULT_VIDEO_DIR = os.path.join(SCRIPT_DIR, "runs", "eval")


def make_env(gym, env_id, video_dir):
    env = gym.make(env_id, frameskip=1, render_mode="rgb_array")
    env = gym.wrappers.AtariPreprocessing(
        env=env,
        screen_size=IMG_HEIGHT,
        grayscale_obs=True,
        frame_skip=4,
        terminal_on_life_loss=False,
        scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=NUM_FRAMES)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: True,
        name_prefix="eval",
    )
    return env


def select_action(model, state, epsilon, env, device):
    if random.random() < epsilon:
        return env.action_space.sample()

    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device) / 255.0
        predicted_q_values = model(state_tensor.unsqueeze(0))
    return predicted_q_values.argmax(dim=1).item()


def main():
    parser = argparse.ArgumentParser(description="Run one Atari eval episode and save it as an mp4.")
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import ale_py
    import gymnasium as gym
    from model import DeepQNetwork

    gym.register_envs(ale_py)

    os.makedirs(args.video_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    env = make_env(gym=gym, env_id=args.env_id, video_dir=args.video_dir)
    model = DeepQNetwork(
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        action_space=env.action_space.n,
        num_frames=NUM_FRAMES,
    ).to(args.device)

    state_dict = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()

    episode_reward = 0.0
    terminated = False
    truncated = False
    state, info = env.reset(seed=args.seed)

    try:
        while not terminated and not truncated:
            action = select_action(
                model=model,
                state=state,
                epsilon=args.epsilon,
                env=env,
                device=args.device,
            )
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
    finally:
        env.close()

    mp4_files = sorted(
        file_name for file_name in os.listdir(args.video_dir) if file_name.endswith(".mp4")
    )
    saved_video = os.path.join(args.video_dir, mp4_files[-1]) if mp4_files else args.video_dir

    print(f"episode_reward={episode_reward:.2f}")
    print(f"saved_video={saved_video}")


if __name__ == "__main__":
    main()
