import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity=10000, frame_height=84, frame_width=84, stack_size=4):
        self.capacity = capacity
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.stack_size = stack_size

        self.frames = np.empty((capacity, frame_height, frame_width), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.has_transition = np.zeros(capacity, dtype=np.bool_)
        self.episode_ids = np.full(capacity, -1, dtype=np.int64)
        self.frame_numbers = np.full(capacity, -1, dtype=np.int64)

        self.size = 0
        self.write_idx = 0
        self.transition_count = 0
        self.total_frames_seen = 0
        self.current_episode_id = -1
        self.current_state_idx = None

    def __len__(self):
        return self.transition_count

    def start_episode(self, observation):
        self.current_episode_id += 1

        for frame in self._extract_frames(observation):
            self.current_state_idx = self._store_frame(frame=frame, episode_id=self.current_episode_id)

    def push(self, action, reward, next_observation, done):
        if self.current_state_idx is None:
            raise RuntimeError("Call start_episode() before pushing transitions.")

        transition_idx = self.current_state_idx
        if self.has_transition[transition_idx]:
            raise RuntimeError("Current replay slot already has transition metadata.")

        self.actions[transition_idx] = action
        self.rewards[transition_idx] = reward
        self.dones[transition_idx] = done
        self.has_transition[transition_idx] = True
        self.transition_count += 1

        next_frame = self._extract_frames(next_observation)[-1]
        self.current_state_idx = self._store_frame(
            frame=next_frame,
            episode_id=self.current_episode_id,
        )

    def sample(self, batch_size, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        sample_indices = self._sample_indices(batch_size=batch_size)

        states = np.stack([self._stack_state(idx) for idx in sample_indices], axis=0)
        new_states = np.stack(
            [self._stack_state(self._offset_index(idx, 1)) for idx in sample_indices],
            axis=0,
        )

        actions = self.actions[sample_indices]
        rewards = self.rewards[sample_indices]
        dones = self.dones[sample_indices].astype(np.float32)

        states = torch.as_tensor(states, device=device, dtype=torch.float32) / 255.0
        new_states = torch.as_tensor(new_states, device=device, dtype=torch.float32) / 255.0
        actions = torch.as_tensor(actions, device=device, dtype=torch.long)
        rewards = torch.as_tensor(rewards, device=device, dtype=torch.float32)
        dones = torch.as_tensor(dones, device=device, dtype=torch.float32)
        return states, actions, rewards, new_states, dones

    def _extract_frames(self, observation):
        frames = np.asarray(observation, dtype=np.uint8)

        if frames.ndim != 3:
            raise ValueError(
                "Expected stacked Atari observation with 3 dimensions, "
                f"got shape {frames.shape}."
            )

        if frames.shape == (self.stack_size, self.frame_height, self.frame_width):
            return frames

        if frames.shape == (self.frame_height, self.frame_width, self.stack_size):
            return np.transpose(frames, (2, 0, 1))

        raise ValueError(
            "Unexpected Atari observation shape "
            f"{frames.shape}; expected either "
            f"({self.stack_size}, {self.frame_height}, {self.frame_width}) or "
            f"({self.frame_height}, {self.frame_width}, {self.stack_size})."
        )

    def _store_frame(self, frame, episode_id):
        slot_idx = self.write_idx

        if self.has_transition[slot_idx]:
            self.transition_count -= 1

        self.frames[slot_idx] = frame
        self.actions[slot_idx] = 0
        self.rewards[slot_idx] = 0.0
        self.dones[slot_idx] = False
        self.has_transition[slot_idx] = False
        self.episode_ids[slot_idx] = episode_id
        self.frame_numbers[slot_idx] = self.total_frames_seen

        self.total_frames_seen += 1
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return slot_idx

    def _sample_indices(self, batch_size):
        if self.transition_count < batch_size:
            raise ValueError("Not enough transitions in replay buffer to sample a batch.")

        candidate_upper = self.size if self.size < self.capacity else self.capacity
        sampled_indices = []
        seen = set()
        attempts = 0
        max_attempts = max(batch_size * 100, self.capacity)

        while len(sampled_indices) < batch_size:
            if attempts >= max_attempts:
                raise RuntimeError("Unable to sample a valid Atari replay batch.")

            candidate_idx = np.random.randint(0, candidate_upper)
            attempts += 1

            if candidate_idx in seen:
                continue

            if not self._is_valid_transition_index(candidate_idx):
                continue

            seen.add(candidate_idx)
            sampled_indices.append(candidate_idx)

        return np.asarray(sampled_indices, dtype=np.int64)

    def _is_valid_transition_index(self, idx):
        if not self.has_transition[idx]:
            return False

        if self.size < self.stack_size + 1:
            return False

        window_indices = [
            self._offset_index(idx, offset)
            for offset in range(-(self.stack_size - 1), 2)
        ]

        frame_numbers = self.frame_numbers[window_indices]
        episode_ids = self.episode_ids[window_indices]

        if np.any(frame_numbers < 0):
            return False

        expected_numbers = frame_numbers[0] + np.arange(len(window_indices))
        if not np.array_equal(frame_numbers, expected_numbers):
            return False

        if not np.all(episode_ids == episode_ids[0]):
            return False

        return True

    def _stack_state(self, end_idx):
        state_indices = [
            self._offset_index(end_idx, offset)
            for offset in range(-(self.stack_size - 1), 1)
        ]
        return self.frames[state_indices]

    def _offset_index(self, idx, offset):
        return (idx + offset) % self.capacity
