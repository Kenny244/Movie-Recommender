from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class MovieLensTransitionDataset(Dataset):
    """Build logged one-step transitions from chronologically ordered ratings."""

    def __init__(
        self,
        ratings: pd.DataFrame,
        movie_features: pd.DataFrame,
        max_history: int = 50,
    ) -> None:
        required_ratings = {"userId", "movieId", "rating", "timestamp"}
        required_movies = {"movieId", "title"}
        missing_ratings = required_ratings.difference(ratings.columns)
        missing_movies = required_movies.difference(movie_features.columns)
        if missing_ratings or missing_movies:
            raise ValueError(
                f"Missing columns: ratings={sorted(missing_ratings)}, "
                f"movie_features={sorted(missing_movies)}"
            )
        if max_history < 1:
            raise ValueError("max_history must be at least 1")

        self.max_history = max_history
        self.movie_features = movie_features.reset_index(drop=True).copy()
        self.movie_ids = self.movie_features["movieId"].tolist()
        self.item_to_index = {
            movie_id: index for index, movie_id in enumerate(self.movie_ids)
        }

        self.sequences: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.transitions: list[tuple[int, int]] = []
        ordered = ratings.sort_values(["userId", "timestamp", "movieId"])

        for _, user_ratings in ordered.groupby("userId", sort=False):
            known = user_ratings[
                user_ratings["movieId"].isin(self.item_to_index)
            ]
            if len(known) < 2:
                continue

            item_indices = np.asarray(
                [self.item_to_index[item] for item in known["movieId"]],
                dtype=np.int64,
            )
            user_rewards = np.clip(
                (known["rating"].to_numpy(dtype=np.float32) - 3.0) / 2.0,
                -1.0,
                1.0,
            )
            sequence_index = len(self.sequences)
            self.sequences.append(item_indices)
            self.rewards.append(user_rewards)
            self.transitions.extend(
                (sequence_index, step)
                for step in range(1, len(item_indices))
            )

        if not self.transitions:
            raise ValueError("No transitions found; each user needs 2+ known ratings")

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        sequence_index, step = self.transitions[index]
        sequence = self.sequences[sequence_index]
        start = max(0, step - self.max_history)
        history = sequence[start:step]
        next_history = sequence[max(0, step + 1 - self.max_history) : step + 1]

        state = np.zeros(self.max_history, dtype=np.int64)
        next_state = np.zeros(self.max_history, dtype=np.int64)
        state[: len(history)] = history + 1
        next_state[: len(next_history)] = next_history + 1

        seen_items = np.zeros(len(self.movie_ids), dtype=np.bool_)
        seen_items[sequence[:step]] = True
        next_seen_items = seen_items.copy()
        next_seen_items[sequence[step]] = True

        terminal = step == len(sequence) - 1
        return (
            torch.from_numpy(state),
            torch.tensor(len(history), dtype=torch.long),
            torch.from_numpy(seen_items),
            torch.tensor(sequence[step], dtype=torch.long),
            torch.tensor(self.rewards[sequence_index][step], dtype=torch.float32),
            torch.from_numpy(next_state),
            torch.tensor(len(next_history), dtype=torch.long),
            torch.from_numpy(next_seen_items),
            torch.tensor(terminal, dtype=torch.float32),
        )


class GRUCQLRecommender(nn.Module):
    """GRU state encoder with factorized item Q-values for offline CQL."""

    def __init__(
        self,
        num_movies: int,
        genre_features: np.ndarray,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if genre_features.ndim != 2 or genre_features.shape[0] != num_movies:
            raise ValueError("genre_features must have shape (num_movies, num_genres)")

        num_genres = genre_features.shape[1]
        self.num_movies = num_movies
        self.movie_embedding = nn.Embedding(
            num_movies + 1, embedding_dim, padding_idx=0
        )
        self.genre_projection = nn.Linear(
            num_genres, embedding_dim, bias=False
        ) if num_genres else None
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, batch_first=True
        )
        self.state_projection = nn.Linear(hidden_dim, embedding_dim)
        self.item_bias = nn.Parameter(torch.zeros(num_movies))

        padded_genres = np.zeros((num_movies + 1, num_genres), dtype=np.float32)
        if num_genres:
            padded_genres[1:] = genre_features.astype(np.float32, copy=False)
        self.register_buffer(
            "genre_features", torch.from_numpy(padded_genres)
        )

    def encode_state(
        self, history_ids: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        history_vectors = self.movie_embedding(history_ids)
        if self.genre_projection is not None:
            history_vectors = history_vectors + self.genre_projection(
                self.genre_features[history_ids]
            )
        encoded, _ = self.gru(history_vectors)
        last_positions = (lengths - 1).view(-1, 1, 1)
        state = encoded.gather(
            1, last_positions.expand(-1, 1, encoded.shape[-1])
        ).squeeze(1)
        return self.state_projection(state)

    def all_item_vectors(self) -> torch.Tensor:
        item_vectors = self.movie_embedding.weight[1:]
        if self.genre_projection is not None:
            item_vectors = item_vectors + self.genre_projection(
                self.genre_features[1:]
            )
        return item_vectors

    def forward(
        self, history_ids: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        state = self.encode_state(history_ids, lengths)
        return (
            state @ self.all_item_vectors().transpose(0, 1)
            / np.sqrt(self.movie_embedding.embedding_dim)
            + self.item_bias
        )


def train_cql(
    model: GRUCQLRecommender,
    dataset: Dataset,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    discount: float = 0.95,
    conservative_weight: float = 0.1,
    target_tau: float = 0.01,
    device: str | torch.device | None = None,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Fit a conservative offline Q-function to logged MovieLens transitions."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    torch.manual_seed(seed)
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(selected_device)
    target_model = deepcopy(model).to(selected_device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    training_history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in loader:
            (
                states,
                lengths,
                seen_items,
                actions,
                rewards,
                next_states,
                next_lengths,
                next_seen_items,
                dones,
            ) = (
                tensor.to(selected_device) for tensor in batch
            )
            q_values = model(states, lengths)
            logged_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = target_model(next_states, next_lengths)
                next_q_values = next_q_values.masked_fill(
                    next_seen_items, -torch.inf
                )
                next_q = next_q_values.max(dim=1).values
                next_q = torch.where(dones.bool(), torch.zeros_like(next_q), next_q)
                targets = rewards + discount * next_q

            bellman_loss = F.smooth_l1_loss(logged_q, targets)
            valid_q_values = q_values.masked_fill(seen_items, -torch.inf)
            conservative_loss = (
                torch.logsumexp(valid_q_values, dim=1) - logged_q
            ).mean()
            loss = bellman_loss + conservative_weight * conservative_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                for target_parameter, parameter in zip(
                    target_model.parameters(), model.parameters()
                ):
                    target_parameter.lerp_(parameter, target_tau)

            total_loss += float(loss.detach())
            total_batches += 1

        training_history.append(
            {"epoch": float(epoch + 1), "loss": total_loss / total_batches}
        )

    model.eval()
    return training_history


def recommend_topk(
    model: GRUCQLRecommender,
    history_movie_ids: list[Any],
    movie_features: pd.DataFrame,
    n: int = 10,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Rank unseen catalog items for a user from their observed history."""
    if n < 1:
        raise ValueError("n must be positive")
    if not history_movie_ids:
        return movie_features.iloc[0:0][["movieId", "title"]].copy()

    item_to_index = {
        movie_id: index for index, movie_id in enumerate(movie_features["movieId"])
    }
    history_indices = [
        item_to_index[item]
        for item in history_movie_ids
        if item in item_to_index
    ][-50:]
    if not history_indices:
        return movie_features.iloc[0:0][["movieId", "title"]].copy()

    selected_device = torch.device(device or next(model.parameters()).device)
    history = torch.zeros((1, 50), dtype=torch.long, device=selected_device)
    shifted_indices = torch.tensor(
        [index + 1 for index in history_indices],
        dtype=torch.long,
        device=selected_device,
    )
    history[0, : len(history_indices)] = shifted_indices
    lengths = torch.tensor([len(history_indices)], dtype=torch.long, device=selected_device)

    with torch.inference_mode():
        scores = model(history, lengths)[0]
    scores = scores.clone()

    seen = {item_to_index[item] for item in history_movie_ids if item in item_to_index}
    if seen:
        scores[list(seen)] = -torch.inf
    count = min(n, int(torch.isfinite(scores).sum()))
    if count == 0:
        return movie_features.iloc[0:0][["movieId", "title"]].copy()

    recommended_indices = torch.topk(scores, k=count).indices.cpu().tolist()
    return movie_features.iloc[recommended_indices][["movieId", "title"]].reset_index(
        drop=True
    )
