"""Train and evaluate four comparable word-level autoregressive RNN models.

Example:
    python train_text_generation.py --epochs 20 --output-dir results
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, GRU, LSTM, SimpleRNN


DATASET_URL = "https://www.gutenberg.org/files/11/11-0.txt"
MODEL_NAMES = ("SimpleRNN", "BidirectionalRNN", "LSTM", "GRU")


def download_dataset(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = requests.get(DATASET_URL, timeout=60)
        response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def tokenize(text: str) -> list[str]:
    """Keep words and punctuation as tokens so generated text remains readable."""
    text = text.lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[.!?,;:]", text)


def prepare_sequences(
    tokens: list[str], sequence_length: int, max_vocabulary: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int],
    dict[int, str],
]:
    counts = pd.Series(tokens).value_counts()
    vocabulary = ["<pad>", "<unk>"] + [
        token for token in counts.index if token not in {"<pad>", "<unk>"}
    ][: max_vocabulary - 2]
    word_to_id = {word: i for i, word in enumerate(vocabulary)}
    ids = [word_to_id.get(token, word_to_id["<unk>"]) for token in tokens]
    sequences = np.asarray(
        [ids[i - sequence_length : i + 1] for i in range(sequence_length, len(ids))],
        dtype=np.int32,
    )
    np.random.default_rng(42).shuffle(sequences)
    split = int(len(sequences) * 0.9)
    x_train, x_test = sequences[:split, :-1], sequences[split:, :-1]
    y_train, y_test = sequences[:split, -1], sequences[split:, -1]
    return x_train, x_test, y_train, y_test, word_to_id, dict(enumerate(vocabulary))


def build_model(name: str, vocabulary_size: int, sequence_length: int, embedding_dim: int, units: int) -> tf.keras.Model:
    recurrent = {
        "SimpleRNN": SimpleRNN(units),
        "BidirectionalRNN": Bidirectional(SimpleRNN(units)),
        "LSTM": LSTM(units),
        "GRU": GRU(units),
    }[name]
    model = Sequential(
        [
            Embedding(vocabulary_size, embedding_dim, input_length=sequence_length),
            recurrent,
            Dense(vocabulary_size, activation="softmax"),
        ],
        name=name,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


def detokenize(words: list[str]) -> str:
    text = " ".join(words)
    return re.sub(r"\s+([.!?,;:])", r"\1", text).capitalize()


def sample_text(
    model: tf.keras.Model,
    seed: str,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    sequence_length: int,
    temperature: float,
    number_of_words: int = 60,
) -> str:
    words = tokenize(seed)
    for _ in range(number_of_words):
        context = [word_to_id.get(word, word_to_id["<unk>"]) for word in words[-sequence_length:]]
        context = [word_to_id["<pad>"]] * (sequence_length - len(context)) + context
        probabilities = model.predict(np.asarray([context]), verbose=0)[0].astype(np.float64)
        probabilities[0] = 0.0
        probabilities = np.log(np.maximum(probabilities, 1e-12)) / temperature
        probabilities = np.exp(probabilities - np.max(probabilities))
        probabilities /= probabilities.sum()
        next_id = np.random.choice(len(probabilities), p=probabilities)
        words.append(id_to_word[int(next_id)])
    return detokenize(words)


def plot_history(histories: dict[str, tf.keras.callbacks.History], output: Path) -> None:
    plt.figure(figsize=(10, 6))
    for name, history in histories.items():
        plt.plot(history.history["val_loss"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Validation loss by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def run(args: argparse.Namespace) -> None:
    tf.keras.utils.set_random_seed(42)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = download_dataset(Path(args.dataset_path))
    tokens = tokenize(raw)
    x_train, x_test, y_train, y_test, word_to_id, id_to_word = prepare_sequences(
        tokens, args.sequence_length, args.max_vocabulary
    )
    (output / "dataset_summary.json").write_text(
        json.dumps(
            {
                "dataset": "Alice's Adventures in Wonderland",
                "source": DATASET_URL,
                "token_count": len(tokens),
                "vocabulary_size": len(word_to_id),
                "sequence_length": args.sequence_length,
                "training_sequences": len(x_train),
                "test_sequences": len(x_test),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "models").mkdir(exist_ok=True)
    histories: dict[str, tf.keras.callbacks.History] = {}
    rows: list[dict[str, float | str]] = []
    samples: list[dict[str, str | float]] = []
    seeds = ["alice was beginning to get very tired", "the king said"]
    for name in MODEL_NAMES:
        model = build_model(
            name, len(word_to_id), args.sequence_length, args.embedding_dim, args.units
        )
        start = time.perf_counter()
        history = model.fit(
            x_train,
            y_train,
            validation_split=0.1,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        )
        training_seconds = time.perf_counter() - start
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        rows.append(
            {
                "model": name,
                "test_loss": float(test_loss),
                "perplexity": float(np.exp(np.clip(test_loss, -10, 10))),
                "test_accuracy": float(test_accuracy),
                "training_time_seconds": training_seconds,
                "parameters": model.count_params(),
            }
        )
        model.save(output / "models" / f"{name}.keras")
        histories[name] = history
        for temperature in (0.5, 1.0, 1.5):
            for seed in seeds:
                samples.append(
                    {
                        "model": name,
                        "seed": seed,
                        "temperature": temperature,
                        "generated_text": sample_text(
                            model, seed, word_to_id, id_to_word, args.sequence_length, temperature
                        ),
                    }
                )
    pd.DataFrame(rows).sort_values("perplexity").to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(samples).to_csv(output / "generated_samples.csv", index=False)
    plot_history(histories, output / "validation_loss.png")
    print(pd.DataFrame(rows).sort_values("perplexity").to_string(index=False))
    print(f"Saved results to {output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--units", type=int, default=128)
    parser.add_argument("--max-vocabulary", type=int, default=10000)
    parser.add_argument("--dataset-path", default="data/alice.txt")
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
