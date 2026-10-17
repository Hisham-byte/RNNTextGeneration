# Autoregressive Text Generation: RNN, BiRNN, LSTM and GRU

This project implements ICT4403 Foundations of Generative AI Assignment 2. It
trains four word-level next-token predictors under the same preprocessing,
sequence length, optimizer, batch size, and epoch settings.

## Objective

Given a fixed sequence of previous words, predict the next word and repeatedly
feed predictions back into the model to generate text. The models are Simple
RNN, Bidirectional RNN, LSTM, and GRU.

## Dataset

The selected dataset is *Alice's Adventures in Wonderland* by Lewis Carroll,
downloaded from [Project Gutenberg](https://www.gutenberg.org/files/11/11-0.txt).
It is a small English literary narrative containing dialogue, punctuation,
descriptive prose, and recurring characters. The script lowercases the text,
tokenizes words and punctuation, maps rare/out-of-vocabulary tokens to
`<unk>`, and creates overlapping windows of 20 input tokens plus one target.
Ninety percent of windows are used for training and ten percent for testing.

## Installation and execution

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_text_generation.py --epochs 20 --output-dir results
```

The command creates `data/alice.txt`, four saved `.keras` models,
`results/metrics.csv`, `results/generated_samples.csv`, `results/dataset_summary.json`,
and `results/validation_loss.png`. For a quick smoke run use
`--epochs 1 --units 32 --embedding-dim 32`.

## Evaluation and analysis

`metrics.csv` reports test cross-entropy loss, perplexity (`exp(loss)`),
next-word accuracy, parameter count, and training time. The completed run
produced the following results:

| Model | Test loss | Perplexity | Test accuracy | Training time |
|---|---:|---:|---:|---:|
| GRU | 4.7548 | 116.1440 | 18.92% | 74.32 s |
| SimpleRNN | 4.8733 | 130.7514 | 18.20% | 31.07 s |
| LSTM | 4.9211 | 137.1574 | 17.91% | 101.61 s |
| Bidirectional RNN | 5.0459 | 155.3812 | 16.66% | 42.69 s |

The generated-samples file contains the same two seed sequences at
temperatures 0.5, 1.0, and 1.5. The observed qualitative behaviour was:

* Lower loss and perplexity, and higher accuracy, indicate better prediction.
* At temperature 0.5, the samples were the most coherent and contextually
  consistent, but conservative and repetitive.
* At temperature 1.0, the samples balanced local coherence and vocabulary
  diversity most effectively.
* At temperature 1.5, the samples were more diverse but became less
  grammatically consistent and included more unrelated word transitions.
* The GRU samples generally maintained the best balance between recognizable
  literary context and diversity. All models showed some repetition, which is
  expected for a small corpus and a short training run.

The GRU is the best overall model for this run: it achieved the lowest test
loss and perplexity and the highest next-word accuracy. SimpleRNN was fastest,
but its predictive metrics were weaker. LSTM required the most training time
without outperforming GRU on this dataset. The Bidirectional RNN performed
worst on the autoregressive test split; although bidirectional layers can use
both directions during encoding, generation still proceeds left-to-right.

Compare samples for coherence, diversity, contextual consistency, and
repeated words or phrases. Select the model using both quantitative and
qualitative evidence, not training loss alone.

Do not invent numerical results: paste the values produced by your run into the
submission report and explain the observed temperature effects.

## Repository deliverables

Include this source file, `requirements.txt`, the generated `results/` files,
and the final report PDF/docx prepared with your student details. The report
should link to this repository and the dataset URL above.
