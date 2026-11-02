# Predicting AI success question by question (PAIEC 2026)

Code for our entry to the Predictive AI Evaluation Competition (NeurIPS 2026) and for every
number in our technical report. Competition submission ID: 1016835.

The model is built in four steps, each a folder with one `model.py`:

| Folder | Report name |
| --- | --- |
| `my_submission/` | First submission (smoothed average starting at 33%) |
| `my_model/` | Step 1: smoothed average |
| `my_model2/` | Step 2: + shared questions |
| `my_model3/` | Step 3: + shared benchmark |
| `my_model5/` | **Step 4: + language-model guess (final submission)** |
| `my_model4/` | Not kept: + track record |
| `my_model6/` | Not kept: + longer trust |
| `dev/baselines/s0_smoothed_033_k5/` | First submission, as scored in the main table |

## Setup

Python 3.12.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Download the public data (measurement-db at the revision pinned in
`data/benchmark_inventory.json`, about 5.7 GB, saved to `data/measurement-db/`).
The dataset is gated: first accept its terms at
https://huggingface.co/datasets/aims-foundations/measurement-db while logged in to
Hugging Face, then log in on your machine with an access token:

```
.venv/bin/hf auth login
.venv/bin/python dev/download_dataset.py
```

The first scoring run builds `data/cache/groups.pkl` (about 1 GB); later runs reuse it.

Run every command from the repository root.

## Reproduce the report

The language model's replies used in the report are in `data/cache/claude_replies.jsonl`
(reply text, stop reason and token counts; no keys). The simulator reads replies only from
this file and never calls the API, so the commands below need no API key and cost nothing.

Main table and Figure 1 (30 platform-like check rounds):
```
.venv/bin/python dev/final_eval.py --draws 30 --seed 1 --pool 5 --max-group-items 150 --tag all --models "constant 0.5" "empirical mean (organisers)" "S0 smoothed (p0 0.33, k 5)" "S1 smoothed (p0 0.5, k 2)" "S2 S1 + same-item siblings" "S3 S2 + benchmark siblings" "S4 S3 + public track record" "S4b S3 + Claude prior" "S4c S4b retuned"
```

Step 1 on mixed check rounds:
```
.venv/bin/python dev/final_eval.py --draws 100 --seed 1 --models "constant 0.5" "empirical mean (organisers)" "S0 smoothed (p0 0.33, k 5)" "S1 smoothed (p0 0.5, k 2)" "S2 S1 + same-item siblings"
```

Step 2 (two round settings, pool 5 and pool 6):
```
.venv/bin/python dev/final_eval.py --draws 100 --seed 1 --pool 5 --max-group-items 250 --models "S1 smoothed (p0 0.5, k 2)" "S2 S1 + same-item siblings"
.venv/bin/python dev/final_eval.py --draws 100 --seed 1 --pool 6 --max-group-items 250 --models "S1 smoothed (p0 0.5, k 2)" "S2 S1 + same-item siblings"
```

Step 3 (same two settings):
```
.venv/bin/python dev/final_eval.py --draws 100 --seed 1 --pool 5 --max-group-items 250 --models "S2 S1 + same-item siblings" "S3 S2 + benchmark siblings" --reference "S2 S1 + same-item siblings" --tag S2vsS3
.venv/bin/python dev/final_eval.py --draws 100 --seed 1 --pool 6 --max-group-items 250 --models "S2 S1 + same-item siblings" "S3 S2 + benchmark siblings" --reference "S2 S1 + same-item siblings" --tag S2vsS3
```

Step 4 and the "longer trust" variant on the third set of rounds (seed 2):
```
.venv/bin/python dev/final_eval.py --draws 30 --seed 2 --pool 5 --max-group-items 150 --models "S3 S2 + benchmark siblings" "S4b S3 + Claude prior" "S4c S4b retuned" --reference "S4b S3 + Claude prior" --tag S3S4bS4c
```

Haiku 5.5 vs Sonnet 5.5:
```
.venv/bin/python dev/final_eval.py --draws 30 --seed 1 --pool 5 --max-group-items 150 --models "S3 S2 + benchmark siblings" "S4b S3 + Claude prior" "S4b Haiku, last line" "S4b Sonnet 5.5, last line" --reference "S4b Haiku, last line" --tag sonnet
```

Claude models vs OpenAI's GPT-5 nano, and how each model's raw guesses compare with the real
pass rates:
```
.venv/bin/python dev/final_eval.py --draws 30 --seed 1 --pool 5 --max-group-items 150 --models "S3 S2 + benchmark siblings" "S4b S3 + Claude prior" "S4b Haiku, last line" "S4b Sonnet 5.5, last line" "S4b GPT-5 nano, last line" --reference "S4b Haiku, last line" --tag nano
.venv/bin/python dev/compare_llm_guesses.py --draws 30 --seed 1 --pool 5 --max-group-items 150
```

Tune a setting on the tuning rounds (seed 0):
```
.venv/bin/python dev/tune_sibling.py --folder <model folder> --constant <name> --values ...
```

Seed 0 gives the tuning rounds, seed 1 the check rounds, seed 2 the third set of rounds.
`final_eval.py` prints the table and writes `results/final_eval_<settings>.csv`. Our own
outputs are already in `results/`, so after a rerun `git diff results/` shows any difference.
The `.log` files there hold the printed tables of our runs.

The coin-flip and running-average rows of Table 1 use the organizers' code, which is not in
this repository. To score them, copy `sample_code_submission/` and
`paiec_baseline/empirical_mean/` from the competition starting kit into the repository root;
otherwise `final_eval.py` skips those two rows.

### Analyses behind single numbers in the report

| Report claim | Script |
| --- | --- |
| About 85% of the error comes from differences between questions; empirical Bayes gives no gain | `dev/eb_potential.py` |
| Same-question correlation between subjects (0.41, Step 2) | `dev/same_item_corr.py` |
| Benchmark difficulty shared between subjects (0.43 and 0.65, Step 3) | `dev/sibling_rate_corr.py` |
| Question text does not predict difficulty within a benchmark | `dev/item_difficulty_check.py`, `dev/feature_signal_check.py`, `dev/feature_columns_check.py` |
| Public track record (dead end) | `dev/rates_table.py`, `dev/build_track_record.py`, `dev/strength_check.py` |
| First language-model test (200 questions) | `dev/llm_prior_test.py`, `dev/llm_prior_analysis.py` (prompt in `dev/my_prompt.py`) |
| Step 1 tuning with the first simulator | `dev/tune_smoothed.py`, `dev/simulate_alc.py` (reads `measurement-db/` at the repository root, not `data/measurement-db/`) |
| Data download and checks | `dev/download_dataset.py`, `dev/inspect_dataset.py`, `dev/list_hf_repo.py` |

## Collect new language-model replies (paid)

Needs an Anthropic API key in `ANTHROPIC_API_KEY`. Only questions missing from the cache are
sent. `--max-dollars` stops the run at that spend.

```
.venv/bin/python dev/prefetch_llm.py --draws 30 --seed 1 --pool 5 --max-group-items 150 --max-dollars 5
```

Add `--model claude-sonnet-5-5` for the Sonnet comparison (about $45 for the 8,506 questions).
Add `--model gpt-5-nano --flex` for the OpenAI comparison; it needs `OPENAI_API_KEY` and costs
about $1 at list price (`--flex` asks for OpenAI's cheaper, slower tier).
New calls can give slightly different replies from ours.

## Build the competition zip

Copy `my_model5/submission_config.example.json` to `my_model5/submission_config.json`, put
your own key in it, then run `.venv/bin/python dev/package_s4b.py`. The zip contains your key:
do not share it. The packaging script validates the zip with the organizers'
`paiec_baseline/check_submission_zip.py` when that file is present.

## License

CC BY 4.0 (see `LICENSE`), except `data/benchmark_inventory.json`, which is derived from
measurement-db and stays under its license, CC BY-SA 4.0.
