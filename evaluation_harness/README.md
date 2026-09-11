# Evaluation Harness

This folder contains the complete evaluation suite for the Apple Support Agent. It satisfies the assignment requirement:
> "Evaluation harness — automated metrics + an LLM-as-judge rubric for reply quality, including evidence of how well your judge agrees with a human."

## Files in this folder

- **`automated_metrics.py`**: The main evaluation harness. It runs the entire agent pipeline (intent classification, retrieval, reply generation, escalation decision) against all 242 human-labeled examples in the test set. It then computes automated metrics (like intent accuracy) and compares them to the Classical ML Baselines.
- **`classical_ml_baseline.py`**: A standalone ML baseline script. It trains a `TF-IDF + Logistic Regression` model on the massive 76k unsupervised dataset and evaluates it on the 242-example test set to establish a rigorous baseline for the LLM agent to beat.
- **`llm_as_judge.py`**: The LLM-as-a-judge implementation. It scores drafted replies on a 1-5 scale across 4 dimensions (Relevance, Tone, Actionability, Conciseness).

## How to use

1. **Run the full evaluation harness**:
   ```bash
   python3 -m evaluation_harness.automated_metrics
   ```
   This will output a full summary and save detailed results to `outputs/eval_results.json` and `outputs/eval_summary.json`.

2. **Generate human-agreement evidence**:
   To prove that the LLM judge is reliable, you can run a script that samples 30 replies for you to score manually, and then computes the correlation and error between your scores and the judge's scores.

   *Step A: Generate the sample CSV*
   ```bash
   python3 -c "from evaluation_harness.llm_as_judge import measure_human_agreement; measure_human_agreement(30)"
   ```
   
   *Step B: Score the sample*
   Open `outputs/human_scoring_sample.csv` and fill in the `human_*` columns (1-5).
   
   *Step C: Compute the agreement*
   ```bash
   python3 -c "from evaluation_harness.llm_as_judge import compute_agreement; compute_agreement()"
   ```
   Paste the resulting table (showing MAE, Correlation, and Within-1 agreement) into your final report!
