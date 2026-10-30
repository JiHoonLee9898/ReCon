This repository contains code for our NeurIPS 2026 main accepted paper:
Renoise Consistency: Unlocking Efficient Self-Correction for Diffusion Large Language Models

## Run

```bash
bash setup.sh                       # install
bash run.sh                         # all at once
bash run.sh llada gsm8k             # one model, one dataset

python run_eval.py --model llada --dataset gsm8k --method abd             # draft row
python run_eval.py --model llada --dataset gsm8k --method abd_recon       # ReCon on those drafts, N = 15
python run_eval.py --model llada --dataset gsm8k --method abd_recon --N 6
python run_eval.py --table
```

A ReCon row runs on the drafts stored by its draft row (`semi_ar` -> `semi_ar_recon`, `abd` -> `abd_recon`),
so the draft row has to be run first. Its steps = draft forwards + 6 ReCon forwards.
