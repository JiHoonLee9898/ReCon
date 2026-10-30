# Scripts

```
setup.sh        environment, model weights, datasets
llada.py        LLaDA:  semi_ar(), abd(), recon()
dream.py        Dream:  semi_ar(), abd(), recon()
answers.py      answer parsing (ReCon vote) and scoring
run_eval.py     one row on one dataset -> accuracy / mean forward passes
run.sh          the whole table
datasets_/      prompts and reference answers
```

## Run

```bash
bash setup.sh                       # once
bash run.sh                         # whole table (about 15 hours on one H100)
bash run.sh llada gsm8k             # one model, one dataset

python run_eval.py --model llada --dataset gsm8k --method abd             # draft row
python run_eval.py --model llada --dataset gsm8k --method abd_recon       # ReCon on those drafts, N = 15
python run_eval.py --model llada --dataset gsm8k --method abd_recon --N 6
python run_eval.py --table
```

A ReCon row runs on the drafts stored by its draft row (`semi_ar` -> `semi_ar_recon`, `abd` -> `abd_recon`),
so the draft row has to be run first. Its steps = draft forwards + 6 ReCon forwards.
