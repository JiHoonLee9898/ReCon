# Renoise Consistency: Unlocking Efficient Self-Correction for Diffusion Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](#)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://jihoonlee9898.github.io/ReCon/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Jihoon Lee**<sup>1</sup>, **Yeongbin Seo**<sup>2</sup>, **Jaehyung Kim**<sup>1</sup>

<sup>1</sup>Yonsei University &nbsp; <sup>2</sup>University of Toronto, Vector Institute

This repository contains code for our NeurIPS 2026 main accepted paper.

![Overview](assets/overview.png)

## Run

```bash
cd ReCon
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

## Citation

Coming soon.

## License

This code is released under the [MIT License](LICENSE). The generation code is based on the official inference code of
[LLaDA](https://github.com/ML-GSAI/LLaDA) and [Dream](https://github.com/HKUNLP/Dream).
