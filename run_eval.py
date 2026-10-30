"""Run one row of the paper's main table on one dataset and print accuracy / mean forward passes.

    python run_eval.py --model llada --dataset gsm8k --method abd           # draft only (stores the draft ids)
    python run_eval.py --model llada --dataset gsm8k --method abd_recon     # ReCon on the stored ABD drafts
    python run_eval.py --model llada --dataset gsm8k --method abd_recon --N 6
    python run_eval.py --table

ReCon rows read the drafts of the matching draft row (semi_ar -> semi_ar_recon, abd -> abd_recon), so the draft row
must be run first (run.sh does). Steps of a ReCon row = draft forwards + ReCon forwards.
Writes results/<model>_<dataset>_<method>[_N6].json (accuracy, mean steps, every generation).
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from answers import SCORERS
from datasets_ import DATASET_MAP
import dream
import llada

MODELS = {"llada": ("models/LLaDA-8B-Instruct", llada), "dream": ("models/Dream-v0-Instruct-7B", dream)}
METHODS = ["semi_ar", "semi_ar_recon", "abd", "abd_recon"]
DATASETS = ["gsm8k", "arcc", "truthfulqa", "math"]


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ----------------------------------------------------------------------------- run
def result_path(output_dir, model, dataset, method, N=15, seed=42):
    tag = method + (f"_N{N}" if "recon" in method and N != 15 else "") + (f"_seed{seed}" if seed != 42 else "")
    return f"{output_dir}/{model}_{dataset}_{tag}.json"


@torch.no_grad()
def run(args):
    init_seed(args.seed)
    model_path = args.model_path or os.environ.get(f"{args.model.upper()}_PATH", MODELS[args.model][0])
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(0).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dataset = DATASET_MAP[args.dataset](tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    init_seed(args.seed)
    torch.cuda.empty_cache()   # keep: the allocator state changes which cuBLAS kernels are picked (bf16 numerics)
    M = MODELS[args.model][1]
    scorer = SCORERS[args.dataset]

    drafts = None
    if args.method.endswith("_recon"):
        draft_file = result_path(args.output_dir, args.model, args.dataset, args.method[:-len("_recon")], seed=args.seed)
        assert os.path.exists(draft_file), f"run the draft row first: {draft_file} not found"
        drafts = json.load(open(draft_file))["generations"]

    forwards = [0]                                   # steps = number of model forward passes per example
    _forward = model.forward
    def counting_forward(*a, **k):
        forwards[0] += 1
        return _forward(*a, **k)
    model.forward = counting_forward

    rows = []
    for i, batch in enumerate(tqdm(loader, desc=f"{args.model} {args.dataset} {args.method}")):
        if args.limit is not None and i >= args.limit:
            break
        forwards[0] = 0
        row = {"question": batch["questions"][0], "ground_truth": batch["answers"][0]}
        if drafts is None:                           # draft row
            prompt = batch["input_ids"].to(model.device)
            ids = getattr(M, args.method)(model, prompt)
            row["generation"] = tokenizer.decode(ids[0, prompt.shape[1]:].tolist(), skip_special_tokens=False)
            row["draft_ids"], row["prompt_len"] = ids[0].tolist(), int(prompt.shape[1])
        else:                                        # ReCon on the stored draft
            d = drafts[i]
            assert d["question"] == row["question"], f"draft {i} is for another question"
            ids = torch.tensor(d["draft_ids"], dtype=torch.long, device=model.device)
            row["generation"] = M.recon(model, tokenizer, ids, int(d["prompt_len"]), args.N)
            forwards[0] += int(d["steps"])
        row["correct"] = bool(scorer(row["generation"], row["ground_truth"]))
        row["steps"] = forwards[0]
        rows.append(row)

    acc = 100.0 * sum(r["correct"] for r in rows) / max(len(rows), 1)
    mean_steps = sum(r["steps"] for r in rows) / max(len(rows), 1)
    os.makedirs(args.output_dir, exist_ok=True)
    fname = result_path(args.output_dir, args.model, args.dataset, args.method, args.N, args.seed)
    with open(fname, "w") as f:
        json.dump({"model": args.model, "dataset": args.dataset, "method": args.method, "N": args.N, "seed": args.seed,
                   "accuracy": acc, "mean_steps": mean_steps, "n": len(rows), "generations": rows}, f, indent=1)
    print(f"[{args.model} | {args.dataset} | {os.path.basename(fname)[:-5]}] accuracy = {acc:.2f}%  "
          f"steps = {mean_steps:.1f}  ({len(rows)} examples)  -> {fname}")


def table(output_dir):
    """accuracy / mean steps of every results file: one block per model, one row per method, one column per dataset."""
    cells = {}
    for fn in glob.glob(f"{output_dir}/*.json"):
        d = json.load(open(fn))
        tag = os.path.basename(fn)[len(d["model"]) + len(d["dataset"]) + 2:-5]
        cells[(d["model"], tag, d["dataset"])] = (d["accuracy"], d["mean_steps"])
    for model in MODELS:
        tags = sorted({t for m, t, _ in cells if m == model}, key=lambda t: (METHODS.index(t.split("_N")[0].split("_seed")[0]), t))
        if not tags:
            continue
        print(f"{model + ' acc (%) / steps':<24}" + "".join(f"{ds:>16}" for ds in DATASETS))
        for t in tags:
            line = "".join(f"{'-':>16}" if (model, t, ds) not in cells else "{:>9.2f} /{:>5.1f}".format(*cells[(model, t, ds)])
                           for ds in DATASETS)
            print(f"{t:<24}" + line)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), default="llada")
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--N", type=int, default=15, help="ReCon candidates (15: Table 2, 6: Table 3 / 4)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="only the first K examples (smoke test)")
    p.add_argument("--model_path", default=None, help="default: models/<model dir> or $LLADA_PATH / $DREAM_PATH")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--table", action="store_true", help="print the table of everything in --output_dir and exit")
    args = p.parse_args()
    if args.table:
        table(args.output_dir)
    else:
        assert args.dataset and args.method, "--dataset and --method are required"
        run(args)
