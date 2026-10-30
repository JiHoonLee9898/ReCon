import torch
import torch.nn.functional as F

from answers import parse_answer

MASK_ID = 151666          # Dream <|mask|>
DOT_TOKEN_ID = 13         # "." is never a UPM seed (App. D.5)

TEMPERATURE = 0.2         # token sampling (semi-AR and ABD)
TOP_P = 0.95
# ABD (Sec. 4.3)
TAU = 0.9                 # in-block confidence threshold for parallel unmasking
K_FALLBACK = 2            # positions unmasked per step when nothing reaches TAU
BETA = 0.5                # forward moving-average threshold that closes a block
ALPHA = 0.9               # backward moving-average threshold that prunes the tail
# ReCon (Sec. 4.2)
REMASK = 0.9              # fraction of the output remasked in every candidate
U = 3                     # UPM: number of lowest-confidence draft tokens always masked
RECON_STEPS = 5           # parallel reconstruction steps


def _logits(model, x):
    """Dream forward with full bidirectional attention; logits[:, t] then scores the token at position t."""
    logits = model(x, "full", None).logits
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


# ============================================================================ semi-AR
def _sample(logits):
    """Token sampled at TEMPERATURE with nucleus (top-p) filtering, and its probability."""
    logits = logits / TEMPERATURE
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    remove = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) > TOP_P
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = 0
    remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_idx, remove)
    probs = torch.softmax(logits.masked_fill(remove, torch.finfo(logits.dtype).min), dim=-1)
    try:
        x0 = torch.distributions.Categorical(probs=probs).sample()
        return torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1), x0
    except Exception:                                   # invalid distribution: fall back to the top token
        return probs.max(dim=-1)


@torch.no_grad()
def semi_ar(model, prompt, steps=128, gen_length=256, block_length=32, mask_id=MASK_ID):
    L = prompt.shape[1]
    x = F.pad(prompt, (0, gen_length), value=mask_id)
    for b0 in range(L, L + gen_length, block_length):
        for _ in range(steps // (gen_length // block_length)):
            mask = x == mask_id
            mask[:, :b0] = False
            mask[:, b0 + block_length:] = False
            n = int(mask.sum())
            if n == 0:
                break
            conf_m, x0_m = _sample(_logits(model, x)[mask])
            conf = torch.full_like(x, -torch.inf, dtype=torch.float32)
            conf[mask] = conf_m.to(torch.float32)
            x0 = torch.zeros_like(x)
            x0[mask] = x0_m
            idx = conf[0].topk(max(1, min(gen_length // steps, n))).indices
            x[0, idx] = x0[0, idx]
    return x


# ============================================================================ ABD
@torch.no_grad()
def abd(model, prompt, steps=128, gen_length=256, mask_id=MASK_ID):

    L = prompt.shape[1]

    def predict(x):
        logits = _logits(model, x)
        lg = logits.clone()
        lg[..., mask_id] = float("-inf")
        return torch.softmax(logits, dim=-1).max(dim=-1).values, _sample(lg)[1]

    x = F.pad(prompt, (0, gen_length), value=mask_id)
    top1 = predict(x)[0][0, L:]                                                            # Tail pruning
    ma = torch.flip(top1, [0]).cumsum(0) / torch.arange(1, gen_length + 1, device=x.device)
    bad = (ma <= ALPHA).nonzero()
    if bad.numel() > 0:
        x = F.pad(prompt, (0, gen_length - int(bad[0, 0]) - 1), value=mask_id)
    count = 0
    while count < steps and (x[0, L:] == mask_id).any():
        top1, cand = predict(x)
        mask_pos = (x[0] == mask_id).nonzero()[:, 0]
        mask_pos = mask_pos[mask_pos >= L]
        ma = top1[0, mask_pos].cumsum(0) / torch.arange(1, mask_pos.numel() + 1, device=x.device)   # Flexible block sizing
        below = ma < BETA
        n = max(1, int(below.nonzero()[0]) if below.any() else mask_pos.numel())
        b0, b1 = int(mask_pos[0]), int(mask_pos[n - 1]) + 1
        while True:                                                                          # Block filling
            pos = b0 + (x[0, b0:b1] == mask_id).nonzero()[:, 0]
            p = top1[0, pos]
            chosen = pos[p >= TAU] if (p >= TAU).any() else pos[torch.topk(p, min(K_FALLBACK, pos.numel())).indices]
            x[0, chosen] = cand[0, chosen]
            count += 1
            if count >= steps or not (x[0, b0:b1] == mask_id).any():
                break
            top1, cand = predict(x)
    return x


# ============================================================================ ReCon
@torch.no_grad()
def recon(model, tokenizer, draft_ids, L, N=15, mask_id=MASK_ID):
    device = model.device
    x = draft_ids.to(device=device, dtype=torch.long)
    T, out = int(x.numel()), slice(L, int(x.numel()))
    n_out = T - L
    probs = torch.softmax(_logits(model, x.unsqueeze(0))[0].float(), dim=-1)
    conf = probs[torch.arange(T, device=device), x][out]
    conf[x[out] == DOT_TOKEN_ID] = 2.0
    draft_text = tokenizer.decode(x[out].tolist(), skip_special_tokens=False)
    upm = torch.topk(conf, min(U, n_out), largest=False).indices + L
    batch = x.unsqueeze(0).repeat(N, 1)
    batch[:, upm] = mask_id
    k_mask = min(n_out, int(round(REMASK * n_out)))
    for b in range(N):
        free = torch.nonzero(batch[b, out] != mask_id).squeeze(1) + L
        extra = k_mask - int(upm.numel())
        if extra > 0 and free.numel() > 0:
            batch[b, free[torch.randperm(int(free.numel()), device=device)[:extra]]] = mask_id
    conf_sum = torch.zeros(N, device=device)
    for s in range(RECON_STEPS):
        is_mask = batch == mask_id
        if not is_mask.any():
            break
        p = torch.softmax(_logits(model, batch).float(), dim=-1)
        p[..., mask_id] = 0.0
        cand_p, cand = p.max(dim=-1)
        conf_sum += (cand_p * is_mask).sum(dim=1) / is_mask.sum(dim=1).clamp(min=1)
        for b in range(N):
            pos = torch.nonzero(is_mask[b]).squeeze(1)
            k = int(pos.numel()) if s == RECON_STEPS - 1 else min(int(pos.numel()), -(-int(pos.numel()) // (RECON_STEPS - s)))
            if k > 0:
                top = pos[torch.topk(cand_p[b, pos], k).indices]
                batch[b, top] = cand[b, top]
    answers = [parse_answer(tokenizer.decode(batch[b].tolist(), skip_special_tokens=False), dream=True) for b in range(N)]
    groups = {}
    for b, a in enumerate(answers):
        if a is not None:
            g = groups.setdefault(a, [0, 0.0]); g[0] += 1; g[1] += float(conf_sum[b]) / RECON_STEPS
    if not groups:
        return draft_text
    voted = max(groups, key=lambda a: (groups[a][0], groups[a][1] / groups[a][0]))
    return "\\boxed{" + str(voted) + "}"
