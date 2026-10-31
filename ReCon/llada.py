import torch

from answers import parse_answer

MASK_ID = 126336          # LLaDA [MASK]
DOT_TOKEN_ID = 13         # "." is never a UPM seed (App. D.5)
MIN_BLOCK_C = 0.1         # minimum block = max(1, round(MIN_BLOCK_C * tokens removed by tail pruning))

# ABD (Sec. 4.3)
TAU = 0.9                 # in-block confidence threshold for parallel unmasking
K_FALLBACK = 2            # positions unmasked per step when nothing reaches TAU
BETA = 0.6                # forward moving-average threshold that closes a block
ALPHA = 0.7               # backward moving-average threshold that prunes the tail

# ReCon (Sec. 4.2)
REMASK = 0.9              # fraction of the output remasked in every candidate
U = 3                     # UPM: number of lowest-confidence draft tokens always masked
RECON_STEPS = 5           # parallel reconstruction steps


# ============================================================================ semi-AR
@torch.no_grad()
def semi_ar(model, prompt, steps=128, gen_length=256, block_length=32, mask_id=MASK_ID):
    """semi-AR baseline: left-to-right blocks of block_length, each revealed over steps/num_blocks steps
    (the same number of tokens per step), greedy tokens, most confident masked positions first."""
    L = prompt.shape[1]
    x = torch.full((1, L + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :L] = prompt
    per_block = steps // (gen_length // block_length)
    ks = [block_length // per_block + (i < block_length % per_block) for i in range(per_block)]
    for b0 in range(L, L + gen_length, block_length):
        for k in ks:
            mask = x == mask_id
            mask[:, b0 + block_length:] = False                  # never touch later blocks
            logits = model(x).logits
            x0 = torch.argmax(logits, dim=-1)
            conf = torch.softmax(logits.to(torch.float64), dim=-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            idx = torch.where(mask, conf, -torch.inf)[0].topk(k).indices
            x[0, idx] = x0[0, idx]
    return x


# ============================================================================ ABD
@torch.no_grad()
def abd(model, prompt, steps=128, gen_length=256, mask_id=MASK_ID):
    L = prompt.shape[1]
    x = torch.full((1, L + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :L] = prompt
    top1 = torch.softmax(model(x).logits, dim=-1).max(dim=-1).values[0, L:]               # Tail pruning
    ma = torch.flip(top1, [0]).cumsum(0) / torch.arange(1, gen_length + 1, device=x.device)
    bad = (ma <= ALPHA).nonzero()
    cut = int(bad[0, 0]) + 1 if bad.numel() > 0 else 0
    if cut > 0:
        x = x[:, :-cut]
    min_block = max(1, int(round(MIN_BLOCK_C * cut)))
    count = 0
    while True:
        probs = torch.softmax(model(x).logits, dim=-1)
        top1 = probs.max(dim=-1).values
        mask_pos = (x[0] == mask_id).nonzero()[:, 0]
        mask_pos = mask_pos[mask_pos >= L]
        if mask_pos.numel() == 0:
            break
        ma = top1[0, mask_pos].cumsum(0) / torch.arange(1, mask_pos.numel() + 1, device=x.device)   # Flexible block sizing
        below = ma < BETA
        n = int(below.nonzero()[0]) if below.any() else mask_pos.numel()
        n = min(max(1, n, min_block), mask_pos.numel())
        b0, b1 = int(mask_pos[0]), int(mask_pos[n - 1]) + 1
        while True:                                                                          # Block filling
            pos = b0 + (x[0, b0:b1] == mask_id).nonzero()[:, 0]
            p = top1[0, pos]
            chosen = pos[p >= TAU] if (p >= TAU).any() else pos[torch.topk(p, min(K_FALLBACK, pos.numel())).indices]
            top2 = probs.topk(2, dim=-1).indices
            cand = torch.where(top2[..., 0] != mask_id, top2[..., 0], top2[..., 1])
            x[0, chosen] = cand[0, chosen]
            count += 1
            if not (x[0, b0:b1] == mask_id).any():
                break
            probs = torch.softmax(model(x).logits, dim=-1)
            top1 = probs.max(dim=-1).values
        if count >= steps or not (x[0, L:] == mask_id).any():
            break
    return x


# ============================================================================ ReCon
@torch.no_grad()
def recon(model, tokenizer, draft_ids, L, N=15, mask_id=MASK_ID):
    device = model.device
    x = draft_ids.to(device=device, dtype=torch.long)
    T, out = int(x.numel()), slice(L, int(x.numel()))
    n_out = T - L
    probs = torch.softmax(model(x.unsqueeze(0)).logits[0], dim=-1)
    conf = probs[torch.arange(T, device=device), x][out].float()                           # p(draft token | draft)
    conf[x[out] == DOT_TOKEN_ID] = 2.0                                                    # never a UPM seed
    draft_text = tokenizer.decode(x[out].tolist(), skip_special_tokens=False)
    upm = torch.topk(conf, min(U, n_out), largest=False).indices + L
    batch = x.unsqueeze(0).repeat(N, 1)
    batch[:, upm] = mask_id
    k_mask = min(n_out, int(round(REMASK * n_out)))
    for b in range(N):                                                                     # uniform random renoising per row
        free = torch.nonzero(batch[b, out] != mask_id).squeeze(1) + L
        extra = k_mask - int(upm.numel())
        if extra > 0 and free.numel() > 0:
            batch[b, free[torch.randperm(int(free.numel()), device=device)[:extra]]] = mask_id
    conf_sum = torch.zeros(N, device=device)
    for s in range(RECON_STEPS):
        is_mask = batch == mask_id
        if not is_mask.any():
            break
        p = torch.softmax(model(batch).logits, dim=-1)
        p[..., mask_id] = 0.0                                                              # never predict the mask token
        cand_p, cand = p.max(dim=-1)
        conf_sum += (cand_p * is_mask).sum(dim=1) / is_mask.sum(dim=1).clamp(min=1)        # mean confidence of this refill
        for b in range(N):
            pos = torch.nonzero(is_mask[b]).squeeze(1)
            k = int(pos.numel()) if s == RECON_STEPS - 1 else min(int(pos.numel()), -(-int(pos.numel()) // (RECON_STEPS - s)))
            if k > 0:
                top = pos[torch.topk(cand_p[b, pos], k).indices]
                batch[b, top] = cand[b, top]
    answers = [parse_answer(tokenizer.decode(batch[b].tolist(), skip_special_tokens=False)) for b in range(N)]
    groups = {}
    for b, a in enumerate(answers):
        if a is not None:
            g = groups.setdefault(a, [0, 0.0]); g[0] += 1; g[1] += float(conf_sum[b]) / RECON_STEPS
    if not groups:
        return draft_text
    voted = max(groups, key=lambda a: (groups[a][0], groups[a][1] / groups[a][0]))
    return "\\boxed{" + voted + "}"
