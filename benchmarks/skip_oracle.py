"""How many (query-block, key-block) tiles of real H3 attention could be skipped while every skipped score is at
least T (natural log units) below its row's maximum, i.e. each skipped key weighs < e^-T of the top key?"""
import sys, torch
BM = 128
for f in sys.argv[1:]:
    d = torch.load(f); q, k = d["q"].cuda().float(), d["k"].cuda().float()
    S = q.shape[2]; H = q.shape[1]
    res = {}
    g = torch.Generator(device="cuda").manual_seed(0)
    for h in range(H):
        for qb in torch.randint(0, S // BM, (6,), generator=g, device="cuda").tolist():
            s = (q[0, h, qb * BM:(qb + 1) * BM] @ k[0, h].T) * 128 ** -0.5  # [128, S]
            gap = s - s.max(1, keepdim=True).values  # <= 0
            for BN in (32, 64):
                n = (S // BN) * BN
                blk = gap[:, :n].reshape(BM, -1, BN).amax(2).amax(0)  # best row of the query block, per key block
                for T in (12, 16, 20):
                    res.setdefault((BN, T), []).append((blk < -T).float().mean().item())
    print(f"--- {f.split('/')[-1]}")
    for (BN, T), v in sorted(res.items()):
        print(f"  key block {BN:3d}, skip when > {T} below row max (each skipped key < e^-{T} = {2.718**-T:.1e} of the top one): {100 * sum(v) / len(v):5.1f}% of tiles skippable")
