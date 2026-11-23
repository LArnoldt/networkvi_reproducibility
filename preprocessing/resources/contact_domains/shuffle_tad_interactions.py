import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "encode", "ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools.csv")


def shuffle_gene_interactions(interactions, shuffle="gene2", seed=42):
    shuffled = interactions.copy()
    rng = np.random.default_rng(seed)
    if shuffle in ("gene1", "both"):
        shuffled["gene1"] = rng.permutation(shuffled["gene1"].values)
    if shuffle in ("gene2", "both"):
        shuffled["gene2"] = rng.permutation(shuffled["gene2"].values)
    return shuffled.reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=REAL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", default=None, help="compare against an existing shuffled file instead of writing")
    args = ap.parse_args()

    real = pd.read_csv(args.real, index_col=0)
    dedup = real.drop_duplicates(keep="first").reset_index(drop=True)
    shuffled = shuffle_gene_interactions(dedup, "gene2", args.seed)

    if args.verify:
        existing = pd.read_csv(args.verify, index_col=0).reset_index(drop=True)
        print("identical:", shuffled.equals(existing), f"({len(shuffled)} rows)")
    else:
        out = args.real.replace(".csv", "_shuffled.csv")
        shuffled.to_csv(out)
        same_tad = len(shuffled.merge(dedup, on=["gene1", "gene2"])) / len(shuffled)
        print(f"wrote {out}: {len(real)} real rows, {len(dedup)} unique pairs; "
              f"{same_tad:.1%} of shuffled pairs are real TAD pairs; {(shuffled.gene1 == shuffled.gene2).sum()} self-pairs")
