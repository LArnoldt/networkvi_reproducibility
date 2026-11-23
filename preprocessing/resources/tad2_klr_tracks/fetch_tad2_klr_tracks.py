import argparse
import os
import tempfile

import numpy as np
import pandas as pd
import pyBigWig
import requests

TRACKS = [
    ("CTCF", "ENCFF979PWH"),
    ("CTCF", "ENCFF736UDR"),
    ("CTCF", "ENCFF534ITO"),
    ("CTCF", "ENCFF682MFJ"),
    ("RAD21", "ENCFF652NKM"),
    ("RAD21", "ENCFF715HLA"),
    ("SMC3", "ENCFF469OWD"),
    ("CREB1", "ENCFF517YHP"),
    ("CREB1", "ENCFF387IRJ"),
    ("CREB3L1", "ENCFF116CGU"),
    ("PHF21A", "ENCFF314SJU"),
    ("RNF2", "ENCFF306PNK"),
    ("RNF2", "ENCFF511JBH"),
    ("ZBTB12", "ENCFF402KJP"),
]

REGION_CHROM = "chr12"
REGION_START = 10220000
REGION_END = 10580000
BIN_SIZE = 1000


def download_bigwig(accession, out_path):
    with requests.get(f"https://www.encodeproject.org/files/{accession}/@@download/{accession}.bigWig", allow_redirects=True, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def bin_maxima(path):
    edges = np.arange(REGION_START, REGION_END + 1, BIN_SIZE)
    bw = pyBigWig.open(path)
    values = [bw.stats(REGION_CHROM, int(a), int(b), type="max")[0] or 0.0 for a, b in zip(edges[:-1], edges[1:])]
    bw.close()
    return edges[:-1], values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "klr_locus_chip_fc.csv"))
    parser.add_argument("--tmp_dir", default=None)
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        for factor, accession in TRACKS:
            path = os.path.join(tmp_dir, f"{accession}.bigWig")
            print(f"downloading {factor} ({accession}) ...", flush=True)
            download_bigwig(accession, path)
            starts, values = bin_maxima(path)
            rows += [(factor, accession, int(s), float(v)) for s, v in zip(starts, values)]
            os.remove(path)
    pd.DataFrame(rows, columns=["factor", "accession", "bin_start", "max_fold_change"]).to_csv(args.out, index=False)
    print(f"wrote {args.out}", flush=True)
