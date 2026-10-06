"""Fetch IRF1/IRF2 K562 ChIP-seq fold-change-over-control tracks, cropped to chr5:96,760,000-97,038,000 (TAD2, Fig. 4f)."""
import os
import tempfile

import pyBigWig
import requests

TRACKS = {
    "IRF1": {
        "accession": "ENCFF630FXX",
        "cell_line": "K562",
        "assembly": "GRCh38",
        "date": "2020-12-01",
    },
    "IRF2": {
        "accession": "ENCFF887HOR",
        "cell_line": "K562",
        "assembly": "GRCh38",
        "date": "2020-12-01",
    },
}

REGION_CHROM = "chr5"
REGION_START = 96760000
REGION_END = 97038000


def download_full_bigwig(accession, out_path):
    r = requests.get(f"https://www.encodeproject.org/files/{accession}/@@download/{accession}.bigWig", allow_redirects=True)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)


def crop_bigwig(full_path, cropped_path):
    bw_in = pyBigWig.open(full_path)
    chrom_length = bw_in.chroms()[REGION_CHROM]
    values = bw_in.values(REGION_CHROM, REGION_START, REGION_END)
    bw_in.close()

    bw_out = pyBigWig.open(cropped_path, "w")
    bw_out.addHeader([(REGION_CHROM, chrom_length)])
    bw_out.addEntries(REGION_CHROM, REGION_START, values=values, span=1, step=1)
    bw_out.close()


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, meta in TRACKS.items():
            full_path = os.path.join(tmp_dir, f"{meta['accession']}.bigWig")
            print(f"downloading {name} ({meta['accession']}) ...", flush=True)
            download_full_bigwig(meta["accession"], full_path)

            cropped_name = (
                f"{meta['accession']}_{meta['cell_line']}_{name}_{meta['assembly']}_{meta['date']}"
                f"_fc_{REGION_CHROM}_{REGION_START}_{REGION_END}.bigWig"
            )
            cropped_path = os.path.join(out_dir, cropped_name)
            crop_bigwig(full_path, cropped_path)
            print(f"wrote {cropped_path}", flush=True)
