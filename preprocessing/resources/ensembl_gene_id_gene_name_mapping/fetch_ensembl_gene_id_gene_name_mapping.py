import os
import sys
import time

from pybiomart import Server


def fetch(max_retries=60, retry_delay=15):
    last_err = None
    for attempt in range(max_retries):
        try:
            server = Server(host="http://www.ensembl.org")
            dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]
            result = dataset.query(attributes=["ensembl_gene_id", "external_gene_name"])
            if "Gene name" in result.columns and "Gene stable ID" in result.columns and len(result) > 0:
                print(f"succeeded on attempt {attempt + 1}/{max_retries}, {len(result)} rows", flush=True)
                return result
            last_err = ValueError(f"unexpected BioMart response columns: {list(result.columns)}")
        except Exception as e:
            last_err = e
        print(f"attempt {attempt + 1}/{max_retries} failed: {last_err}", flush=True)
        time.sleep(retry_delay)
    raise RuntimeError(f"BioMart query failed after {max_retries} attempts") from last_err


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ensembl_gene_id_gene_name_mapping.csv")
    result = fetch()
    result.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
