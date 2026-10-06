import sklearn
import sys
sys.path.append("/sc-projects/sc-proj-dh-ukb-intergenics/analysis/development/arnoldtl/code/scib-metrics/src")
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import anndata as ad
import random
import numpy as np
import pandas as pd
import os


def calculate_scib_metrics(output_dir=None, embedding=None, adata=None, labels_key=None, modality_key=None, batch_key=None, patient_key=None, incorporate_batch_scib=False, incorporate_patient_scib=False, query=False, n_jobs=1):

    def run_benchmarker(embedding, label_eval=False, batch_eval=False, label_key=None, batch_key=None, labels=None, batch=None, query=False):

        if label_eval and batch_eval:
            bio_conservation_metrics = BioConservation(isolated_labels=True, nmi_ari_cluster_labels_leiden=False,
                            nmi_ari_cluster_labels_kmeans=True,
                            silhouette_label={"chunk_size": 64}, clisi_knn=True)
            batch_correction_metrics = BatchCorrection(silhouette_batch={"chunk_size": 64}, ilisi_knn=True,
                            kbet_per_label=True, graph_connectivity=True,
                            pcr_comparison=False)
        elif label_eval:
            bio_conservation_metrics = BioConservation(isolated_labels=True, nmi_ari_cluster_labels_leiden=False,
                            nmi_ari_cluster_labels_kmeans=True,
                            silhouette_label={"chunk_size": 64}, clisi_knn=True)
            batch_correction_metrics = BatchCorrection(silhouette_batch=False, ilisi_knn=False,
                            kbet_per_label=False, graph_connectivity=False,
                            pcr_comparison=False)
            batch = np.zeros_like(labels)
        elif batch_eval:
            bio_conservation_metrics = BioConservation(isolated_labels=False, nmi_ari_cluster_labels_leiden=False,
                            nmi_ari_cluster_labels_kmeans=False,
                            silhouette_label=False, clisi_knn=False)
            batch_correction_metrics = BatchCorrection(silhouette_batch={"chunk_size": 64}, ilisi_knn=True,
                            kbet_per_label=True, graph_connectivity=True,
                            pcr_comparison=False)
            labels = np.zeros_like(batch)


        adata = ad.AnnData(X=np.zeros(embedding.shape), obs=pd.DataFrame(data={"labels": labels, "batch": batch}))
        adata.obsm["X_emb"] = embedding

        bm = Benchmarker(
            adata,
            batch_key="batch",
            label_key="labels",
            embedding_obsm_keys=["X_emb"],
            n_jobs=n_jobs,
            bio_conservation_metrics=bio_conservation_metrics,
            batch_correction_metrics=batch_correction_metrics,
        )

        bm.benchmark()

        df_scib = bm.get_results(min_max_scale=False)
        
        if query is True:
            query_name = "_query"
        else:
            query_name = ""
            
        if label_eval and output_dir is not None:
            df_scib.to_csv(os.path.join(output_dir, f"scib_results_{label_key}{query_name}.csv"))
        elif batch_eval and output_dir is not None:
            df_scib.to_csv(os.path.join(output_dir, f"scib_results_{batch_key}{query_name}.csv"))

        if label_eval:
            scib_label_batch = df_scib.loc["X_emb", "Bio conservation"]
        elif batch_eval:
            scib_label_batch = df_scib.loc["X_emb", "Batch correction"]

        return scib_label_batch

    scib_label = run_benchmarker(embedding, label_eval=True, label_key="label", labels=np.array(adata.obs[labels_key]).astype("str"), query=query)
    if patient_key is not None:
        scib_patient = run_benchmarker(embedding, label_eval=True, label_key="patient_label", labels=np.array(adata.obs[patient_key]).astype("str"), query=query)
        scib_patient = run_benchmarker(embedding, batch_eval=True, batch_key="patient_batch", batch=np.array(adata.obs[patient_key]).astype("str"), query=query)

    scib_batches = []
    for scib_batch_key, incorporate_scib in zip([modality_key, batch_key], [True, incorporate_batch_scib]):
        if scib_batch_key is not None:
            scib_batch = run_benchmarker(embedding, batch_eval=True, batch_key=scib_batch_key, batch=np.array(adata.obs[scib_batch_key]).astype("str"), query=query)
            if incorporate_scib:
                scib_batches.append(scib_batch)

    if len(scib_batches) == 0:
        if incorporate_patient_scib and patient_key is not None:
            scib = 0.6*scib_label+0.4*scib_patient
        else:
            scib = scib_label
    elif len(scib_batches) == 1:
        if incorporate_patient_scib and patient_key is not None:
            scib = 0.4*scib_label+0.3*scib_patient+0.3*scib_batches[0]
        else:
            scib = 0.6*scib_label+0.4*scib_batches[0]
    elif len(scib_batches) == 2:
        if incorporate_patient_scib and patient_key is not None:
            scib = 0.4*scib_label+0.3*scib_patient+sum([0.15 * scib_batch for scib_batch in scib_batches])
        else:
            scib = 0.5*scib_label+sum([0.25*scib_batch for scib_batch in scib_batches])


    return scib

def calculate_label_transfer_metrics(true_labels, predicted_labels):

    label_transfer_metrics = {
        "f1_score_micro": sklearn.metrics.f1_score(true_labels, predicted_labels, average='micro'),
        "f1_score_macro": sklearn.metrics.f1_score(true_labels, predicted_labels, average='macro'),
        "f1_score_weighted": sklearn.metrics.f1_score(true_labels, predicted_labels, average='weighted'),
        "accuracy_score": sklearn.metrics.accuracy_score(true_labels, predicted_labels),
        "balanced_accuracy_score": sklearn.metrics.balanced_accuracy_score(true_labels, predicted_labels),
        "precision_score_micro": sklearn.metrics.precision_score(true_labels, predicted_labels, average='micro'),
        "precision_score_macro": sklearn.metrics.precision_score(true_labels, predicted_labels, average='macro'),
        "precision_score_weighted": sklearn.metrics.precision_score(true_labels, predicted_labels, average='weighted'),
        "recall_score_micro": sklearn.metrics.recall_score(true_labels, predicted_labels, average='micro'),
        "recall_score_macro": sklearn.metrics.recall_score(true_labels, predicted_labels, average='macro'),
        "recall_score_weighted": sklearn.metrics.recall_score(true_labels, predicted_labels, average='weighted'),
        "confusion_matrix": sklearn.metrics.confusion_matrix(true_labels, predicted_labels, labels=sorted(np.unique(true_labels)))
    }

    return label_transfer_metrics
    