# -*- coding: utf-8 -*-
"""Tune BERTopic UMAP/HDBSCAN on cached embeddings: UMAP per (n_neighbors x seed),
then sweep HDBSCAN min_cluster_size cheaply. Report topic count, outlier %, and
cross-seed ARI stability to choose one robust setting."""
import numpy as np, pandas as pd, itertools, time
from pathlib import Path
import os
DATA_ROOT = Path(os.environ.get("BIBLIO_DATA_ROOT", Path.cwd())).resolve()

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.metrics import adjusted_rand_score

OUT = DATA_ROOT / "06_全球分析" / "bertopic_full"
name = "chik"
emb = np.load(OUT/f"emb_{name}.npy")
n = len(emb)
print("emb:", emb.shape, flush=True)

NEIGHBORS = [15, 30, 50]
SEEDS = [42, 123, 2024]
MCS = [25, 40, 60, 100]

# cache UMAP 5-d embeddings
umap_emb = {}
for k in NEIGHBORS:
    for s in SEEDS:
        u = UMAP(n_neighbors=k, n_components=5, min_dist=0.0, metric="cosine",
                 random_state=s, low_memory=True)
        t0=time.time()
        umap_emb[(k,s)] = u.fit_transform(emb)
        print(f"UMAP n_neighbors={k} seed={s} done {time.time()-t0:.0f}s", flush=True)

rows=[]
for k in NEIGHBORS:
    for mc in MCS:
        labels_by_seed=[]
        for s in SEEDS:
            h = HDBSCAN(min_cluster_size=mc, metric="euclidean",
                        cluster_selection_method="eom", prediction_data=True)
            lab = h.fit_predict(umap_emb[(k,s)])
            labels_by_seed.append(lab)
        stats=[]
        for lab in labels_by_seed:
            nt = len(set(lab))-(1 if -1 in lab else 0)
            out = 100*(lab==-1).mean()
            stats.append((nt,out))
        aris=[]
        for i in range(len(SEEDS)):
            for j in range(i+1,len(SEEDS)):
                aris.append(adjusted_rand_score(labels_by_seed[i],labels_by_seed[j]))
        med_topics=int(np.median([x[0] for x in stats]))
        med_out=float(np.median([x[1] for x in stats]))
        rows.append({"n_neighbors":k,"min_cluster":mc,
                     "topics_median":med_topics,"outlier_median":round(med_out,1),
                     "ARI_mean":round(float(np.mean(aris)),3),
                     "topic_range":f"{min(x[0] for x in stats)}-{max(x[0] for x in stats)}",
                     "outlier_range":f"{min(x[1] for x in stats):.0f}-{max(x[1] for x in stats):.0f}"})
res=pd.DataFrame(rows)
res.to_csv(OUT/"chik_tuning_grid.csv",index=False,encoding="utf-8-sig")
pd.set_option("display.width",200)
print(res.to_string(index=False))
