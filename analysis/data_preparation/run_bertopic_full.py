# -*- coding: utf-8 -*-
"""Research-grade FULL-CORPUS BERTopic rerun (no sampling).
Independent of the Streamlit app (the app's quick/full caps are a UI-latency design choice).
Tuned setting (chik): UMAP n_neighbors=30, HDBSCAN min_cluster_size=max(60,0.6% of corpus),
eom; reports coverage, raw and outlier-reduced outlier %, all parameters, multi-seed ARI.
Outliers are reassigned by embedding similarity (BERTopic.reduce_outliers, strategy=embeddings).

Usage:  python run_bertopic_full.py chik
        python run_bertopic_full.py chik,dengue,zika,wnv
"""
import sys, time
import numpy as np, pandas as pd
from pathlib import Path
import os
DATA_ROOT = Path(os.environ.get("BIBLIO_DATA_ROOT", Path.cwd())).resolve()


OUT = DATA_ROOT / "06_全球分析" / "bertopic_full"
OUT.mkdir(parents=True, exist_ok=True)
DATA = DATA_ROOT / "02_原始数据_WoS"
FILES = {
 "chik":  DATA/"Merged_Chikungunya_WOS_Full_桌面6月7日原始版_77MB.txt",
 "dengue":DATA/"四病毒补充WOS"/"Merged_dengue_WOS_Full.txt",
 "zika":  DATA/"四病毒补充WOS"/"Merged_ZIKA_WOS_Full.txt",
 "wnv":   DATA/"四病毒补充WOS"/"Merged_west_nile_WOS_Full.txt",
}
N_NEIGHBORS = 30
SEEDS = (42, 123, 2024)

def load_records(path):
    df = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig",
                     low_memory=False, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    df = df[df["DT"].fillna("").str.contains("Article|Review", case=False, regex=True)].copy()
    if "UT" in df: df = df.drop_duplicates(subset="UT")
    df["Year"] = pd.to_numeric(df["PY"], errors="coerce")
    df = df[df["Year"] <= 2025]
    ti = df["TI"].fillna("").str.strip(); ab = df["AB"].fillna("").str.strip()
    docs = [(t + ". " + a).strip() if a not in ("","nan") else t for t,a in zip(ti,ab)]
    df["doc"] = [d if len(d) > 20 else None for d in docs]
    return df[df["doc"].notna()].copy()

def run_one(name):
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic import BERTopic
    from bertopic.representation import MaximalMarginalRelevance
    from sklearn.metrics import adjusted_rand_score
    t0=time.time()

    df = load_records(FILES[name]); docs = df["doc"].tolist()
    years = df["Year"].astype(int).tolist(); n = len(docs)
    print(f"[{name}] docs={n}", flush=True)
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_cache = OUT/f"emb_{name}.npy"
    emb = np.load(emb_cache) if emb_cache.exists() else emb_model.encode(
        docs, show_progress_bar=True, batch_size=128); np.save(emb_cache, emb) if not emb_cache.exists() else None

    min_cluster = int(max(60, round(n*0.006)))
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    rep = MaximalMarginalRelevance(diversity=0.3)

    raw_labels, primary = [], None
    for seed in SEEDS:
        umap = UMAP(n_neighbors=N_NEIGHBORS, n_components=5, min_dist=0.0,
                    metric="cosine", random_state=seed, low_memory=True)
        hdb = HDBSCAN(min_cluster_size=min_cluster, metric="euclidean",
                      cluster_selection_method="eom", prediction_data=True)
        model = BERTopic(embedding_model=emb_model, umap_model=umap, hdbscan_model=hdb,
                         vectorizer_model=vectorizer, representation_model=rep,
                         calculate_probabilities=False, language="english", verbose=False)
        topics, _ = model.fit_transform(docs, embeddings=emb)
        topics = np.array(topics); raw_labels.append(topics)
        ntop = len(set(topics))-(1 if -1 in topics else 0)
        print(f"[{name}] seed={seed} raw: topics={ntop} outlier={100*(topics==-1).mean():.1f}%", flush=True)
        if seed == SEEDS[0]: primary = model

    aris = [adjusted_rand_score(raw_labels[0], raw_labels[i]) for i in range(1,len(SEEDS))]
    raw = raw_labels[0]
    raw_topics = len(set(raw))-(1 if -1 in raw else 0)
    raw_out = 100*(raw==-1).mean()

    # reassign outliers to nearest topic using embeddings
    red = np.array(primary.reduce_outliers(docs, raw.tolist(), strategy="embeddings", embeddings=emb))
    primary.update_topics(docs, topics=red, vectorizer_model=vectorizer, representation_model=rep)
    red_topics = len(set(red))-(1 if -1 in red else 0)
    red_out = 100*(red==-1).mean()
    print(f"[{name}] reduced: topics={red_topics} outlier={red_out:.1f}%  ARI={np.mean(aris):.3f}", flush=True)

    info = primary.get_topic_info()
    info.to_csv(OUT/f"bertopic_{name}_topics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"UT": df["UT"].values if "UT" in df else np.arange(n),
                  "Year": years, "Topic_raw": raw, "Topic": red}).to_csv(
        OUT/f"bertopic_{name}_doc_topics.csv", index=False, encoding="utf-8-sig")
    try:
        tot = primary.topics_over_time(docs, years, topics=red.tolist(), nr_bins=10)
        tot.to_csv(OUT/f"bertopic_{name}_topics_over_time.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        print("  topics_over_time failed:", e)

    report = {"virus":name,"n_docs":n,"min_cluster_size":min_cluster,"umap_n_neighbors":N_NEIGHBORS,
      "raw_topics":raw_topics,"raw_outlier_pct":round(raw_out,1),
      "final_topics":red_topics,"final_outlier_pct":round(red_out,1),
      "coverage_pct":round(100*(1-red_out/100),1),
      "embedding":"all-MiniLM-L6-v2 (384d)","hdbscan_method":"eom",
      "vectorizer":"CountVectorizer(1-2gram,stop_words,min_df=1)",
      "representation":"MMR(diversity=0.3)","outlier_reduction":"embeddings",
      "seeds":list(SEEDS),"ARI_mean_vs_seed42":[round(x,3) for x in aris],
      "minutes":round((time.time()-t0)/60,1)}
    with open(OUT/f"bertopic_{name}_report.txt","w",encoding="utf-8") as f:
        for k,v in report.items(): f.write(f"{k}: {v}\n")
    print(f"[{name}] DONE {report}", flush=True)
    return report

if __name__ == "__main__":
    targets = sys.argv[1].split(",") if len(sys.argv)>1 else ["chik"]
    for r in [run_one(t) for t in targets]: print(r)
