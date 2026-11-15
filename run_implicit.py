"""Implicit-feedback ALS for top-10 ranking; hyperparameters selected on validation NDCG.

Execute after run_pipeline.py. Uses the exact same time split and all-catalog metric protocol.
"""
from __future__ import annotations
import os
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, functions as F
from run_pipeline import write_json


def top_lists(model, observed: pd.DataFrame, k=10):
    uf = model.userFactors.toPandas().sort_values("id")
    vf = model.itemFactors.toPandas().sort_values("id")
    uids = uf.id.to_numpy(dtype=np.int32)
    mids = vf.id.to_numpy(dtype=np.int32)
    u = np.stack(uf.features).astype(np.float32)
    v = np.stack(vf.features).astype(np.float32)
    index = {int(mid): i for i, mid in enumerate(mids)}
    seen = observed.groupby("userId").movieId.agg(set).to_dict()
    lists, records = {}, []
    for start in range(0, len(uids), 256):
        scores = u[start:start+256] @ v.T
        for row, uid in enumerate(uids[start:start+256]):
            uid = int(uid)
            blocked = [index[mid] for mid in seen.get(uid, set()) if mid in index]
            scores[row, blocked] = -np.inf
            top = np.argsort(-scores[row], kind="stable")[:k]
            lists[uid] = [int(mids[j]) for j in top if np.isfinite(scores[row, j])]
            records.extend({"userId": uid, "movieId": int(mids[j]), "rank": pos,
                            "score": float(scores[row, j])} for pos, j in enumerate(top, 1)
                           if np.isfinite(scores[row, j]))
    return lists, pd.DataFrame(records), (uids, mids, u, v)


def evaluate(lists, heldout, observed, catalog):
    relevant_df = heldout[(heldout.rating >= 4) & heldout.userId.isin(lists) & heldout.movieId.isin(catalog)]
    truth = relevant_df.groupby("userId").movieId.agg(set).to_dict()
    history = observed.groupby("userId").size()
    rows = []
    for uid, relevant in sorted(truth.items()):
        hits = np.array([int(mid in relevant) for mid in lists[int(uid)]])
        dcg = float((hits / np.log2(np.arange(2, len(hits)+2))).sum())
        idcg = float((1 / np.log2(np.arange(2, min(len(relevant), 10)+2))).sum())
        rows.append({"userId": int(uid), "precision_at_10": float(hits.sum()/10),
                     "recall_at_10": float(hits.sum()/len(relevant)), "ndcg_at_10": dcg/idcg,
                     "positives": len(relevant), "history_count": int(history.get(uid, 0))})
    peruser = pd.DataFrame(rows)
    metrics = peruser[["precision_at_10", "recall_at_10", "ndcg_at_10"]].mean().to_dict()
    return {**metrics, "users": len(rows), "eligible_positives": len(relevant_df)}, peruser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = root / "artifacts"
    dataset = json.loads((out / "dataset.json").read_text())
    t80, t90 = dataset["split"]["t80"], dataset["split"]["t90"]
    spark = (SparkSession.builder.master("local[4]").appName("MovieLens-Implicit-ALS-Ranking")
        .config("spark.sql.shuffle.partitions", "4").config("spark.default.parallelism", "4")
        .config("spark.driver.bindAddress", "127.0.0.1").config("spark.ui.showConsoleProgress", "false")
        .config("spark.eventLog.enabled", "true").config("spark.eventLog.compress", "true")
        .config("spark.eventLog.dir", (out/"eventlog").as_uri()).getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    data = spark.read.parquet(str(root/"data/processed/ratings"))
    pdf = data.toPandas()
    train_pdf = pdf[pdf.timestamp <= t80].copy()
    val_pdf = pdf[(pdf.timestamp > t80) & (pdf.timestamp <= t90)].copy()
    trainval_pdf = pdf[pdf.timestamp <= t90].copy()
    test_pdf = pdf[pdf.timestamp > t90].copy()
    def feedback(d):
        return d.withColumn("preference", F.when(F.col("rating") >= 4, 1.0).otherwise(0.0)).repartition(4).cache()
    train = feedback(data.filter(F.col("timestamp") <= t80))
    train.count()
    def estimator(rank, alpha):
        return ALS(userCol="userId", itemCol="movieId", ratingCol="preference", implicitPrefs=True,
            rank=rank, alpha=alpha, regParam=0.1, maxIter=10, seed=42, numUserBlocks=4,
            numItemBlocks=4, coldStartStrategy="drop", checkpointInterval=-1)
    rows = []
    for rank in [20, 40]:
        for alpha in [1.0, 10.0, 40.0]:
            print(json.dumps({"stage": "implicit_tuning", "rank": rank, "alpha": alpha}), flush=True)
            begin = time.perf_counter()
            model = estimator(rank, alpha).fit(train)
            fit_seconds = time.perf_counter()-begin
            # Score only users appearing in validation for parameter selection.
            # Item factor catalog and seen-item exclusion are identical for all candidates.
            uid_subset = pd.DataFrame({"id": sorted(set(train_pdf.userId) & set(val_pdf.userId))})
            class SubsetFactors:
                userFactors = model.userFactors.join(spark.createDataFrame(uid_subset), "id")
                itemFactors = model.itemFactors
            recs, _, factors = top_lists(SubsetFactors(), train_pdf)
            met, _ = evaluate(recs, val_pdf, train_pdf, factors[1])
            rows.append({"rank": rank, "alpha": alpha, "regParam": 0.1, "maxIter": 10,
                         "fit_seconds": fit_seconds, **met})
            pd.DataFrame(rows).to_csv(out/"tables/implicit_tuning.csv", index=False)
            model._java_obj.userFactors().unpersist()
            model._java_obj.itemFactors().unpersist()
    best = max(rows, key=lambda x: x["ndcg_at_10"])
    write_json(out/"implicit_best.json", best)
    trainval = feedback(data.filter(F.col("timestamp") <= t90))
    trainval.count()
    begin = time.perf_counter()
    model = estimator(best["rank"], best["alpha"]).fit(trainval)
    final_fit_seconds = time.perf_counter()-begin
    model.write().overwrite().save(str(out/"models/implicit_als"))
    recs, rec_df, factors = top_lists(model, trainval_pdf)
    met, peruser = evaluate(recs, test_pdf, trainval_pdf, factors[1])
    movies = pd.read_csv(out/"tables/movies.csv")
    rec_df.merge(movies, on="movieId", how="left").to_csv(out/"tables/implicit_recommendations.csv", index=False)
    uids, mids, u, v = factors
    np.savez_compressed(out/"models/implicit_factors.npz", user_ids=uids, item_ids=mids, user_factors=u, item_factors=v)
    overlap = rec_df[["userId", "movieId"]].merge(trainval_pdf[["userId", "movieId"]], on=["userId", "movieId"])
    assert len(overlap) == 0
    assert len(rec_df) == len(uids)*10
    peruser["model"] = "Implicit ALS"
    previous = pd.read_csv(out/"tables/ranking_per_user.csv")
    previous["model"] = previous.model.replace({"ALS": "Explicit ALS"})
    combined = pd.concat([previous, peruser], ignore_index=True)
    combined.to_csv(out/"tables/ranking_all_per_user.csv", index=False)
    summary = combined.groupby("model").agg(precision_at_10=("precision_at_10", "mean"),
        recall_at_10=("recall_at_10", "mean"), ndcg_at_10=("ndcg_at_10", "mean"), users=("userId", "size")).reset_index()
    summary.to_csv(out/"tables/ranking_all_models.csv", index=False)
    combined["activity_group"] = pd.cut(combined.history_count, bins=[0,100,300,float("inf")],labels=["1-100","101-300","301+"])
    combined.groupby(["model","activity_group"],observed=True).agg(users=("userId","size"),
        recall_at_10=("recall_at_10","mean"),ndcg_at_10=("ndcg_at_10","mean")).reset_index().to_csv(out/"tables/activity_all_models.csv",index=False)
    result = {"best": best, "test": met, "final_fit_seconds": final_fit_seconds,
              "generated_users": len(uids), "generated_recommendations": len(rec_df), "seen_overlap": len(overlap),
              "application_id": spark.sparkContext.applicationId, "catalog_size": len(mids),
              "feedback_definition": "preference=1 for rating>=4, otherwise 0; confidence=1+alpha*preference"}
    write_json(out/"implicit_metrics.json", result)
    print(json.dumps({"stage":"implicit_complete", **result},ensure_ascii=False),flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
