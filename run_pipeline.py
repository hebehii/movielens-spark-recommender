"""Reproducible MovieLens-1M batch pipeline using real Spark MLlib ALS.

Run: python run_pipeline.py --root .
All fitted models, split manifests and measurements are written under artifacts/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from pyspark import __version__ as spark_version
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, functions as F, types as T


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def score_errors(actual, predicted):
    delta = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {"rmse": float(np.sqrt(np.mean(delta * delta))), "mae": float(np.mean(np.abs(delta))), "n": len(delta)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--master", default="local[4]")
    parser.add_argument("--performance-repeats", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    raw = root / "data/raw"
    out = root / "artifacts"
    for p in [raw, out, out / "tables", out / "models", out / "eventlog", out / "figures"]:
        p.mkdir(parents=True, exist_ok=True)
    status = out / "status.json"
    def update(stage, **kwargs):
        write_json(status, {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), **kwargs})
        print(json.dumps({"stage": stage, **kwargs}, ensure_ascii=False), flush=True)

    archive = raw / "ml-1m.zip"
    source_url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    if not archive.exists():
        urllib.request.urlretrieve(source_url, archive)
    if not (raw / "ml-1m/ratings.dat").exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(raw)
    ratings_file = raw / "ml-1m/ratings.dat"
    movies_file = raw / "ml-1m/movies.dat"
    # MovieLens titles are ISO-8859-1; convert once before Spark CSV ingest.
    movies_utf8 = raw / "movies_utf8.dat"
    movies_utf8.write_text(movies_file.read_text(encoding="latin-1"), encoding="utf-8")

    started = time.perf_counter()
    spark = (SparkSession.builder.master(args.master).appName("MovieLens-1M-ALS-Course-Project")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.compress", "true")
        .config("spark.eventLog.dir", (out / "eventlog").as_uri())
        .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    java_version = spark.sparkContext._jvm.java.lang.System.getProperty("java.version")
    update("spark_ready", application_id=spark.sparkContext.applicationId, ui=spark.sparkContext.uiWebUrl)

    schema = T.StructType([T.StructField("userId", T.IntegerType()), T.StructField("movieId", T.IntegerType()),
                           T.StructField("rating", T.FloatType()), T.StructField("timestamp", T.LongType())])
    original = spark.read.option("sep", "::").schema(schema).csv(str(ratings_file))
    raw_count = original.count()
    invalid_expr = (F.col("userId").isNull() | F.col("movieId").isNull() | F.col("rating").isNull()
                    | F.col("timestamp").isNull() | ~F.col("rating").between(1, 5)
                    | (F.col("userId") <= 0) | (F.col("movieId") <= 0))
    invalid_count = original.filter(invalid_expr).count()
    valid = original.filter(~invalid_expr)
    duplicate_count = valid.count() - valid.select("userId", "movieId").distinct().count()
    # Keep the most recent record for any duplicate pair, breaking ties deterministically.
    from pyspark.sql.window import Window
    w = Window.partitionBy("userId", "movieId").orderBy(F.desc("timestamp"), F.desc("rating"))
    cleaned = valid.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn").cache()
    clean_count = cleaned.count()
    movie_schema = "movieId INT, title STRING, genres STRING"
    movies = spark.read.option("sep", "::").schema(movie_schema).csv(str(movies_utf8)).cache()
    missing_movie_count = cleaned.join(movies.select("movieId"), "movieId", "left_anti").count()
    cleaned.write.mode("overwrite").parquet(str(root / "data/processed/ratings"))
    movies.write.mode("overwrite").parquet(str(root / "data/processed/movies"))
    cleaned.createOrReplaceTempView("ratings")
    movies.createOrReplaceTempView("movies")
    ratings_pdf = cleaned.toPandas().sort_values(["timestamp", "userId", "movieId"]).reset_index(drop=True)
    movies_pdf = movies.toPandas().sort_values("movieId")
    movies_pdf.to_csv(out / "tables/movies.csv", index=False)
    ratings_pdf.groupby("rating").size().rename("count").reset_index().to_csv(out / "tables/rating_distribution.csv", index=False)
    user_activity = spark.sql("SELECT userId, COUNT(*) AS count FROM ratings GROUP BY userId").toPandas()
    user_activity.to_csv(out / "tables/user_activity.csv", index=False)
    item_stats = spark.sql("""SELECT r.movieId, m.title, COUNT(*) AS count, AVG(r.rating) AS mean_rating
        FROM ratings r JOIN movies m ON r.movieId=m.movieId GROUP BY r.movieId,m.title""").toPandas()
    item_stats.to_csv(out / "tables/movie_statistics.csv", index=False)

    # Exact quantiles and explicit <= boundaries make the time split reproducible.
    t80, t90 = [int(x) for x in ratings_pdf.timestamp.quantile([0.8, 0.9]).tolist()]
    train = cleaned.filter(F.col("timestamp") <= t80).repartition(4).cache()
    validation = cleaned.filter((F.col("timestamp") > t80) & (F.col("timestamp") <= t90)).cache()
    test = cleaned.filter(F.col("timestamp") > t90).cache()
    n_train, n_val, n_test = train.count(), validation.count(), test.count()
    val_pdf = ratings_pdf[(ratings_pdf.timestamp > t80) & (ratings_pdf.timestamp <= t90)].copy()
    test_pdf = ratings_pdf[ratings_pdf.timestamp > t90].copy()
    train_pdf = ratings_pdf[ratings_pdf.timestamp <= t80].copy()
    dataset = {
        "name": "MovieLens 1M", "url": "https://grouplens.org/datasets/movielens/1m/", "download_url": source_url,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "ratings_sha256": hashlib.sha256(ratings_file.read_bytes()).hexdigest(),
        "raw_ratings": raw_count, "clean_ratings": clean_count, "invalid_records": invalid_count,
        "duplicate_pairs": duplicate_count, "missing_movie_records": missing_movie_count,
        "users": int(ratings_pdf.userId.nunique()), "rated_movies": int(ratings_pdf.movieId.nunique()),
        "catalog_movies": len(movies_pdf), "mean_rating": float(ratings_pdf.rating.mean()),
        "sparsity": float(1-clean_count/(ratings_pdf.userId.nunique()*ratings_pdf.movieId.nunique())),
        "min_timestamp": int(ratings_pdf.timestamp.min()), "max_timestamp": int(ratings_pdf.timestamp.max()),
        "split": {"method": "global chronological 80/10/10, timestamp <= boundary", "t80": t80, "t90": t90,
                  "train": n_train, "validation": n_val, "test": n_test}}
    write_json(out / "dataset.json", dataset)
    update("data_ready", dataset=dataset)

    def als_for(rank, reg, seed=42):
        return ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank=rank, regParam=reg,
            maxIter=10, seed=seed, coldStartStrategy="drop", nonnegative=False, implicitPrefs=False,
            numUserBlocks=4, numItemBlocks=4, checkpointInterval=-1)

    runs = []
    for rank in [10, 20, 40]:
        for reg in [0.01, 0.1]:
            update("tuning", rank=rank, regParam=reg, completed=len(runs))
            begin = time.perf_counter()
            model = als_for(rank, reg).fit(train)
            elapsed = time.perf_counter() - begin
            pred = model.transform(validation).select("userId", "movieId", "rating", "prediction").toPandas()
            metrics = score_errors(pred.rating, pred.prediction)
            runs.append({"rank": rank, "regParam": reg, "maxIter": 10, "fit_seconds": elapsed,
                         "val_rmse": metrics["rmse"], "val_mae": metrics["mae"],
                         "val_predicted": len(pred), "val_total": n_val, "coverage": len(pred)/n_val})
            pd.DataFrame(runs).to_csv(out / "tables/tuning.csv", index=False)
            # Release cached latent factors from each candidate model.
            model._java_obj.userFactors().unpersist()
            model._java_obj.itemFactors().unpersist()

    best = min(runs, key=lambda x: x["val_rmse"])
    write_json(out / "best_parameters.json", best)
    trainval = cleaned.filter(F.col("timestamp") <= t90).repartition(4).cache()
    trainval_count = trainval.count()
    final_train_pdf = ratings_pdf[ratings_pdf.timestamp <= t90].copy()
    update("final_training", best=best, trainval=trainval_count)
    start_fit = time.perf_counter()
    model = als_for(best["rank"], best["regParam"]).fit(trainval)
    final_fit_seconds = time.perf_counter()-start_fit
    model.write().overwrite().save(str(out / "models/als"))
    predictions = model.transform(test).select("userId", "movieId", "prediction").toPandas()
    all_pred = test_pdf.merge(predictions, on=["userId", "movieId"], how="left", validate="one_to_one")
    global_mean = float(final_train_pdf.rating.mean())
    item_means = final_train_pdf.groupby("movieId").rating.mean()
    all_pred["global_mean_prediction"] = global_mean
    all_pred["item_mean_prediction"] = all_pred.movieId.map(item_means).fillna(global_mean)
    all_pred["als_with_fallback"] = all_pred.prediction.fillna(all_pred.item_mean_prediction)
    warm = all_pred.prediction.notna()
    eval_rows = []
    for cohort, mask in [("all_test", np.ones(len(all_pred), dtype=bool)), ("warm_test", warm.to_numpy())]:
        d = all_pred[mask]
        for label, col in [("Global mean", "global_mean_prediction"), ("Item mean", "item_mean_prediction"),
                           ("ALS + fallback" if cohort == "all_test" else "ALS", "als_with_fallback")]:
            eval_rows.append({"cohort": cohort, "model": label, **score_errors(d.rating, d[col])})
    pd.DataFrame(eval_rows).to_csv(out / "tables/rating_metrics.csv", index=False)
    all_pred.to_csv(out / "tables/test_predictions.csv", index=False)

    # Serving layer: export Spark-trained factors and use batched matrix products.
    users = model.userFactors.toPandas().sort_values("id")
    items = model.itemFactors.toPandas().sort_values("id")
    user_ids = users.id.to_numpy(dtype=np.int32)
    item_ids = items.id.to_numpy(dtype=np.int32)
    u = np.stack(users.features).astype(np.float32)
    v = np.stack(items.features).astype(np.float32)
    np.savez_compressed(out / "models/factors.npz", user_ids=user_ids, item_ids=item_ids, user_factors=u, item_factors=v)
    item_idx = {int(mid): j for j, mid in enumerate(item_ids)}
    seen = final_train_pdf.groupby("userId").movieId.agg(set).to_dict()
    pop = final_train_pdf[final_train_pdf.rating >= 4].groupby("movieId").size()
    pop_scores = np.array([pop.get(mid, 0) for mid in item_ids], dtype=np.float64)
    # Stable descending scores; item_id order resolves ties reproducibly.
    pop_order = np.argsort(-pop_scores, kind="stable")
    topk = 10
    als_recs, popular_recs = {}, {}
    rec_records = []
    update("recommendation_scoring", users=len(user_ids), items=len(item_ids), candidates="all trained items minus seen")
    rec_start = time.perf_counter()
    for left in range(0, len(user_ids), 256):
        batch = u[left:left+256] @ v.T
        for row, user_id in enumerate(user_ids[left:left+256]):
            uid = int(user_id)
            watched = seen.get(uid, set())
            blocked = [item_idx[mid] for mid in watched if mid in item_idx]
            batch[row, blocked] = -np.inf
            order = np.argsort(-batch[row], kind="stable")[:topk]
            als_recs[uid] = [int(item_ids[j]) for j in order if np.isfinite(batch[row, j])]
            popular_recs[uid] = [int(item_ids[j]) for j in pop_order if int(item_ids[j]) not in watched][:topk]
            for rankpos, j in enumerate(order, 1):
                if np.isfinite(batch[row, j]):
                    rec_records.append({"userId": uid, "movieId": int(item_ids[j]), "rank": rankpos,
                                        "prediction": float(batch[row, j])})
    scoring_seconds = time.perf_counter()-rec_start
    rec_df = pd.DataFrame(rec_records).merge(movies_pdf, on="movieId", how="left")
    rec_df.to_csv(out / "tables/recommendations.csv", index=False)
    final_train_pdf.merge(movies_pdf, on="movieId", how="left").to_parquet(out / "tables/history.parquet", index=False)
    # Evaluate the full-catalog ranking task, with no sampled negatives.
    positive = test_pdf[test_pdf.rating >= 4]
    eligible = positive[positive.userId.isin(user_ids) & positive.movieId.isin(item_ids)]
    relevant = eligible.groupby("userId").movieId.agg(set).to_dict()
    ranking_per_user = []
    for uid, truth in sorted(relevant.items()):
        for name, recs in [("Popular", popular_recs), ("ALS", als_recs)]:
            top = recs[int(uid)]
            hits = np.array([int(mid in truth) for mid in top])
            dcg = float((hits/np.log2(np.arange(2, len(hits)+2))).sum())
            idcg = float((1/np.log2(np.arange(2, min(len(truth), topk)+2))).sum())
            ranking_per_user.append({"userId": int(uid), "model": name, "precision_at_10": float(hits.sum()/topk),
                "recall_at_10": float(hits.sum()/len(truth)), "ndcg_at_10": dcg/idcg,
                "positives": len(truth), "history_count": len(seen[int(uid)])})
    per_user = pd.DataFrame(ranking_per_user)
    per_user.to_csv(out / "tables/ranking_per_user.csv", index=False)
    ranking = per_user.groupby("model")[["precision_at_10", "recall_at_10", "ndcg_at_10"]].mean().reset_index()
    ranking["users"] = len(relevant)
    ranking.to_csv(out / "tables/ranking_metrics.csv", index=False)
    per_user["activity_group"] = pd.cut(per_user.history_count, bins=[0, 100, 300, float("inf")],
        labels=["1-100", "101-300", "301+"])
    segments = per_user.groupby(["model", "activity_group"], observed=True).agg(
        users=("userId", "size"), recall_at_10=("recall_at_10", "mean"), ndcg_at_10=("ndcg_at_10", "mean")).reset_index()
    segments.to_csv(out / "tables/activity_metrics.csv", index=False)
    # Verify no watched items leak into any generated top-10 list.
    overlap = rec_df[["userId", "movieId"]].merge(final_train_pdf[["userId", "movieId"]], on=["userId", "movieId"])
    assert len(overlap) == 0, "Recommendation contains an already-rated movie"
    assert rec_df.groupby("userId").movieId.nunique().min() == 10
    assert n_train+n_val+n_test == clean_count
    assert train_pdf.timestamp.max() < val_pdf.timestamp.min()
    assert val_pdf.timestamp.max() < test_pdf.timestamp.min()
    final_metrics = {"best": best, "final_fit_seconds": final_fit_seconds, "trainval_ratings": trainval_count,
        "test_ratings": n_test, "warm_test_ratings": int(warm.sum()), "cold_test_ratings": int((~warm).sum()),
        "warm_coverage": float(warm.mean()), "ranking_users": len(relevant), "ranking_test_positives": len(positive),
        "ranking_eligible_positives": len(eligible), "ranking_catalog_size": len(item_ids),
        "generated_users": len(user_ids), "generated_recommendations": len(rec_df),
        "scoring_seconds": scoring_seconds, "seen_overlap": len(overlap),
        "rating_metrics": eval_rows, "ranking_metrics": ranking.to_dict(orient="records")}
    write_json(out / "metrics.json", final_metrics)
    update("evaluation_complete", metrics=final_metrics)

    # Controlled input-partition experiment. Fixed rank, regularization, ALS blocks and data.
    # Pre-materialize each input; exclude repartition/cache preparation from measured fit time.
    performance = []
    for repeat in range(1, args.performance_repeats+1):
        for partitions in [2, 4, 8]:
            update("performance", repeat=repeat, input_partitions=partitions)
            data_p = trainval.repartition(partitions).cache()
            data_p.count()
            begin = time.perf_counter()
            m = als_for(best["rank"], best["regParam"]).fit(data_p)
            fit_seconds = time.perf_counter()-begin
            performance.append({"repeat": repeat, "input_partitions": partitions,
                                "train_ratings": trainval_count, "rank": best["rank"],
                                "regParam": best["regParam"], "maxIter": 10,
                                "user_blocks": 4, "item_blocks": 4, "fit_seconds": fit_seconds})
            pd.DataFrame(performance).to_csv(out / "tables/performance_raw.csv", index=False)
            m._java_obj.userFactors().unpersist()
            m._java_obj.itemFactors().unpersist()
            data_p.unpersist()
    perf = pd.DataFrame(performance).groupby("input_partitions").fit_seconds.agg(["median", "min", "max", "count"]).reset_index()
    perf.to_csv(out / "tables/performance_summary.csv", index=False)
    env = {"python": sys.version, "spark": spark_version, "java": java_version, "platform": platform.platform(),
        "spark_master": args.master, "driver_memory": spark.sparkContext.getConf().get("spark.driver.memory", "default"),
        "shuffle_partitions": 4, "default_parallelism": spark.sparkContext.defaultParallelism,
        "application_id": spark.sparkContext.applicationId, "cpu_quota_cores": 4, "memory_limit_gib": 16,
        "seed": 42, "elapsed_seconds": time.perf_counter()-started,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "performance_repeats": args.performance_repeats}
    write_json(out / "environment.json", env)
    update("complete", elapsed_seconds=env["elapsed_seconds"])
    spark.stop()


if __name__ == "__main__":
    main()
