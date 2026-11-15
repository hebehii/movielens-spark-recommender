"""Independent checks against saved predictions, source data and ranking lists."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from zipfile import ZipFile

root = Path(__file__).resolve().parent
art = root / "artifacts"
dataset = json.loads((art/"dataset.json").read_text())
metrics = json.loads((art/"metrics.json").read_text())
if not (art/"tables/test_predictions.csv").exists():
    raise SystemExit("请先执行 bash run_all.sh 生成测试明细；原始逐条评分不随仓库再分发。")
if not (root/"data/raw/ml-1m/ratings.dat").exists():
    with ZipFile(root/"data/raw/ml-1m.zip") as archive:
        archive.extract("ml-1m/ratings.dat", root/"data/raw")
raw = pd.read_csv(root/"data/raw/ml-1m/ratings.dat",sep="::",engine="python",
                  names=["userId","movieId","rating","timestamp"])
assert len(raw)==dataset["raw_ratings"]
t80,t90 = dataset["split"]["t80"], dataset["split"]["t90"]
actual_test=raw[raw.timestamp>t90].sort_values(["userId","movieId"])
pred=pd.read_csv(art/"tables/test_predictions.csv").sort_values(["userId","movieId"])
assert np.array_equal(actual_test[["userId","movieId","rating"]].to_numpy(),pred[["userId","movieId","rating"]].to_numpy())
assert len(pred)==dataset["split"]["test"]
ratings=pd.read_csv(art/"tables/rating_metrics.csv")
for _,row in ratings.iterrows():
    cohort=pred if row.cohort=="all_test" else pred[pred.prediction.notna()]
    column={"Global mean":"global_mean_prediction","Item mean":"item_mean_prediction",
            "ALS":"prediction","ALS + fallback":"als_with_fallback"}[row.model]
    error=cohort[column].to_numpy()-cohort.rating.to_numpy()
    assert np.isclose(np.sqrt(np.mean(error**2)),row.rmse,atol=1e-7)
    assert np.isclose(np.mean(np.abs(error)),row.mae,atol=1e-7)
train=raw[raw.timestamp<=t90]
train_pairs=set(zip(train.userId,train.movieId))
for name in ["recommendations.csv","implicit_recommendations.csv"]:
    rec=pd.read_csv(art/"tables"/name)
    assert rec.groupby("userId").size().eq(10).all()
    assert not rec.duplicated(["userId","movieId"]).any()
    assert not (set(zip(rec.userId,rec.movieId)) & train_pairs)
    assert rec.title.notna().all()

# Recompute implicit ranking directly from saved rec lists and raw held-out positives.
rec=pd.read_csv(art/"tables/implicit_recommendations.csv")
factors=np.load(art/"models/implicit_factors.npz")
positive=actual_test[(actual_test.rating>=4)&actual_test.userId.isin(factors["user_ids"])&actual_test.movieId.isin(factors["item_ids"])]
truth=positive.groupby("userId").movieId.agg(set).to_dict()
recmap=rec.sort_values(["userId","rank"]).groupby("userId").movieId.agg(list).to_dict()
rows=[]
for uid,relevant in truth.items():
    positions=[j+1 for j,m in enumerate(recmap[uid]) if m in relevant]
    dcg=sum(1/np.log2(p+1) for p in positions)
    ideal=sum(1/np.log2(p+1) for p in range(1,min(10,len(relevant))+1))
    rows.append([len(positions)/10,len(positions)/len(relevant),dcg/ideal])
summary=pd.read_csv(art/"tables/ranking_all_models.csv").set_index("model")
assert np.allclose(np.mean(rows,axis=0),summary.loc["Implicit ALS",["precision_at_10","recall_at_10","ndcg_at_10"]].astype(float).to_numpy(),atol=1e-10)
assert summary.users.nunique()==1
perf=pd.read_csv(art/"tables/performance_raw.csv")
assert len(perf)==9 and perf.groupby("input_partitions").size().eq(3).all()
assert perf.fit_seconds.gt(0).all()
checks={"raw_dataset_count":"PASS","time_split_and_test_membership":"PASS","rating_metrics_recomputed":"PASS",
        "no_seen_or_duplicate_recommendations":"PASS","implicit_ranking_recomputed":"PASS",
        "equal_ranking_user_cohort":"PASS","performance_three_repeats":"PASS"}
(art/"verification.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2))
print(json.dumps(checks,ensure_ascii=False,indent=2))
