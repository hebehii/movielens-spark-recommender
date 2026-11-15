"""Read-only demo of actual offline Spark results. Run: streamlit run app.py"""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
st.set_page_config(page_title="MovieLens · Spark 推荐实验", page_icon="🎬", layout="wide")
st.markdown("""<style>
.block-container{padding-top:4rem;max-width:1400px}
h1{letter-spacing:-.04em} [data-testid="stMetric"]{background:#132735;padding:16px;border-radius:12px}
[data-testid="stMetricValue"]{color:#58d7c0} .eyebrow{color:#58d7c0;font-size:13px;letter-spacing:3px}
</style>""", unsafe_allow_html=True)

required = [ART/"metrics.json", ART/"tables/history.parquet", ART/"tables/recommendations.csv"]
if not all(p.exists() for p in required):
    st.info("首次运行请先执行 bash run_all.sh，从官方地址下载 MovieLens 并生成页面所需数据。")
    st.markdown("[MovieLens 1M 官方数据集](https://grouplens.org/datasets/movielens/1m/)")
    st.stop()

@st.cache_data
def load():
    ds = json.loads((ART/"dataset.json").read_text())
    met = json.loads((ART/"metrics.json").read_text())
    history = pd.read_parquet(ART/"tables/history.parquet")
    explicit = pd.read_csv(ART/"tables/recommendations.csv")
    implicit_path = ART/"tables/implicit_recommendations.csv"
    implicit = pd.read_csv(implicit_path) if implicit_path.exists() else None
    return ds, met, history, explicit, implicit

ds, met, history, explicit, implicit = load()
st.markdown('<div class="eyebrow">MOVIELENS / SPARK MLLIB</div>', unsafe_allow_html=True)
st.title("从百万条评分，到你的下一部电影")
st.caption("Spark SQL 数据处理 · 显式 ALS 评分预测 · 隐式 ALS 个性化推荐")
page = st.sidebar.radio("项目模块", ["数据概览", "个性化推荐", "实验对比"])
st.sidebar.caption("MovieLens 1M · 历史数据离线实验")
st.sidebar.markdown("[数据集来源](https://grouplens.org/datasets/movielens/1m/)")

if page == "数据概览":
    cols = st.columns(4)
    for c, label, value in zip(cols,["评分记录","用户数量","有评分电影","矩阵稀疏率"],
            [f'{ds["clean_ratings"]:,}',f'{ds["users"]:,}',f'{ds["rated_movies"]:,}',f'{ds["sparsity"]:.2%}']):
        c.metric(label,value)
    left,right=st.columns(2)
    with left:
        st.subheader("评分分布")
        counts=pd.read_csv(ART/"tables/rating_distribution.csv").set_index("rating")
        st.bar_chart(counts,color="#58d7c0")
    with right:
        st.subheader("评分次数最多的电影")
        stats=pd.read_csv(ART/"tables/movie_statistics.csv").nlargest(10,"count")
        st.dataframe(stats[["title","count","mean_rating"]].rename(columns={"title":"电影","count":"评分次数","mean_rating":"平均分"}),hide_index=True,
                     column_config={"平均分":st.column_config.NumberColumn(format="%.2f")},width="stretch")
    st.subheader("数据处理结果")
    st.write(f'原始 {ds["raw_ratings"]:,} 条；无效记录 {ds["invalid_records"]} 条；重复用户–电影对 {ds["duplicate_pairs"]} 对；无法关联电影的记录 {ds["missing_movie_records"]} 条。')
    st.caption("全量数据用于描述性统计；模型、热门榜单和评分基线只使用对应训练时间段的数据。")

elif page == "个性化推荐":
    options = ["隐式 ALS（推荐排序）", "显式 ALS（预测评分）"] if implicit is not None else ["显式 ALS（预测评分）"]
    c1,c2=st.columns([1,2])
    users=sorted(explicit.userId.unique().tolist())
    with c1:
        uid=st.selectbox("选择历史用户 ID",users,index=users.index(1) if 1 in users else 0)
    with c2:
        method=st.selectbox("推荐方法",options)
    hist=history[history.userId==uid].sort_values(["rating","timestamp"],ascending=[False,False])
    st.caption(f"用户 {uid} · 训练期历史 {len(hist)} 条 · 推荐列表已过滤训练期评分过的电影")
    left,right=st.columns([1,1.5])
    with left:
        st.subheader("历史高分电影")
        st.dataframe(hist[["title","rating","genres"]].head(12).rename(columns={"title":"电影","rating":"评分","genres":"类型"}),hide_index=True,width="stretch")
    with right:
        st.subheader("Top-10 推荐")
        recs=(implicit if method.startswith("隐式") else explicit)
        recs=recs[recs.userId==uid].sort_values("rank")
        score="score" if method.startswith("隐式") else "prediction"
        label="相关性得分" if score=="score" else "预测评分"
        st.dataframe(recs[["rank","title","genres",score]].rename(columns={"rank":"排名","title":"电影","genres":"类型",score:label}),
            hide_index=True,width="stretch",column_config={label:st.column_config.NumberColumn(format="%.3f")})
    st.caption("隐式得分用于排序，不是喜爱概率；显式预测评分为未裁剪模型输出。页面读取离线模型结果，不会重新训练。")

else:
    rating=pd.read_csv(ART/"tables/rating_metrics.csv")
    all_test=rating[rating.cohort=="all_test"]
    baseline=float(all_test.loc[all_test.model=="Item mean","rmse"].iloc[0])
    final=float(all_test.loc[all_test.model=="ALS + fallback","rmse"].iloc[0])
    c1,c2,c3=st.columns(3)
    c1.metric("ALS + 冷启动回退 RMSE",f"{final:.4f}")
    c2.metric("相对电影均分 RMSE 降幅",f"{(baseline-final)/baseline:.2%}")
    c3.metric("ALS 原生预测覆盖率",f'{met["warm_coverage"]:.2%}')
    st.subheader("评分预测 / 全部测试评分")
    st.dataframe(all_test[["model","rmse","mae","n"]],hide_index=True,width="stretch")
    path=ART/"tables/ranking_all_models.csv"
    if path.exists():
        ranking=pd.read_csv(path)
        st.subheader("Top-10 推荐 / 全候选集评价")
        st.dataframe(ranking,hide_index=True,width="stretch")
        st.bar_chart(ranking.set_index("model")[["recall_at_10","ndcg_at_10"]],color=["#58d7c0","#f5aa66"])
    st.subheader("显式 ALS 参数实验")
    st.dataframe(pd.read_csv(ART/"tables/tuning.csv"),hide_index=True,width="stretch")
    st.caption("参数在验证集选择；训练+验证集重训后，在按时间保留的测试集报告结果。冷启动评分回退为训练期电影均分，未知电影再回退为全局均分。")
