"""Plot only measured CSV/JSON artifacts, never hard-code experimental outcomes."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

ART=ROOT/"artifacts"
FIG=ART/"figures"
FIG.mkdir(parents=True,exist_ok=True)
font=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if font.exists():
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"]=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False,
    "axes.edgecolor":"#BAC8D0","axes.labelcolor":"#314B5E","xtick.color":"#314B5E",
    "ytick.color":"#314B5E","text.color":"#12293C","figure.facecolor":"#F4F7FA",
    "axes.facecolor":"#F4F7FA","axes.titleweight":"bold","savefig.facecolor":"#F4F7FA"})
TEAL="#168B83"; NAVY="#223F59"; GOLD="#D99345"
def save(fig,name):
    fig.tight_layout(pad=1.3)
    fig.savefig(FIG/name,dpi=180,bbox_inches="tight")
    plt.close(fig)

dist=pd.read_csv(ART/"tables/rating_distribution.csv")
fig,ax=plt.subplots(figsize=(7.2,4.1))
bars=ax.bar(dist.rating,dist["count"]/1000,color=["#A9BCC7"]*3+[TEAL,TEAL],width=.6)
ax.bar_label(bars,labels=[f"{n:,}" for n in dist["count"]],padding=5,fontsize=10)
ax.set(xlabel="评分",ylabel="记录数量 / 千条",xticks=[1,2,3,4,5],ylim=(0,dist["count"].max()/1000*1.18))
save(fig,"rating_distribution.png")

users=pd.read_csv(ART/"tables/user_activity.csv")
fig,ax=plt.subplots(figsize=(7.2,4.1))
ax.hist(users["count"],bins=np.geomspace(users["count"].min(),users["count"].max()+1,25),color=TEAL,edgecolor="#F4F7FA")
ax.set_xscale("log");ax.set(xlabel="每位用户的评分次数（对数坐标）",ylabel="用户数量")
save(fig,"user_activity.png")

tune=pd.read_csv(ART/"tables/tuning.csv")
fig,ax=plt.subplots(figsize=(8.5,4.4))
for reg,c in [(0.01,GOLD),(0.1,TEAL)]:
    sub=tune[tune.regParam==reg]
    ax.plot(sub["rank"],sub.val_rmse,"o-",lw=2.5,color=c,label=f"regParam = {reg}")
    for _,r in sub.iterrows():ax.annotate(f"{r.val_rmse:.4f}",(r["rank"],r.val_rmse),xytext=(0,9),textcoords="offset points",ha="center",fontsize=10)
ax.set(xlabel="潜在因子数 rank",ylabel="验证集 RMSE（越低越好）",xticks=[10,20,40],ylim=(.84,1.065));ax.legend(frameon=False)
save(fig,"explicit_tuning.png")

path=ART/"tables/implicit_tuning.csv"
if path.exists():
    tune_i=pd.read_csv(path)
    mat=tune_i.pivot(index="rank",columns="alpha",values="ndcg_at_10")
    fig,ax=plt.subplots(figsize=(8.5,4.0))
    im=ax.imshow(mat.to_numpy(),cmap="YlGnBu",aspect="auto")
    for i in range(len(mat)):
        for j in range(len(mat.columns)):
            val=mat.iloc[i,j]
            ax.text(j,i,f"{val:.4f}",ha="center",va="center",fontsize=17,color="white" if val>mat.to_numpy().mean() else NAVY)
    ax.set(xticks=range(len(mat.columns)),xticklabels=[str(int(x)) for x in mat.columns],yticks=range(len(mat)),yticklabels=mat.index,
           xlabel="置信度系数 alpha",ylabel="潜在因子数 rank")
    fig.colorbar(im,ax=ax,label="验证集 NDCG@10（越高越好）")
    save(fig,"implicit_tuning.png")

rating=pd.read_csv(ART/"tables/rating_metrics.csv")
sub=rating[rating.cohort=="all_test"]
fig,axes=plt.subplots(1,2,figsize=(11.6,4.4))
labels=["全局均分","电影均分","ALS + 回退"]
for ax,metric in zip(axes,["rmse","mae"]):
    bars=ax.bar(labels,sub[metric],color=["#A9BCC7",GOLD,TEAL],width=.6)
    ax.bar_label(bars,fmt="%.4f",padding=6,fontsize=12)
    ax.set(ylabel=metric.upper(),ylim=(0,sub[metric].max()*1.2))
save(fig,"rating_metrics.png")

path=ART/"tables/ranking_all_models.csv"
if path.exists():
    rank=pd.read_csv(path).set_index("model").loc[["Popular","Explicit ALS","Implicit ALS"]]
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.4))
    names=["热门推荐","显式 ALS","隐式 ALS"]
    for ax,metric in zip(axes,["recall_at_10","ndcg_at_10"]):
        bars=ax.bar(names,rank[metric],color=[GOLD,"#A9BCC7",TEAL],width=.6)
        ax.bar_label(bars,fmt="%.4f",padding=6,fontsize=12)
        ax.set(ylabel={"recall_at_10":"Recall@10","ndcg_at_10":"NDCG@10"}[metric],ylim=(0,rank[metric].max()*1.25))
    save(fig,"ranking_metrics.png")

perf=pd.read_csv(ART/"tables/performance_summary.csv")
fig,ax=plt.subplots(figsize=(8.5,4.0))
ax.errorbar(perf.input_partitions,perf["median"],yerr=np.vstack([perf["median"]-perf["min"],perf["max"]-perf["median"]]),
            fmt="o-",lw=2.5,capsize=7,color=TEAL)
for _,r in perf.iterrows():ax.annotate(f'{r["median"]:.3f} s',(r.input_partitions,r["median"]),xytext=(0,15),textcoords="offset points",ha="center")
ax.set(xlabel="输入分区数",ylabel="ALS fit 耗时 / 秒",xticks=[2,4,8],ylim=(0,perf["max"].max()*1.3))
ax.text(.02,.04,"每种配置重复 3 次；点为中位数，误差线为最小–最大值",transform=ax.transAxes,fontsize=10)
save(fig,"performance.png")

path=ART/"tables/activity_all_models.csv"
if path.exists():
    seg=pd.read_csv(path)
    fig,ax=plt.subplots(figsize=(8.5,4.3))
    for name,color,label in [("Popular",GOLD,"热门推荐"),("Explicit ALS","#A9BCC7","显式 ALS"),("Implicit ALS",TEAL,"隐式 ALS")]:
        sub=seg[seg.model==name].set_index("activity_group").reindex(["1-100","101-300","301+"])
        ax.plot(range(3),sub.ndcg_at_10,"o-",label=label,color=color,lw=2.4)
    ax.set(xticks=range(3),xticklabels=["1–100 条","101–300 条","301 条及以上"],xlabel="训练期历史评分数量",ylabel="NDCG@10")
    ax.legend(frameon=False)
    save(fig,"activity_metrics.png")
print("Figures:",", ".join(p.name for p in sorted(FIG.glob("*.png"))))
