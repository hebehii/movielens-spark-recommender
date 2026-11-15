# 基于 Spark MLlib 的电影评分预测与个性化推荐系统

使用 MovieLens 1M 构建批处理推荐流程，包括 Spark 数据清洗、显式 ALS 评分预测、隐式 ALS Top-10 推荐、时间划分评估和 Streamlit 展示。

## 1. 环境

实测：Linux、4 核 CPU 配额、16 GiB 内存上限、Java 21.0.13、Python 3.12.14、Spark 4.0.1，运行方式 `local[4]`，driver memory 为 4g。该部署为单机并行模式。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
java -version
bash run_all.sh
streamlit run app.py
```

浏览器打开 `http://localhost:8501`。首次使用请执行 `bash run_all.sh`，下载官方数据并生成页面所需的历史评分与预测明细。仓库保存已训练模型、汇总实验结果和图表；原始评分、历史评分文件及含真实评分的测试明细不在仓库中再分发。

## 2. 数据与许可

- 数据集：https://grouplens.org/datasets/movielens/1m/
- 原始压缩包：https://files.grouplens.org/datasets/movielens/ml-1m.zip
- 数据的使用和再分发遵循原始 MovieLens README；原许可见 `data/MOVIELENS_README.txt` 及脚本下载后解压的 `data/raw/ml-1m/README`。
- 原始电影标题为 ISO-8859-1 编码，先转换为 UTF-8。评分记录使用 `::` 分隔符。
- SHA-256 摘要记录在 artifacts/dataset.json。原始数据为历史公开数据，不包含本项目作者个人信息。

## 3. 脚本与产物

| 文件 | 作用 |
|---|---|
| run_pipeline.py | Spark 数据清洗、SQL 聚合、显式 ALS 参数实验、评分评估、热门基线、推荐与分区性能实验 |
| run_implicit.py | 隐式反馈构造、6 组参数验证、重训和全候选集 Top-10 评价 |
| make_figures.py | 从结果 CSV 生成实验图表 |
| verify_results.py | 从原始数据与推荐列表重新计算关键指标，并检查时间划分与已评分过滤 |
| app.py | 展示数据概览、历史用户推荐和真实实验结果 |
| artifacts/tables/ | 参数实验、汇总指标、推荐列表和性能原始数据；真实评分明细在复现后生成 |
| artifacts/models/ | 两个 Spark ALS 模型与 NumPy 因子导出 |
| artifacts/eventlog/ | 两次 Spark 应用的压缩事件日志 |
| artifacts/screenshots/ | 实际 Spark UI 与 Streamlit 页面截图 |
| logs/ | 实际运行日志 |

## 4. 实验协议

以全局时间顺序划分训练、验证、测试集，时间戳阈值为 975768738、978133376。边界归入前一段，得到 800,168 / 100,020 / 100,021 条记录。精确划分规则见 dataset.json。

所有均分基线、热门计数、已评分过滤和模型因子仅使用相应训练时间段。全量数据只用于描述性统计。显式模型在训练集拟合，在验证集中可预测的 23,840 条评分上按 RMSE 选参；隐式模型在验证集符合候选条件且至少有一条正反馈的 492 名用户上按宏平均 NDCG@10 选参。参数选定后合并训练和验证集重训，使用最后 100,021 条记录报告测试结果。

显式模型：rank∈{10,20,40}，regParam∈{0.01,0.1}，maxIter=10，seed=42。预测不裁剪到 [1,5]。冷启动评分先回退到训练期电影平均分，未知电影回退到训练期全局均分。

隐式模型：rating≥4 定义 preference=1，否则为 0；confidence=1+alpha*preference。rank∈{20,40}，alpha∈{1,10,40}，regParam=0.1，maxIter=10，seed=42。低分记录的偏好值为 0，不被解释为正反馈。

Top-10 评价只使用最终训练期已知用户、已知电影的测试正反馈。共 1,158 名用户、52,063 条正反馈、3,678 个训练期电影候选；每个用户排除全部训练期已评分电影，使用全部剩余电影而非采样负例。三个方法使用同一用户范围和候选口径。Precision@10、Recall@10、NDCG@10 按用户宏平均；冷启动用户与电影不混入此排序评估。热门基线使用训练期 rating≥4 的计数。

Spark 负责模型训练；导出因子后，以 NumPy 分批矩阵乘法计算离线推荐并做已评分过滤。隐式相关性分数不是概率，显式预测值也不是推荐点击率。

输入分区实验在同一份 900,188 条最终训练数据上，比较 2、4、8 个输入分区，每项 3 次。固定 ALS 用户/物品分块数为 4，rank、正则化、迭代数不变；计时前物化输入缓存，fit 耗时不包含数据重分区与页面加载。结果为单机条件下的中位数及最小–最大范围。

## 5. 本次运行结果

| 任务 | 方法 | 结果 |
|---|---|---|
| 全部测试评分 RMSE | 电影均分 | 0.954259 |
| 全部测试评分 RMSE | 显式 ALS + 冷启动回退 | 0.899629 |
| Warm 测试评分 RMSE | 显式 ALS | 0.895800 |
| Top-10 Recall / NDCG | 热门推荐 | 0.056833 / 0.212281 |
| Top-10 Recall / NDCG | 显式 ALS | 0.012951 / 0.033012 |
| Top-10 Recall / NDCG | 隐式 ALS | 0.076525 / 0.231834 |

显式最优 rank=10、regParam=0.1；隐式最优 rank=40、alpha=1。评分预测误差降低不自动意味着 Top-10 排序更好。显式模型在全候选集排序中弱于热门基线，隐式模型在本次实验中改善了这一点。分组结果也表明隐式模型并非在每个用户活跃度分组都超过热门推荐，应结合表格分析。

结果来源为 artifacts/tables/*.csv，而非示例或手填数据。重新执行可能因运行时调度和库版本产生耗时差异，浮点指标也可能有细微变化。`verify_results.py` 使用完整复现后生成的明细产物做独立复核；`python verify_models.py` 可直接检查仓库中已保存模型与因子导出的一致性，无需下载原始数据。

## 6. 已知不足与后续方向

当前为公开历史数据上的离线评估，未进行在线 A/B 测试；未观察到的电影不能直接认定为用户不喜欢。新用户缺少个性化因子，新电影缺少协同信息。候选集为训练期已知电影；未加入内容特征、多样性重排或增量更新。后续可扩展到内容与协同混合推荐、更大数据集及多节点部署。

## 7. 技术参考

- Spark ML collaborative filtering：https://spark.apache.org/docs/4.0.1/ml-collaborative-filtering.html
- Hu, Koren, Volinsky (2008), Collaborative Filtering for Implicit Feedback Datasets. DOI: https://doi.org/10.1109/ICDM.2008.22
- Harper & Konstan (2015), The MovieLens Datasets: History and Context. DOI: https://doi.org/10.1145/2827872

## 8. 资源下载

- 源代码：本仓库根目录中的 Python 脚本。
- 已训练模型：[artifacts/models](artifacts/models/)。
- 运行日志：[logs](logs/)。
- 汇总实验表：[artifacts/tables](artifacts/tables/)。
- 原始数据：[MovieLens 官方下载](https://grouplens.org/datasets/movielens/1m/)。

MovieLens 原始许可要求在未经单独许可时不再分发数据，因此本仓库通过脚本从官方地址获取原始数据，并排除原始评分的逐条副本。

## 9. 模型 ZIP 下载

[下载显式与隐式 ALS 模型 ZIP](https://github.com/hebehii/movielens-spark-recommender/raw/refs/heads/main/downloads/als-models.zip) · [SHA-256 校验](downloads/checksums.json)。

完整项目可使用 GitHub 的 Code → Download ZIP 获取。原始评分数据仍由复现脚本从官方地址下载。
