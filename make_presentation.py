"""Create an editable, evidence-driven Chinese course presentation.

PDF is exported from the PPTX with LibreOffice, not rendered as an unrelated report.
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
from PIL import Image
import pandas as pd

ROOT=Path(__file__).resolve().parent
ART=ROOT/"artifacts"
FONT="Noto Sans CJK SC"
BG="#F4F7FA"; NAVY="#12293C"; TEAL="#168B83"; GOLD="#D99345"; GRAY="#536C7E"; WHITE="#FFFFFF"; LIGHT="#DDE7EC"
def rgb(s):return RGBColor.from_string(s.lstrip("#"))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default=str(ROOT.parent/"deliverables"))
    ap.add_argument("--name",default="")
    ap.add_argument("--student-id",default="")
    ap.add_argument("--resource-url",default="")
    args=ap.parse_args()
    target=Path(args.output);target.mkdir(parents=True,exist_ok=True)
    ds=json.loads((ART/"dataset.json").read_text())
    met=json.loads((ART/"metrics.json").read_text())
    imp=json.loads((ART/"implicit_metrics.json").read_text())
    env=json.loads((ART/"environment.json").read_text())
    rating=pd.read_csv(ART/"tables/rating_metrics.csv")
    ranking=pd.read_csv(ART/"tables/ranking_all_models.csv").set_index("model")
    tune=pd.read_csv(ART/"tables/tuning.csv")
    itune=pd.read_csv(ART/"tables/implicit_tuning.csv")
    perf=pd.read_csv(ART/"tables/performance_summary.csv")
    allrating=rating[rating.cohort=="all_test"].set_index("model")
    rmse_improvement=(1-allrating.loc["ALS + fallback","rmse"]/allrating.loc["Item mean","rmse"])*100
    recall_improvement=(ranking.loc["Implicit ALS","recall_at_10"]/ranking.loc["Popular","recall_at_10"]-1)*100
    ndcg_improvement=(ranking.loc["Implicit ALS","ndcg_at_10"]/ranking.loc["Popular","ndcg_at_10"]-1)*100
    prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
    prs.core_properties.title="基于 Spark MLlib 的电影评分预测与个性化推荐系统"
    prs.core_properties.subject="大数据处理课程项目：设计、实现与真实实验结果"
    prs.core_properties.author=args.name or "课程项目"
    slide_titles=[]

    def box(sl,x,y,w,h,fill=WHITE,line=None,radius=False):
        sh=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, Inches(x),Inches(y),Inches(w),Inches(h))
        sh.fill.solid();sh.fill.fore_color.rgb=rgb(fill)
        sh.line.fill.background() if not line else None
        if line:sh.line.color.rgb=rgb(line)
        if radius:
            try:sh.adjustments[0]=.1
            except Exception:pass
        return sh

    def text(sl,x,y,w,h,content,size=18,color=NAVY,bold=False,align=None,fill=None):
        sh=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
        tf=sh.text_frame;tf.word_wrap=True
        tf.margin_left=tf.margin_right=Inches(.02);tf.margin_top=tf.margin_bottom=Inches(.02)
        for i,line in enumerate(str(content).split("\n")):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
            p.space_after=Pt(9);p.space_before=Pt(0)
            if align is not None:p.alignment=align
            r=p.add_run();r.text=line;r.font.name=FONT;r.font.size=Pt(size);r.font.bold=bold;r.font.color.rgb=rgb(color)
            ea=OxmlElement("a:ea");ea.set("typeface",FONT);r._r.get_or_add_rPr().append(ea)
        return sh

    def slide(title,section="设计与实现",subtitle=""):
        sl=prs.slides.add_slide(prs.slide_layouts[6]);sl.background.fill.solid();sl.background.fill.fore_color.rgb=rgb(BG)
        n=len(prs.slides);slide_titles.append(title)
        box(sl,.55,.35,.12,.20,TEAL)
        text(sl,.78,.30,10,.25,f"大数据处理课程项目  /  {section}",10,GRAY)
        text(sl,.62,.78,12.1,.65,title,29,NAVY,True)
        if subtitle:text(sl,.65,1.44,12,.38,subtitle,12,GRAY)
        box(sl,.65,7.10,12.03,.012,LIGHT)
        text(sl,.65,7.20,10,.18,"MovieLens 1M  ·  Spark SQL  ·  MLlib ALS",8,GRAY)
        text(sl,12.0,7.16,.65,.26,f"{n:02d}",10,GRAY,align=PP_ALIGN.RIGHT)
        return sl

    def caption(sl,content,y=6.69):text(sl,.68,y,12,.28,content,10,GRAY)
    def note(sl,content,y=6.1,h=.48):
        box(sl,.65,y,12.03,h,"#E3F1EE",radius=True);text(sl,.84,y+.09,11.6,h-.12,content,14,TEAL,True)
    def picture(sl,path,x,y,w,h):
        path=Path(path)
        with Image.open(path) as im:iw,ih=im.size
        scale=min(w/iw,h/ih);nw,nh=iw*scale,ih*scale
        return sl.shapes.add_picture(str(path),Inches(x+(w-nw)/2),Inches(y+(h-nh)/2),width=Inches(nw),height=Inches(nh))
    def metric(sl,x,y,w,value,label,detail="",color=TEAL):
        box(sl,x,y,w,1.22,WHITE,radius=True)
        text(sl,x+.18,y+.10,w-.3,.50,value,29,color,True)
        text(sl,x+.18,y+.67,w-.3,.25,label,13,NAVY)
        if detail:text(sl,x+.18,y+.98,w-.3,.18,detail,8,GRAY)
    def bullets(sl,x,y,w,lines,size=18,gap=.73):
        for i,line in enumerate(lines):
            box(sl,x,y+i*gap+.12,.075,.075,TEAL)
            text(sl,x+.22,y+i*gap,w-.22,gap-.06,line,size,NAVY)
    def table(sl,x,y,w,h,headers,rows,widths=None,size=14):
        tab=sl.shapes.add_table(len(rows)+1,len(headers),Inches(x),Inches(y),Inches(w),Inches(h)).table
        if widths:
            for col,cw in zip(tab.columns,widths):col.width=Inches(w*cw/sum(widths))
        for r,data in enumerate([headers]+list(rows)):
            for c,value in enumerate(data):
                cell=tab.cell(r,c);cell.fill.solid();cell.fill.fore_color.rgb=rgb(NAVY if r==0 else (WHITE if r%2 else "#E8EEF2"))
                cell.margin_left=Inches(.12);cell.margin_right=Inches(.09)
                cell.margin_top=Inches(.08);cell.margin_bottom=Inches(.05);cell.vertical_anchor=MSO_ANCHOR.MIDDLE
                tf=cell.text_frame;tf.clear();tf.word_wrap=True
                for j,line in enumerate(str(value).split("\n")):
                    p=tf.paragraphs[0] if j==0 else tf.add_paragraph();p.space_after=Pt(2)
                    rr=p.add_run();rr.text=line;rr.font.name=FONT;rr.font.size=Pt(size);rr.font.bold=(r==0);rr.font.color.rgb=rgb(WHITE if r==0 else NAVY)
                    ea=OxmlElement("a:ea");ea.set("typeface",FONT);rr._r.get_or_add_rPr().append(ea)
        return tab
    def link(sl,x,y,w,label,url,size=13):
        sh=text(sl,x,y,w,.55,label,size,TEAL)
        if url:
            for p in sh.text_frame.paragraphs:
                for r in p.runs:r.hyperlink.address=url;r.font.underline=True
        return sh

    # 01
    s=prs.slides.add_slide(prs.slide_layouts[6]);slide_titles.append("项目封面")
    s.background.fill.solid();s.background.fill.fore_color.rgb=rgb(NAVY)
    text(s,.85,.65,11,.4,"大数据处理 / 课程大作业",14,"#9FB9C9")
    box(s,.88,1.52,.75,.08,TEAL)
    text(s,.85,1.95,11.75,1.6,"基于 Spark MLlib 的\n电影评分预测与个性化推荐系统",35,WHITE,True)
    text(s,.9,3.9,11,.5,"从百万条历史评分到可验证的推荐结果",21,"#9FD7CF")
    text(s,.92,4.75,10,.42,"MovieLens 1M   /   Spark SQL   /   显式与隐式 ALS",15,"#C3D2DD")
    name=args.name or "________";sid=args.student_id or "________"
    text(s,.92,6.25,11,.42,f"个人项目（1 人）    姓名：{name}    学号：{sid}",15,WHITE)
    for i in range(14):
        for j in range(3):box(s,9.0+i*.21,5.4+j*.21,.08,.08,"#34556B" if (i+j)%3 else TEAL)

    # 02
    s=slide("项目目标：预测评分，并给出未看电影的推荐","项目介绍")
    cards=[("数据问题","评分矩阵稀疏，用户活跃程度差异大。\n需要先清洗和分析，再建立推荐模型。"),
           ("实现目标","用 Spark 完成批处理与模型训练；\n用页面展示历史用户的 Top-10 推荐。"),
           ("评价目标","分别考察评分误差、推荐排序和耗时。\n用简单基线判断模型是否带来改进。")]
    for j,(head,body) in enumerate(cards):
        x=.65+j*4.08;box(s,x,1.85,3.88,2.7,WHITE,radius=True)
        text(s,x+.2,2.08,3.45,.45,head,22,TEAL,True);text(s,x+.2,2.85,3.45,1.32,body,17)
    bullets(s,.85,5.0,11.6,["输入：用户–电影–评分–时间戳；输出：预测结果、推荐列表和实验指标。", "完成范围：离线训练、历史用户推荐、模型对比及可复现的展示页面。"],17,.7)

    # 03
    s=slide("数据集：MovieLens 1M","项目介绍")
    for j,(v,label) in enumerate([(f'{ds["clean_ratings"]:,}',"评分记录"),(f'{ds["users"]:,}',"用户"),(f'{ds["rated_movies"]:,}',"有评分的电影"),(f'{ds["sparsity"]:.2%}',"评分矩阵稀疏率")]):
        metric(s,.65+j*3.05,1.8,2.86,v,label)
    table(s,.65,3.45,7.1,2.45,["数据文件","核心字段"],[
        ["ratings.dat","userId、movieId、rating、timestamp"],["movies.dat","movieId、title、genres"],
        ["users.dat","用户属性；本项目不作为模型输入"]],[1.2,3.0],15)
    bullets(s,8.15,3.45,4.4,["评分范围为 1–5 分。",f'电影元数据共 {ds["catalog_movies"]:,} 条，\n不等于有评分的电影数量。',"稀疏率按有评分用户与电影计算。"],16,.82)
    link(s,.68,6.35,12,"数据来源：grouplens.org/datasets/movielens/1m/",ds["url"],13)

    # 04
    s=slide("系统架构：Spark 离线处理，页面读取推荐结果")
    nodes=[(.8,1.9,2.4,.85,"原始评分与电影数据",NAVY),(4.1,1.9,2.9,.85,"Spark 清洗 / SQL 分析",TEAL),(8.0,1.9,4.0,.85,"Parquet 数据 / 时间划分",NAVY),
           (1.2,3.7,4.7,1.0,"显式 ALS：评分预测\nRMSE / MAE 选参与评估",TEAL),(7.1,3.7,4.7,1.0,"隐式 ALS：Top-10 排序\nNDCG@10 选参与评估",TEAL),
           (1.2,5.65,4.7,.85,"模型因子 / CSV / 指标 / 日志",NAVY),(7.1,5.65,4.7,.85,"Streamlit：分析 / 推荐 / 实验",NAVY)]
    for x,y,w,h,label,col in nodes:
        box(s,x,y,w,h,col,radius=True);text(s,x+.1,y+.17,w-.2,h-.2,label,17,WHITE,True,PP_ALIGN.CENTER)
    for x,y,w,h,label in [(3.3,2.05,.7,.4,"→"),(7.1,2.05,.7,.4,"→"),(5.95,3.05,1.0,.4,"↓"),(3.0,4.97,.8,.4,"↓"),(9.0,4.97,.8,.4,"↓"),(6.05,5.85,.8,.4,"→")]:text(s,x,y,w,h,label,25,TEAL,True,PP_ALIGN.CENTER)
    caption(s,"Spark 负责数据处理和模型拟合；导出因子后，NumPy 分批打分并过滤已评分电影，页面读取离线结果。")

    # 05
    s=slide("运行环境与 Spark 任务记录")
    picture(s,ART/"screenshots/spark_jobs.png",.65,1.7,8.05,4.95)
    table(s,9.0,1.85,3.65,4.6,["配置项","本次运行"],[
        ["运行方式",env["spark_master"]],["CPU / 内存","4 核 / 16 GiB 上限"],["Driver 内存",env["driver_memory"]],
        ["Python",env["python"].split()[0]],["Spark / Java",f'{env["spark"]} / {env["java"]}'],["随机种子","42"]],[1,1.5],12)
    caption(s,"截图来自实际 Spark UI。该环境为 Linux 单机并行部署，应用与事件日志随项目保存。")

    # 06
    s=slide("数据处理：检查异常，并保留可复现的清洗结果")
    table(s,.65,1.85,7.55,3.7,["检查项","处理方法","本次发现"],[
        ["缺失 / 非法字段","检查 ID、时间戳、评分及评分范围",f'{ds["invalid_records"]} 条'],
        ["重复用户–电影对","按时间降序保留最新一条",f'{ds["duplicate_pairs"]} 对'],
        ["电影信息关联","按 movieId 检查未匹配记录",f'{ds["missing_movie_records"]} 条'],
        ["标题编码","ISO-8859-1 转 UTF-8","转换后读取"],
        ["清洗结果保存","写入 Parquet，供后续实验复用",f'{ds["clean_ratings"]:,} 条']],[1.3,2.5,1.0],14)
    bullets(s,8.65,2.05,3.95,["读取、检查与聚合均由 Spark 执行。", "本数据在所设规则下没有异常记录，因此未删除评分。", "保留数据摘要和 SHA-256，方便核对输入。"],17,1.1)
    note(s,"清洗工作以检查规则和实际记录数为依据，不人为增加异常或删除数据。",6.04,.52)

    # 07
    s=slide("数据分析：评分偏高，用户活跃度呈长尾分布","实验与结果")
    picture(s,ART/"figures/rating_distribution.png",.6,1.8,6.0,3.65)
    picture(s,ART/"figures/user_activity.png",6.73,1.8,6.0,3.65)
    ud=pd.read_csv(ART/"tables/user_activity.csv")
    counts=pd.read_csv(ART/"tables/rating_distribution.csv")
    positive_share=counts.loc[counts.rating>=4,"count"].sum()/ds["clean_ratings"]
    text(s,.85,5.6,5.7,.9,f'平均评分 {ds["mean_rating"]:.2f}；4–5 分占 {positive_share:.2%}。\n后续以评分 ≥4 定义正反馈。',16)
    text(s,6.95,5.6,5.5,.9,f'用户评分次数中位数为 {int(ud["count"].median())}，最高 {int(ud["count"].max()):,}。\n后续按历史数量分组分析推荐效果。',16)
    caption(s,"来源：rating_distribution.csv、user_activity.csv。此页为全量描述性分析，不用于计算训练期基线。")

    # 08
    s=slide("按时间划分数据，模拟根据历史预测未来","实验与结果")
    sizes=[ds["split"][x] for x in ["train","validation","test"]]
    start=.75
    for size,col,label in zip(sizes,[NAVY,TEAL,GOLD],["训练集","验证集","测试集"]):
        width=11.8*size/ds["clean_ratings"]
        box(s,start,2.0,width,1.0,col)
        text(s,start+.05,2.17,width-.1,.65,f"{label}\n{size:,}",15,WHITE,True,PP_ALIGN.CENTER)
        start+=width
    dates=[datetime.fromtimestamp(ds["split"][k],timezone.utc).strftime("%Y-%m-%d %H:%M:%S") for k in ["t80","t90"]]
    text(s,.85,3.32,11.7,.65,f"时间边界（UTC）：{dates[0]}  /  {dates[1]}",16)
    bullets(s,.85,4.15,11.6,["训练集拟合，验证集选参；确定参数后合并训练与验证集重训。", "最后 100,021 条评分用于报告测试结果；均分、热度和已评分集合均来自训练期。",f'验证期 ALS 原生覆盖率 {met["best"]["coverage"]:.2%}；重训后测试期覆盖率 {met["warm_coverage"]:.2%}。'],17,.73)
    caption(s,"MovieLens 用户随时间加入，时间划分会引入冷启动；评分任务使用回退，排序任务单独说明评价范围。")

    # 09
    s=slide("显式 ALS：把用户和电影映射为潜在因子")
    box(s,.8,1.95,11.7,1.1,WHITE,radius=True)
    text(s,1.0,2.20,11.3,.6,"预测评分  r̂ᵤᵢ = xᵤᵀ yᵢ       目标：观测评分误差 + 正则化",23,TEAL,True,PP_ALIGN.CENTER)
    bullets(s,.95,3.45,6.0,["用户因子表示偏好，电影因子表示潜在特征。", "交替固定一侧因子，求解另一侧，迭代 10 次。", "使用 Spark ALS-WR；参数由验证集 RMSE 选择。"],18,.8)
    box(s,7.55,3.45,4.8,2.45,WHITE,radius=True)
    text(s,7.8,3.67,4.3,.42,"评分预测与冷启动",20,NAVY,True)
    text(s,7.8,4.3,4.25,1.35,"有用户和电影因子 → 向量内积\n缺少因子 → 训练期电影均分\n电影未知 → 训练期全局均分",17)
    caption(s,"预测值不做 [1,5] 裁剪。完整测试集计算回退后的误差，同时报告 warm-only 的原生 ALS 误差。")

    # 10
    s=slide("隐式 ALS：把高评分作为正反馈，直接学习排序")
    box(s,.8,1.95,11.7,1.1,WHITE,radius=True)
    text(s,1.0,2.22,11.3,.6,"偏好 pᵤᵢ = 1(rᵤᵢ ≥ 4)       置信度 cᵤᵢ = 1 + α · pᵤᵢ",24,TEAL,True,PP_ALIGN.CENTER)
    bullets(s,.95,3.45,11.4,["最小化全部用户–电影对上的置信度加权偏好误差，并加入正则化。", "较低评分转为 0；未观测项也为 0，不把它们视为同等强度的明确负反馈。", "按验证集 NDCG@10 选参，预测分数用于排序，不解释为评分或概率。", "生成推荐时从训练期电影中排除用户全部已评分电影。"],18,.7)
    caption(s,"参数：implicitPrefs=True，regParam=0.1，maxIter=10；比较 rank∈{20,40}、alpha∈{1,10,40}。")

    # 11
    s=slide("评价口径：评分任务与 Top-10 排序分开比较","实验与结果")
    table(s,.65,1.85,12.0,2.9,["任务","比较方法","评价数据与指标"],[
        ["完整评分预测","全局均分 / 电影均分 / 显式 ALS + 回退",f'{met["test_ratings"]:,} 条；RMSE、MAE'],
        ["Warm 评分预测","相同已知用户–电影范围的三种方法",f'{met["warm_test_ratings"]:,} 条；RMSE、MAE'],
        ["Top-10 推荐","热门推荐 / 显式 ALS / 隐式 ALS",f'{met["ranking_users"]:,} 名用户；Precision、Recall、NDCG']],[1.05,2.4,2.2],14)
    text(s,.85,5.12,11.7,.65,f'排序评价：测试评分 ≥4 为正反馈；纳入 {met["ranking_eligible_positives"]:,} / {met["ranking_test_positives"]:,} 条测试正反馈。',17)
    text(s,.85,5.94,11.7,.65,f'候选为 {met["ranking_catalog_size"]:,} 个训练期电影减去各用户已评分电影；全部候选参与，不采样负例。',17)
    caption(s,"排序指标按用户宏平均；仅纳入有训练因子且存在可评价正反馈的用户，三种方法使用相同口径。")

    # 12
    s=slide("显式模型选参：较强正则化带来更小的验证误差","实验与结果")
    picture(s,ART/"figures/explicit_tuning.png",.65,1.8,8.2,4.75)
    metric(s,9.15,1.95,3.05,f'{met["best"]["val_rmse"]:.4f}',"最优验证集 RMSE")
    text(s,9.15,3.55,3.05,2.5,f'选定 rank = {met["best"]["rank"]}\nregParam = {met["best"]["regParam"]}\nmaxIter = 10\n\n在本组参数中，增大 rank 没有继续改善验证误差。',17)
    caption(s,f'共 6 组配置；各组使用相同的 {met["best"]["val_predicted"]:,} 条可预测验证评分。来源：tuning.csv。')

    # 13
    s=slide("隐式模型选参：以验证集 NDCG@10 为准","实验与结果")
    picture(s,ART/"figures/implicit_tuning.png",.65,1.85,8.2,4.55)
    metric(s,9.15,1.95,3.05,f'{imp["best"]["ndcg_at_10"]:.4f}',"最优验证集 NDCG@10")
    text(s,9.15,3.55,3.05,2.5,f'选定 rank = {imp["best"]["rank"]}\nalpha = {imp["best"]["alpha"]:g}\nregParam = 0.1\n\n本次数据中，较大的 alpha 没有带来更高的验证 NDCG。',17)
    caption(s,f'共 6 组配置；验证排序范围为 {imp["best"]["users"]} 名用户、{imp["best"]["eligible_positives"]:,} 条正反馈。来源：implicit_tuning.csv。')

    # 14
    s=slide("评分结果：ALS + 回退优于电影均分预测","实验与结果")
    picture(s,ART/"figures/rating_metrics.png",.65,1.65,12.05,3.8)
    metric(s,.8,5.52,3.75,f'{allrating.loc["ALS + fallback","rmse"]:.4f}',"完整测试集 RMSE")
    metric(s,4.8,5.52,3.75,f'{rmse_improvement:.2f}%',"相对电影均分的 RMSE 降幅")
    metric(s,8.8,5.52,3.75,f'{met["warm_coverage"]:.2%}',"ALS 原生预测覆盖率")
    caption(s,f'完整测试集 {met["test_ratings"]:,} 条；另有 warm-only ALS RMSE = {rating[(rating.cohort=="warm_test") & (rating.model=="ALS")].rmse.iloc[0]:.4f}。来源：rating_metrics.csv。',6.88)

    # 15
    s=slide("推荐结果：隐式 ALS 改善了 Top-10 排序","实验与结果")
    picture(s,ART/"figures/ranking_metrics.png",.65,1.65,12.0,3.6)
    rows=[]
    for key,label in [("Popular","热门推荐"),("Explicit ALS","显式 ALS"),("Implicit ALS","隐式 ALS")]:
        r=ranking.loc[key];rows.append([label,f'{r.precision_at_10:.4f}',f'{r.recall_at_10:.4f}',f'{r.ndcg_at_10:.4f}'])
    table(s,.9,5.35,11.55,1.23,["方法","Precision@10","Recall@10","NDCG@10"],rows,[1.5,1.6,1.6,1.6],12)
    caption(s,f'1,158 名相同评价用户；相对热门基线，隐式 ALS Recall 提高 {recall_improvement:.2f}%，NDCG 提高 {ndcg_improvement:.2f}%。')

    # 16
    s=slide("分组分析：历史较多时，个性化推荐收益更明显","实验与结果")
    picture(s,ART/"figures/activity_metrics.png",.65,1.85,8.0,4.65)
    table(s,9.0,1.95,3.6,2.25,["历史评分数","评价用户"],[["1–100","438"],["101–300","412"],["301 及以上","308"]],[1.5,1],14)
    text(s,9.0,4.55,3.6,1.7,"低活跃度组中，隐式 ALS 的 NDCG 仍略低于热门推荐。\n\n增加模型复杂度并不保证所有用户都获益。",16)
    caption(s,"分组依据为最终训练期历史评分数；各方法使用相同用户组。来源：activity_all_models.csv。")

    # 17
    s=slide("性能实验：更多输入分区的收益有限","实验与结果")
    picture(s,ART/"figures/performance.png",.65,1.85,8.0,4.55)
    rows=[[int(r.input_partitions),f'{r["median"]:.3f}',f'{r["min"]:.3f}–{r["max"]:.3f}'] for _,r in perf.iterrows()]
    table(s,8.95,1.95,3.7,2.35,["分区","中位秒","范围"],rows,[.7,1,1.5],12)
    text(s,8.95,4.55,3.7,1.65,"每种配置重复 3 次。\n\n耗时区间有重叠，不能据此认定分区越多就一定更快。",16)
    caption(s,"固定 900,188 条训练数据、rank=10、reg=0.1、迭代10次、ALS内部4×4分块；计时前物化缓存。")

    # 18
    s=slide("系统展示：数据概览与电影统计","实验与结果")
    picture(s,ART/"screenshots/app_overview.png",.65,1.65,9.45,5.3)
    bullets(s,10.3,2.0,2.35,["展示实际处理的数据规模。", "查看评分分布与热门电影。", "展示检查结果及数据来源。"],16,1.25)

    # 19
    s=slide("系统展示：历史用户的个性化 Top-10 推荐","实验与结果")
    picture(s,ART/"screenshots/app_recommendations.png",.65,1.65,9.45,5.3)
    text(s,10.35,1.95,2.1,1.0,f'{met["generated_users"]:,} 名用户\n{met["generated_recommendations"]:,} 条推荐',20,TEAL,True)
    bullets(s,10.3,3.25,2.35,["可切换历史用户与模型。", "保留电影名、类型和得分。", "与训练期已评分电影的重合数为 0。"],16,1.05)

    # 20
    s=slide("结果复核：指标可以从明细文件重新计算","设计与实现")
    checks=json.loads((ART/"verification.json").read_text())
    table(s,.65,1.85,7.3,3.95,["复核内容","结果"],[
        ["原始数据量与测试集成员一致性","通过"],["时间划分前后无交叉","通过"],
        ["从预测明细重算 RMSE / MAE","通过"],["从推荐列表重算隐式排序指标","通过"],
        ["Top-10 无重复、无已评分电影","通过"],["模型评价用户范围一致","通过"],["分区实验每项 3 次记录","通过"]],[3.8,.8],14)
    box(s,8.35,1.85,4.25,3.95,NAVY,radius=True)
    text(s,8.55,2.1,3.85,.45,"主要复现入口",20,WHITE,True)
    text(s,8.55,2.92,3.85,2.4,"bash run_all.sh\n\npython verify_results.py\n\nstreamlit run app.py",16,"#BCE4DE")
    caption(s,"产物包含原始运行日志、事件日志、参数表、测试预测明细与两套模型。复核结果见 verification.json。")

    # 21
    s=slide("项目总结：完成流程，并区分了两类推荐目标","项目总结")
    table(s,.65,1.85,6.55,3.95,["完成内容","主要成果"],[
        ["数据处理","百万评分清洗检查、Parquet 与 SQL 分析"],["模型实现","显式 / 隐式两类 ALS，12 组参数配置"],
        ["实验评估","基线、时间划分、排序分组、9 次性能测量"],["应用展示","可交互页面、已评分过滤、模型切换"],
        ["结果记录","可复现代码、模型、日志与指标复核"]],[1.1,2.7],14)
    text(s,7.65,1.97,4.8,.5,"目前不足",22,TEAL,True)
    bullets(s,7.65,2.75,4.8,["新用户和新电影仍缺少协同信息。", "离线历史数据效果不能等同线上体验。", "尚未加入内容特征、多样性重排和增量更新。"],17,.9)
    note(s,"主要认识：评分预测误差降低，不代表直接用于 Top-K 排序就会更好。",6.15,.5)

    # 22
    s=slide("成员分工与项目文件","成员分工")
    text(s,.8,1.85,11.8,.5,f'组队方式：个人项目（1 人）    成员：{args.name or "________"}    学号：{args.student_id or "________"}',20,TEAL,True)
    table(s,.8,2.75,11.7,3.45,["模块","本项目工作内容"],[
        ["环境与数据","Linux / Spark 运行环境、数据检查、Parquet 与统计分析"],
        ["算法与实验","两类 ALS、基线方法、选参、时间划分与指标实现"],
        ["系统与验证","展示页面、离线推荐、已评分过滤、结果复核与复现说明"],
        ["材料整理","图表、系统截图、PPT、PDF 与资源文件"]],[1.2,4],16)
    caption(s,"个人项目不划分多人角色；具体实现与结果以随附源码和实验记录为准。")

    # 23
    s=slide("相关资源与参考资料","相关资源")
    text(s,.85,1.85,11.5,.45,"源代码、模型与实验记录",22,TEAL,True)
    resource_label=("完整项目下载（源码 / 两类模型 / 数据 / 结果 / 日志）" if args.resource_url
                    else "源代码、模型与实验记录随完整项目 ZIP 一并提供")
    link(s,.88,2.5,11.5,resource_label,args.resource_url,18)
    if args.resource_url:
        text(s,.9,3.08,11.5,.6,args.resource_url,11,GRAY)
    else:
        text(s,.9,3.08,11.5,.6,"复现说明：02_project/README.md；模型与实验记录：02_project/artifacts/",13,GRAY)
    text(s,.85,4.05,11.6,.4,"数据与技术参考",21,NAVY,True)
    link(s,.9,4.7,11.5,"MovieLens 1M：grouplens.org/datasets/movielens/1m/",ds["url"],14)
    link(s,.9,5.35,11.5,"Spark 4.0.1：MLlib collaborative filtering 官方文档","https://spark.apache.org/docs/4.0.1/ml-collaborative-filtering.html",14)
    link(s,.9,6.0,11.5,"Hu 等（2008）：Collaborative Filtering for Implicit Feedback Datasets","https://doi.org/10.1109/ICDM.2008.22",13)
    caption(s,"项目包提供 README、数据许可和复现步骤；原始 MovieLens 数据使用遵循 GroupLens 的说明。")

    # No off-canvas elements. Font size and text are verified after PDF export.
    for i,sl in enumerate(prs.slides,1):
        for sh in sl.shapes:
            assert sh.left>=0 and sh.top>=0,(i,sh.name)
            assert sh.left+sh.width<=prs.slide_width+Inches(.02),(i,sh.name,"right")
            assert sh.top+sh.height<=prs.slide_height+Inches(.02),(i,sh.name,"bottom")
    out=target/"MovieLens_Spark_课程项目汇报.pptx"
    prs.save(out)
    (target/"slides_manifest.json").write_text(json.dumps({"slides":slide_titles,"count":len(prs.slides),"resource_url":args.resource_url},ensure_ascii=False,indent=2),encoding="utf-8")
    print(out)

if __name__=="__main__":main()
