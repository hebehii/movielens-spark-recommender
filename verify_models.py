"""Verify published Spark model files against factor exports; no raw dataset needed."""
from pathlib import Path
import os
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import numpy as np
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession

root=Path(__file__).resolve().parent
spark=(SparkSession.builder.master("local[2]").appName("Verify-Published-ALS-Models")
       .config("spark.ui.enabled","false").config("spark.ui.showConsoleProgress","false")
       .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
results=[]
try:
    for directory,export in [("als","factors.npz"),("implicit_als","implicit_factors.npz")]:
        model=ALSModel.load(str(root/"artifacts/models"/directory))
        saved=np.load(root/"artifacts/models"/export)
        counts={}
        for label,frame in [("user",model.userFactors),("item",model.itemFactors)]:
            factors=frame.toPandas().sort_values("id")
            values=np.stack(factors.features).astype(np.float32)
            assert np.array_equal(factors.id.to_numpy(),saved[label+"_ids"]),directory+" IDs"
            assert np.isfinite(values).all(),directory+" nonfinite weights"
            assert np.allclose(values,saved[label+"_factors"],atol=1e-6),directory+" factor mismatch"
            assert values.std()>0,directory+" constant factors"
            counts[label+"s"]=len(factors)
        results.append({"model":directory,"rank":model.rank,"factor_consistency":"PASS",**counts})
    print(json.dumps(results,ensure_ascii=False,indent=2))
    (root/"artifacts/published_models_verification.json").write_text(json.dumps(results,indent=2))
finally:
    spark.stop()
