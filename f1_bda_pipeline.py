"""
BDA Project (CSE557) - Does Qualifying Decide the Race?
Grid-position impact analysis and finishing-position prediction on F1 data (1950-2024) with PySpark.

Team: Arnav Chatterjee, Khuushi Chand Navin  |  Faculty guide: Uzma Sulthana  |  Ramaiah Institute of Technology

Dataset: Formula 1 World Championship (1950-2024), Rohan Rao, Kaggle
         https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
Paper compared: Urdhwareshe (2025), DOI 10.20944/preprints202504.1471.v1

HOW TO RUN (see README.md for Windows / Colab notes)
  1. pip install -r requirements.txt            (Java 17 or 21 must be installed for Spark 4)
  2. Download the Kaggle dataset, unzip the CSVs into ./data
  3. python f1_bda_pipeline.py --data ./data --out ./output
Outputs (CSVs, charts, metrics.json) are written to ./output. Full run: about 3 minutes on 2 cores.

Workflow: Data -> Cleaning -> Processing -> Analysis -> Insights
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

TABLES = ["circuits", "constructors", "drivers", "races", "results", "qualifying",
          "lap_times", "pit_stops", "status", "seasons", "driver_standings",
          "constructor_standings", "constructor_results", "sprint_results"]


def timed(label, fn, log):
    """Run fn(), record wall-clock seconds in log, return fn's result."""
    t0 = time.perf_counter()
    out = fn()
    log[label] = round(time.perf_counter() - t0, 3)
    print(f"[time] {label}: {log[label]} s")
    return out


def timed_median(label, fn, log, runs=5):
    """Run fn() `runs` times after one warm-up; record the median wall-clock seconds."""
    fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    log[label] = round(sorted(ts)[len(ts) // 2], 3)
    print(f"[median of {runs}] {label}: {log[label]} s")


# ---------------------------------------------------------------- 1. DATA (ingestion)
def load_tables(spark, data_dir):
    dfs = {}
    for t in TABLES:
        path = os.path.join(data_dir, f"{t}.csv")
        if not os.path.exists(path):
            print(f"[warn] {t}.csv not found, skipping")
            continue
        # the dataset marks missing values with the literal string \N
        dfs[t] = spark.read.csv(path, header=True, inferSchema=True, nullValue="\\N")
    return dfs


def completeness(df, cols):
    """Percentage of non-null values per column."""
    n = df.count()
    row = df.select([F.count(F.col(c)).alias(c) for c in cols]).collect()[0]
    return {c: round(100.0 * row[c] / n, 2) if n else 0.0 for c in cols}


# ---------------------------------------------------------------- 2. CLEANING
def to_seconds(col):
    """'1:23.456' -> 83.456 ; null/blank -> null."""
    parts = F.split(col, ":")
    return F.when(col.isNull() | (F.trim(col) == ""), None).otherwise(
        F.when(F.size(parts) == 2,
               parts.getItem(0).cast("double") * 60 + parts.getItem(1).cast("double"))
         .otherwise(col.cast("double")))


def clean(dfs, quality):
    res = dfs["results"]
    quality["results_rows_raw"] = res.count()
    quality["results_completeness_raw_%"] = completeness(res, ["grid", "position", "positionOrder", "milliseconds", "fastestLap"])

    # duplicates
    # (raceId, driverId) repeats are 1950s-60s shared drives (one driver, two cars); keep the better finish
    res = (res.withColumn("_rk", F.row_number().over(Window.partitionBy("raceId", "driverId").orderBy("positionOrder")))
              .filter(F.col("_rk") == 1).drop("_rk"))
    laps = dfs["lap_times"].dropDuplicates(["raceId", "driverId", "lap"])
    quality["results_rows_after_dedup"] = res.count()
    quality["duplicate_rows_removed"] = quality["results_rows_raw"] - quality["results_rows_after_dedup"]
    quality["lap_rows"] = laps.count()

    # type conversion
    res = (res.withColumn("grid", F.col("grid").cast("int"))
              .withColumn("positionOrder", F.col("positionOrder").cast("int"))
              .withColumn("points", F.col("points").cast("double")))

    # drop rows with no grid / finish (same rule as the paper)
    res = res.filter(F.col("grid").isNotNull() & F.col("positionOrder").isNotNull())

    # grid = 0 means pit-lane start -> put at back of that race's grid
    w_race = Window.partitionBy("raceId")
    res = res.withColumn("grid_clean",
                         F.when(F.col("grid") == 0, F.max("grid").over(w_race) + 1)
                          .otherwise(F.col("grid")))

    # finished flag from status table ('Finished' or '+n Lap(s)')
    status = dfs["status"]
    res = (res.join(status, "statusId", "left")
              .withColumn("finished", F.when(F.col("status").rlike("^(Finished|\\+\\d+ Laps?)$"), 1).otherwise(0)))

    races = (dfs["races"].select("raceId", "year", "round", "circuitId", "date")
                          .withColumn("date", F.to_date("date"))
                          .withColumn("decade", (F.floor(F.col("year") / 10) * 10).cast("int")))
    circuits = dfs["circuits"].select("circuitId", F.col("name").alias("circuit"), "country")
    res = res.join(races, "raceId").join(circuits, "circuitId", "left")

    # qualifying: best of q1/q2/q3 in seconds, gap to pole in %
    q = dfs["qualifying"]
    q = (q.withColumn("q1s", to_seconds(F.col("q1").cast("string")))
          .withColumn("q2s", to_seconds(F.col("q2").cast("string")))
          .withColumn("q3s", to_seconds(F.col("q3").cast("string")))
          .withColumn("best_q", F.least("q1s", "q2s", "q3s"))
          .dropDuplicates(["raceId", "driverId"]))
    q = q.withColumn("pole_q", F.min("best_q").over(Window.partitionBy("raceId")))
    q = q.withColumn("quali_gap_pct", 100.0 * (F.col("best_q") - F.col("pole_q")) / F.col("pole_q"))
    res = res.join(q.select("raceId", "driverId", "quali_gap_pct"), ["raceId", "driverId"], "left")
    res = res.withColumn("quali_missing", F.when(F.col("quali_gap_pct").isNull(), 1).otherwise(0))

    res = res.withColumn("positions_gained", F.col("grid_clean") - F.col("positionOrder"))
    quality["results_rows_clean"] = res.count()
    quality["results_completeness_clean_%"] = completeness(res, ["grid_clean", "positionOrder", "quali_gap_pct", "year", "circuit"])
    return res, laps


# ---------------------------------------------------------------- 3/4. PROCESSING + ANALYSIS
def analysis_corr_by_decade(res):
    return (res.groupBy("decade")
               .agg(F.corr("grid_clean", "positionOrder").alias("grid_finish_corr"),
                    F.avg(F.when(F.col("grid_clean") == 1, F.when(F.col("positionOrder") == 1, 1).otherwise(0))).alias("pole_win_rate"),
                    F.countDistinct("raceId").alias("races"))
               .orderBy("decade"))


def analysis_overtaking_index(res, min_races=5, from_year=2010):
    """Modern era only: grid sizes and reliability in the 1950s-80s make older races incomparable."""
    fin = res.filter((F.col("finished") == 1) & (F.col("year") >= from_year) & (F.col("grid") > 0))
    return (fin.groupBy("circuit", "country")
               .agg(F.avg(F.abs("positions_gained")).alias("overtaking_index"),
                    F.countDistinct("raceId").alias("races"))
               .filter(F.col("races") >= min_races)
               .orderBy(F.desc("overtaking_index")))


def analysis_phase_gains(res, laps, pits):
    """Average positions gained per driver-race in each race phase (1996+ lap data)."""
    base = res.select("raceId", "driverId", "grid_clean")
    lp = laps.select("raceId", "driverId", "lap", F.col("position").cast("int").alias("pos")).join(base, ["raceId", "driverId"])
    w = Window.partitionBy("raceId", "driverId").orderBy("lap")
    lp = lp.withColumn("prev_pos", F.coalesce(F.lag("pos").over(w), F.col("grid_clean")))
    lp = lp.withColumn("delta", F.col("prev_pos") - F.col("pos"))
    lp = lp.withColumn("race_laps", F.max("lap").over(Window.partitionBy("raceId")))
    if pits is not None:
        p = pits.select("raceId", "driverId", F.col("lap").alias("pit_lap"))
        # a pit cycle = pit lap and the 2 laps after it
        cyc = (p.withColumn("off", F.explode(F.array(F.lit(0), F.lit(1), F.lit(2))))
                .select("raceId", "driverId", (F.col("pit_lap") + F.col("off")).alias("lap"))
                .dropDuplicates().withColumn("in_pit_cycle", F.lit(1)))
        lp = lp.join(cyc, ["raceId", "driverId", "lap"], "left").fillna({"in_pit_cycle": 0})
    else:
        lp = lp.withColumn("in_pit_cycle", F.lit(0))
    lp = lp.withColumn("phase",
                       F.when(F.col("lap") == 1, "1_start (lap 1)")
                        .when(F.col("in_pit_cycle") == 1, "2_pit cycle")
                        .when(F.col("lap") > F.col("race_laps") * 2 / 3, "4_final third")
                        .otherwise("3_middle (no pit)"))
    per_driver = lp.groupBy("raceId", "driverId", "phase").agg(F.sum("delta").alias("gain"))
    n_dr = lp.select("raceId", "driverId").distinct().count()
    return (per_driver.groupBy("phase")
                      .agg((F.sum(F.abs("gain")) / F.lit(n_dr)).alias("avg_abs_positions_changed_per_driver_race"),
                           (F.sum("gain") / F.lit(n_dr)).alias("avg_net_gain_per_driver_race"))
                      .orderBy("phase"))


# ---------------------------------------------------------------- ML (comparison with paper)
def build_features(res, train_end=2022):
    df = res.filter((F.col("year") >= 2010) & (F.col("year") <= 2023))
    # driver form: avg finish over previous 5 races (no leakage: rows -5..-1)
    wd = Window.partitionBy("driverId").orderBy("date", "raceId").rowsBetween(-5, -1)
    df = df.withColumn("driver_form5", F.avg("positionOrder").over(wd))
    # constructor form: per-race team avg finish, then previous 5 races
    team = df.groupBy("constructorId", "raceId", "date").agg(F.avg("positionOrder").alias("team_avg"))
    wc = Window.partitionBy("constructorId").orderBy("date", "raceId").rowsBetween(-5, -1)
    team = team.withColumn("constructor_form5", F.avg("team_avg").over(wc)).select("constructorId", "raceId", "constructor_form5")
    df = df.join(team, ["constructorId", "raceId"], "left")
    # circuit overtaking index from TRAINING years only (no leakage into 2023)
    oi = (res.filter((F.col("year") >= 2010) & (F.col("year") <= train_end) & (F.col("finished") == 1) & (F.col("grid") > 0))
             .groupBy("circuitId").agg(F.avg(F.abs("positions_gained")).alias("circuit_oi")))
    df = df.join(oi, "circuitId", "left")
    # fill nulls (start of career / new circuit) with neutral values
    df = df.fillna({"driver_form5": 10.5, "constructor_form5": 10.5, "circuit_oi": 3.0, "quali_gap_pct": 5.0})
    return df


FEATURES = ["grid_clean", "quali_gap_pct", "quali_missing", "driver_form5", "constructor_form5", "circuit_oi", "round"]


def split(frame, cols, train_end=2022, test_year=2023):
    """Assemble features and split by season. The training set is put in one partition in a fixed
    (raceId, driverId) order: Spark's Random Forest bootstrap depends on row order, so this makes
    results identical between this script and the notebook, run after run."""
    d = VectorAssembler(inputCols=cols, outputCol="features").transform(frame).select(
        "raceId", "driverId", "year", "grid_clean", "features", F.col("positionOrder").cast("double").alias("label"))
    train = d.filter(F.col("year") <= train_end).coalesce(1).sortWithinPartitions("raceId", "driverId").cache()
    test = d.filter(F.col("year") == test_year).cache()
    return train, test


def rf_model():
    return RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=200, maxDepth=8, seed=42)


def run_models(df, metrics, train_end=2022, test_year=2023):
    train, test = split(df, FEATURES, train_end, test_year)
    metrics["train_rows"], metrics["test_rows"] = train.count(), test.count()

    def evaluate(pred, name):
        out = {}
        for m in ["rmse", "mae", "r2"]:
            out[m] = round(RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName=m).evaluate(pred), 3)
        metrics[name] = out
        print(name, out)
        return pred

    # baseline: predicted finish = grid position
    evaluate(test.withColumn("prediction", F.col("grid_clean").cast("double")), "baseline_grid_equals_finish")

    rf = timed("train_random_forest", lambda: rf_model().fit(train), metrics.setdefault("timings_s", {}))
    rf_pred = evaluate(rf.transform(test), "random_forest")
    metrics["rf_feature_importance"] = {f: round(float(v), 4) for f, v in zip(FEATURES, rf.featureImportances.toArray())}

    gbt = GBTRegressor(featuresCol="features", labelCol="label", maxIter=100, maxDepth=5, seed=42)
    gbt_model = timed("train_gbt", lambda: gbt.fit(train), metrics["timings_s"])
    evaluate(gbt_model.transform(test), "gbt")

    # Replication checks against the paper:
    # (1) the paper's inputs, incl. 'laps' completed, which is only known AFTER the race
    # (2) the paper's filter: rows with no finish position (non-classified drivers) removed
    pf = ["grid_clean", "laps", "constructorId", "round"]
    dfp = df.withColumn("laps", F.col("laps").cast("double")).withColumn("constructorId", F.col("constructorId").cast("double"))
    finishers = F.col("position").isNotNull()

    def rf_eval(frame, cols, name):
        tr, te = split(frame, cols, train_end, test_year)
        evaluate(rf_model().fit(tr).transform(te), name)

    rf_eval(dfp, pf, "random_forest_paper_features_incl_laps")
    rf_eval(df.filter(finishers), FEATURES, "random_forest_classified_only")
    rf_eval(dfp.filter(finishers), pf, "random_forest_paper_features_classified_only")
    metrics["test_rows_classified_only"] = df.filter(finishers & (F.col("year") == test_year)).count()

    metrics["paper_tabnet_driver_model"] = {"rmse": 2.87, "mae": 2.17, "r2": 0.75}
    return rf_pred


# ---------------------------------------------------------------- PERFORMANCE
def performance(spark, res, data_dir, perf):
    """Every timing = median of 5 runs after a warm-up run."""
    # Spark: overtaking-index query on an uncached vs a cached DataFrame
    res.unpersist()
    timed_median("spark_overtaking_index_uncached", lambda: analysis_overtaking_index(res).collect(), perf)
    res.cache().count()
    timed_median("spark_overtaking_index_cached", lambda: analysis_overtaking_index(res).collect(), perf)

    # Spark: lap-level aggregation (589k rows) at different partition counts
    laps = spark.read.csv(os.path.join(data_dir, "lap_times.csv"), header=True, inferSchema=True, nullValue="\\N")
    for p in [1, 2, 4, 8]:
        lp = laps.repartition(p).cache()
        lp.count()
        timed_median(f"spark_lap_agg_{p}_partitions",
                     lambda: lp.groupBy("raceId", "driverId").agg(F.avg("milliseconds"), F.min("position"), F.max("position")).count(), perf)
        lp.unpersist()

    # pandas baseline vs Spark: same read + group-by on one machine
    def pandas_lap():
        pdf = pd.read_csv(os.path.join(data_dir, "lap_times.csv"), na_values="\\N")
        return len(pdf.groupby(["raceId", "driverId"]).agg({"milliseconds": "mean", "position": ["min", "max"]}))
    timed_median("pandas_lap_agg_incl_read", pandas_lap, perf)

    def spark_lap():
        l2 = spark.read.csv(os.path.join(data_dir, "lap_times.csv"), header=True, inferSchema=True, nullValue="\\N")
        return l2.groupBy("raceId", "driverId").agg(F.avg("milliseconds"), F.min("position"), F.max("position")).count()
    timed_median("spark_lap_agg_incl_read", spark_lap, perf)

    # correctness check: Spark and pandas must produce the same number of (race, driver) groups
    perf["check_groups_spark"] = spark_lap()
    perf["check_groups_pandas"] = pandas_lap()


# ---------------------------------------------------------------- 5. INSIGHTS (charts)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_axisbelow(True)


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=200, facecolor="white")
    plt.close(fig)


def charts(corr_pd, oi_pd, phase_pd, pred_pd, rmse, out):
    # 1. grid-finish correlation by decade
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x, y = corr_pd["decade"], corr_pd["grid_finish_corr"]
    ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=6, markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(f"{y.iloc[0]:.2f}", (x.iloc[0], y.iloc[0]), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.annotate(f"{y.iloc[-1]:.2f}", (x.iloc[-1], y.iloc[-1]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{d}s" for d in x])
    ax.set_ylim(0.4, 0.7); ax.yaxis.grid(True, color=GRID, lw=0.8)
    _style(ax, f"Grid position decides more than ever: r = {y.iloc[0]:.2f} in the 1950s, {y.iloc[-1]:.2f} in the 2020s",
           "Pearson correlation between starting grid slot and final finishing order, all entries incl. retirements")
    _save(fig, out, "chart_corr_by_decade.png")

    # 2. circuit overtaking index (2010-2024), every circuit with >= 5 races, hardest highlighted
    o = oi_pd.sort_values("overtaking_index")
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    cols = [ORANGE if i < 3 else BLUE for i in range(len(o))]
    ax.barh(o["circuit"], o["overtaking_index"], color=cols, height=0.7)
    ax.invert_yaxis(); ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_xlabel("Average places changed, grid to finish (finishers only)", fontsize=9, color=MUTED)
    c3 = list(o["country"].head(3))
    _style(ax, f"Hardest tracks to gain places: {c3[0]}, {c3[1]} and {c3[2]}", "Circuit overtaking index, 2010–2024, circuits with at least 5 races (orange = 3 lowest)")
    _save(fig, out, "chart_overtaking_index.png")

    # 3. net places gained per race phase (diverging about zero)
    labels = {"1_start (lap 1)": "Lap 1", "2_pit cycle": "Pit-stop laps", "3_middle (no pit)": "Middle stint", "4_final third": "Final third"}
    ph = phase_pd.copy(); ph["label"] = ph["phase"].map(labels)
    v = ph["avg_net_gain_per_driver_race"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(ph["label"], v, color=[BLUE if a >= 0 else ORANGE for a in v], width=0.55)
    ax.axhline(0, color=MUTED, lw=1)
    for i, a in enumerate(v):
        ax.text(i, a + (0.15 if a >= 0 else -0.15), f"{a:+.1f}", ha="center", va="bottom" if a >= 0 else "top", fontsize=9, color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_ylabel("Net places gained per driver", fontsize=9, color=MUTED)
    ax.set_ylim(min(v) - 1, max(v) + 1)
    _style(ax, f"Drivers drop {abs(v.iloc[1]):.1f} places around pit stops and win {v.iloc[2]:.1f} back mid-race",
           "Avg net places gained per driver per race, 1996–2024 lap data (pit-stop laps identified from 2011)")
    _save(fig, out, "chart_phase_gains.png")

    # 4. predicted vs actual, 2023 test season
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([1, 22], [1, 22], ls="--", color=MUTED, lw=1)
    ax.scatter(pred_pd["label"], pred_pd["prediction"], s=22, color=BLUE, alpha=0.45, edgecolors="white", linewidths=0.5)
    ax.set_xlim(0, 23); ax.set_ylim(0, 23); ax.grid(True, color=GRID, lw=0.8)
    ax.set_xlabel("Actual finishing position", fontsize=9, color=MUTED); ax.set_ylabel("Predicted", fontsize=9, color=MUTED)
    ax.spines["left"].set_visible(True); ax.spines["left"].set_color(MUTED)
    _style(ax, f"Random Forest, 2023 season: RMSE {rmse:.2f}", "Each dot is one driver in one race; dashed line = perfect prediction")
    _save(fig, out, "chart_pred_vs_actual.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./output")
    ap.add_argument("--skip-perf", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    spark = (SparkSession.builder.appName("F1-BDA-CSE557")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.sql.execution.arrow.pyspark.enabled", "false")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    report = {"timings_s": {}, "quality": {}}

    dfs = timed("ingest_14_tables", lambda: load_tables(spark, a.data), report["timings_s"])
    res, laps = timed("clean_and_join", lambda: clean(dfs, report["quality"]), report["timings_s"])
    res = res.cache()

    corr = timed("q_corr_by_decade", lambda: analysis_corr_by_decade(res).toPandas(), report["timings_s"])
    oi = timed("q_overtaking_index", lambda: analysis_overtaking_index(res).toPandas(), report["timings_s"])
    phase = timed("q_phase_gains", lambda: analysis_phase_gains(res, laps, dfs.get("pit_stops")).toPandas(), report["timings_s"])
    corr.to_csv(os.path.join(a.out, "corr_by_decade.csv"), index=False)
    oi.to_csv(os.path.join(a.out, "overtaking_index.csv"), index=False)
    phase.to_csv(os.path.join(a.out, "phase_gains.csv"), index=False)

    feats = build_features(res)
    pred = run_models(feats, report)
    pred_pd = pred.select("year", "label", "prediction").toPandas()
    pred_pd.to_csv(os.path.join(a.out, "predictions_2023.csv"), index=False)

    if not a.skip_perf:
        performance(spark, res, a.data, report["timings_s"])

    charts(corr, oi, phase, pred_pd, report["random_forest"]["rmse"], a.out)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    spark.stop()


if __name__ == "__main__":
    main()
