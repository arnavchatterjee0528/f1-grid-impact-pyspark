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


def analysis_decomposed_dynamics(res, laps, pits, status_df, from_year=2011):
    """
    Decomposed Position Dynamics Framework (Modern Era, 2011-2024).
    Separates:
      1. Opening-lap position changes (grid vs line 1).
      2. Pit-stop-related position drop and post-pit stabilisation.
      3. Retirement-related position gains (DNFs dropping out).
      4. Normal-running position swaps (track swap proxy between timing lines).
      5. Final grid-to-finish position changes.
    """
    if pits is None:
        raise ValueError("pit_stops table is required for decomposed position dynamics.")

    # 1. Base modern tables (2011-2024)
    res_m = res.filter(F.col("year") >= from_year).select(
        "raceId", "year", "driverId", "grid_clean", "positionOrder", "statusId"
    )
    laps_m = laps.join(res_m.select("raceId", "year").distinct(), "raceId").select(
        "raceId", "year", "driverId", "lap", F.col("position").cast("int").alias("pos")
    )
    pits_m = pits.join(res_m.select("raceId", "year").distinct(), "raceId").select(
        "raceId", "driverId", "stop", "lap", "duration", "milliseconds"
    )

    # ---------------------------------------------------- A. Opening Lap Analysis
    l1 = laps_m.filter(F.col("lap") == 1).select(
        "raceId", "driverId", F.col("pos").alias("lap1_pos")
    )
    op = res_m.join(l1, ["raceId", "driverId"], "left")
    # A missing Lap 1 timing record is not proof of a DNF or a non-start.
    op = op.withColumn("missing_lap1_observation",
                       F.when(F.col("lap1_pos").isNull(), 1).otherwise(0))
    op = op.withColumn(
        "lap1_gain",
        F.when(F.col("lap1_pos").isNotNull(), F.col("grid_clean") - F.col("lap1_pos"))
    )

    op_summary = (
        op.groupBy("year")
          .agg(
              F.count("*").alias("classified_result_entries"),
              F.sum("missing_lap1_observation").alias("missing_lap1_observations"),
              F.count("lap1_pos").alias("lap1_observed_drivers"),
              F.avg("lap1_gain").alias("mean_net_lap1_gain"),
              F.avg(F.abs("lap1_gain")).alias("mean_abs_lap1_places_changed"),
              (F.sum(F.abs("lap1_gain")) / 2.0).alias("estimated_lap1_swaps")
          )
          .orderBy("year")
    )

    # ---------------------------------------------------- B. Pit Stop Impact Analysis
    # Immediate position drop: pre-pit (lap - 1) vs pit lap
    # Stabilised position: post-pit (lap + 2) vs pre-pit
    p_prep = (
        pits_m.join(
            laps_m.select("raceId", "driverId", (F.col("lap") + 1).alias("lap"), F.col("pos").alias("pre_pit_pos")),
            ["raceId", "driverId", "lap"], "left"
        )
        .join(
            laps_m.select("raceId", "driverId", F.col("lap"), F.col("pos").alias("pit_lap_pos")),
            ["raceId", "driverId", "lap"], "left"
        )
        .join(
            laps_m.select("raceId", "driverId", (F.col("lap") - 2).alias("lap"), F.col("pos").alias("stabilised_pos")),
            ["raceId", "driverId", "lap"], "left"
        )
    )
    # Positions lost during transit: pre_pit_pos - pit_lap_pos (negative indicates drop)
    p_prep = p_prep.withColumn("immediate_transit_drop", F.col("pre_pit_pos") - F.col("pit_lap_pos"))
    p_prep = p_prep.withColumn("net_strategic_delta_2laps", F.col("pre_pit_pos") - F.col("stabilised_pos"))

    pit_summary = (
        p_prep.groupBy("stop")
              .agg(
                  F.count("*").alias("pit_events_total"),
                  F.count("immediate_transit_drop").alias("valid_immediate_position_pairs"),
                  F.count("net_strategic_delta_2laps").alias("valid_two_lap_position_pairs"),
                  F.avg("immediate_transit_drop").alias("avg_immediate_transit_drop"),
                  F.avg("net_strategic_delta_2laps").alias("avg_net_delta_after_2laps"),
                  F.avg(F.abs("immediate_transit_drop")).alias("avg_abs_immediate_transit_volatility")
              )
              .filter(F.col("stop") <= 4)
              .orderBy("stop")
    )

    # ---------------------------------------------------- C. Running Laps & Non-finishers
    # Sequence of running laps for lap >= 2
    w_drv = Window.partitionBy("raceId", "driverId").orderBy("lap")
    laps_seq = laps_m.withColumn("prev_pos", F.lag("pos", 1).over(w_drv))
    laps_seq = laps_seq.withColumn("prev_lap", F.lag("lap", 1).over(w_drv))
    # Keep strictly consecutive laps
    running = laps_seq.filter(
        (F.col("lap") >= 2) & (F.col("prev_lap") == (F.col("lap") - 1)) & F.col("prev_pos").isNotNull()
    )

    # Exclude transitions in a race/lap window where ANY driver pits on lap k or k-1.
    pit_laps_window = (
        pits_m.select("raceId", F.col("lap").alias("p_lap"))
              .withColumn("off", F.explode(F.array(F.lit(0), F.lit(1))))
              .select("raceId", (F.col("p_lap") + F.col("off")).alias("lap"))
              .dropDuplicates()
              .withColumn("race_lap_has_pit_event", F.lit(1))
    )
    running = (running.join(pit_laps_window, ["raceId", "lap"], "left")
                      .fillna({"race_lap_has_pit_event": 0}))

    # Broad non-finisher category, not confirmed mechanical/on-track retirements.
    # Status data also includes disqualifications, withdrawals and other outcomes.
    res_status = res_m.join(status_df, "statusId", "left")
    finished_cond = F.col("status").rlike(r"^(Finished|\+\d+ Laps?)$")
    nonfinishers = res_status.filter(~finished_cond).select("raceId", "driverId")

    # Last recorded lap is only an approximate exit point for a non-finisher.
    max_lap_per_nonfinisher = (
        laps_m.join(nonfinishers, ["raceId", "driverId"], "inner")
              .groupBy("raceId", "driverId")
              .agg(F.max("lap").alias("last_recorded_lap"))
    )
    nonfinisher_pos = (
        max_lap_per_nonfinisher.join(
            laps_m.select("raceId", "driverId", "lap", F.col("pos").alias("nonfinisher_pos")),
            (max_lap_per_nonfinisher.raceId == laps_m.raceId) &
            (max_lap_per_nonfinisher.driverId == laps_m.driverId) &
            (max_lap_per_nonfinisher.last_recorded_lap == laps_m.lap),
            "inner"
        )
        .select(
            max_lap_per_nonfinisher.raceId,
            (max_lap_per_nonfinisher.last_recorded_lap + 1).alias("lap"),
            "nonfinisher_pos"
        )
    )

    # Count non-finishers whose last recorded position was ahead
    ret_ahead = (
        running.join(nonfinisher_pos, ["raceId", "lap"], "inner")
               .filter(F.col("prev_pos") > F.col("nonfinisher_pos"))
               .groupBy("raceId", "driverId", "lap")
               .agg(F.count("*").alias("nonfinisher_ahead_count"))
    )
    running = running.join(ret_ahead, ["raceId", "driverId", "lap"], "left").fillna({"nonfinisher_ahead_count": 0})

    # Delta positions gained on timing line
    running = running.withColumn("raw_delta", F.col("prev_pos") - F.col("pos"))
    # Estimated non-finisher-related gain, based on last recorded position
    running = running.withColumn(
        "nonfinisher_gain",
        F.least(
            F.when(F.col("raw_delta") > 0, F.col("raw_delta")).otherwise(F.lit(0)),
            F.col("nonfinisher_ahead_count")
        )
    )

    nonfinisher_summary = (
        running.filter(F.col("nonfinisher_ahead_count") > 0)
               .groupBy("year")
               .agg(
                   F.countDistinct("raceId").alias("races_with_nonfinishers"),
                   F.sum("nonfinisher_gain").alias("estimated_nonfinisher_positions_gained"),
                   F.avg("nonfinisher_gain").alias("avg_estimated_gain_per_active_driver_on_nonfinisher_laps")
               )
               .orderBy("year")
    )

    # ---------------------------------------------------- D. Normal-Running Swaps Proxy
    # This is a position-change proxy, NOT a verified pairwise overtake count.
    # Only retain transitions outside any race/lap window containing a pit stop.
    running = running.withColumn(
        "is_normal_position_change_proxy",
        F.when((F.col("race_lap_has_pit_event") == 0) &
               (F.col("nonfinisher_gain") == 0) &
               (F.col("raw_delta") != 0), 1).otherwise(0)
    )

    normal_position_changes_yearly = (
        running.groupBy("year")
               .agg(
                   F.count("*").alias("active_driver_laps"),
                   F.sum("race_lap_has_pit_event").alias("driver_lap_transitions_in_race_laps_with_pits"),
                   F.sum("nonfinisher_gain").alias("estimated_nonfinisher_gains"),
                   F.sum("is_normal_position_change_proxy").alias("normal_position_change_driver_laps"),
                   F.sum(F.when(F.col("is_normal_position_change_proxy") == 1, F.abs("raw_delta")).otherwise(0)).alias("normal_total_places_changed"),
                   (F.sum(F.when(F.col("is_normal_position_change_proxy") == 1, F.abs("raw_delta")).otherwise(0)) / 2.0).alias("estimated_normal_position_change_units"),
                   (100.0 * (F.sum(F.when(F.col("is_normal_position_change_proxy") == 1, F.abs("raw_delta")).otherwise(0)) / 2.0) / F.count("*")).alias("estimated_position_change_units_per_100_driver_lap_transitions")
               )
               .orderBy("year")
    )

    # ---------------------------------------------------- E. Overall Dynamics Summary
    total_active_laps = running.count()
    total_lap1_observed = int(op.agg(F.count("lap1_pos")).first()[0] or 0)
    total_lap1_swaps = float(op.agg(F.sum(F.abs("lap1_gain")) / 2.0).first()[0] or 0.0)
    total_pit_stops = pits_m.count()
    total_nonfinisher_gains = int(running.agg(F.sum("nonfinisher_gain")).first()[0] or 0)
    total_normal_changes = float(running.filter(F.col("is_normal_position_change_proxy") == 1).agg(F.sum(F.abs("raw_delta")) / 2.0).first()[0] or 0.0)

    summary_rows = [
        ("1_opening_lap_position_change_proxy", total_lap1_swaps, total_lap1_observed,
         round(total_lap1_swaps / total_lap1_observed, 4) if total_lap1_observed else 0.0,
         "Estimated position-change units / driver with Lap 1 observation"),
        ("2_pit_transit_events", float(total_pit_stops), total_active_laps,
         round(float(total_pit_stops) / total_active_laps, 4) if total_active_laps else 0.0,
         "Pit stops / consecutive active driver-lap transitions"),
        ("3_nonfinisher_related_gains", float(total_nonfinisher_gains), total_active_laps,
         round(float(total_nonfinisher_gains) / total_active_laps, 4) if total_active_laps else 0.0,
         "Estimated positions gained / consecutive active driver-lap transitions"),
        ("4_normal_running_position_change_proxy", total_normal_changes, total_active_laps,
         round(total_normal_changes / total_active_laps, 4) if total_active_laps else 0.0,
         "Estimated position-change units / consecutive active driver-lap transitions"),
    ]
    summary_df = res.sql_ctx.createDataFrame(
        summary_rows, ["component", "total_events", "denominator", "rate", "rate_definition"]
    )

    return op_summary, pit_summary, nonfinisher_summary, normal_position_changes_yearly, summary_df


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

    # Fair comparison: evaluate the all-entry RF and a finisher-only RF
    # on exactly the same classified test rows.
    classified_df = df.filter(finishers)
    tr_classified, te_classified = split(classified_df, FEATURES, train_end, test_year)
    metrics["classified_train_rows"] = tr_classified.count()
    metrics["test_rows_classified_only"] = te_classified.count()

    evaluate(
        te_classified.withColumn("prediction", F.col("grid_clean").cast("double")),
        "baseline_grid_equals_finish_common_classified_test"
    )
    evaluate(
        rf.transform(te_classified),
        "random_forest_all_training_common_classified_test"
    )
    classified_rf = timed(
        "train_random_forest_classified_only",
        lambda: rf_model().fit(tr_classified),
        metrics["timings_s"]
    )
    evaluate(
        classified_rf.transform(te_classified),
        "random_forest_classified_training_common_test"
    )

    rf_eval(dfp.filter(finishers), pf, "random_forest_paper_features_classified_only")

    # External result from Urdhwareshe (2025), not trained or reproduced here.
    # The paper is a preprint and uses a different dataset/features/experiment.
    metrics["external_research_benchmark_not_reproduced"] = {
        "source": "Atharva Urdhwareshe (2025), The Use of Machine Learning in Predicting Formula 1 Race Outcomes",
        "doi": "10.20944/preprints202504.1471.v1",
        "status": "preprint; not peer reviewed",
        "model": "TabNet driver model",
        "reported_metrics": {"rmse": 2.87, "mae": 2.17, "r2": 0.75},
        "comparison_note": "Published paper result only; not directly comparable because dataset, features, filtering, and experiment implementation differ."
    }
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


def charts(corr_pd, oi_pd, pit_pd, pred_pd, rmse, out):
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

    # 3. Pit-stop position impact by stop number.
    pit_plot = pit_pd.sort_values("stop")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    vals = pit_plot["avg_immediate_transit_drop"]
    ax.bar(pit_plot["stop"].astype(str), vals,
           color=[ORANGE if v < 0 else BLUE for v in vals], width=0.58)
    ax.axhline(0, color=MUTED, lw=1)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_xlabel("Pit stop number", fontsize=9, color=MUTED)
    ax.set_ylabel("Mean position change (pre-pit minus pit lap)", fontsize=9, color=MUTED)
    _style(ax, "Position change around pit stops",
           "Negative values indicate a worse timing-line position on the pit lap; descriptive association, not causal pit loss")
    _save(fig, out, "chart_pit_stop_impact.png")

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
    # Decomposed Position Dynamics Framework (2011-2024)
    print("\n--- Running Decomposed Position Dynamics Framework ---")
    op_summary, pit_summary, nonfinisher_summary, normal_position_changes_yearly, summary_df = timed(
        "q_decomposed_dynamics",
        lambda: analysis_decomposed_dynamics(res, laps, dfs.get("pit_stops"), dfs["status"]),
        report["timings_s"]
    )
    op_pd = op_summary.toPandas()
    pit_pd = pit_summary.toPandas()
    nonfinisher_pd = nonfinisher_summary.toPandas()
    norm_pd = normal_position_changes_yearly.toPandas()
    sum_pd = summary_df.toPandas()

    op_pd.to_csv(os.path.join(a.out, "opening_lap_analysis.csv"), index=False)
    pit_pd.to_csv(os.path.join(a.out, "pit_stop_impact.csv"), index=False)
    nonfinisher_pd.to_csv(os.path.join(a.out, "nonfinisher_impact.csv"), index=False)
    norm_pd.to_csv(os.path.join(a.out, "normal_running_position_changes.csv"), index=False)
    sum_pd.to_csv(os.path.join(a.out, "position_dynamics_summary.csv"), index=False)

    # Validations & assertions
    assert len(op_pd) > 0, "opening_lap_analysis must not be empty"
    assert len(pit_pd) > 0, "pit_stop_impact must not be empty"
    assert len(nonfinisher_pd) > 0, "nonfinisher_impact must not be empty"
    assert len(norm_pd) > 0, "normal_running_swaps must not be empty"
    assert len(sum_pd) == 4, "position_dynamics_summary must contain exactly 4 components"
    assert op_pd["lap1_observed_drivers"].sum() > 0, "Lap 1 observations must be positive"
    assert norm_pd["active_driver_laps"].sum() > 0, "Active driver laps must be positive"

    # The previous phase_gains.csv was a synthetic combination of unrelated
    # metrics, not a valid race-phase analysis. Remove stale legacy outputs.
    for stale_name in ("phase_gains.csv", "chart_phase_gains.png"):
        stale_path = os.path.join(a.out, stale_name)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    corr.to_csv(os.path.join(a.out, "corr_by_decade.csv"), index=False)
    oi.to_csv(os.path.join(a.out, "overtaking_index.csv"), index=False)

    feats = build_features(res)
    pred = run_models(feats, report)
    pred_pd = pred.select("year", "label", "prediction").toPandas()
    pred_pd.to_csv(os.path.join(a.out, "predictions_2023.csv"), index=False)

    if not a.skip_perf:
        performance(spark, res, a.data, report["timings_s"])

    charts(corr, oi, pit_pd, pred_pd, report["random_forest"]["rmse"], a.out)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    spark.stop()


if __name__ == "__main__":
    main()
