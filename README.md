# Does Qualifying Decide the Race?

**Grid-position impact analysis and finishing-position prediction on 75 years of Formula 1 data, using PySpark.**

Big Data Analytics (CSE557) project, Department of Computer Science and Engineering, Ramaiah Institute of Technology, 2026.
Team: Arnav Chatterjee, Khuushi Chand Navin. Faculty guide: Uzma Sulthana.

The pipeline joins 14 relational tables (26,759 race results and 589,081 lap records, 1950–2024) in Apache Spark. It then measures how much grid position decides a race, which circuits are hardest to overtake at, and where in a race places change hands. Finally, it trains Spark MLlib models to predict finishing position and compares them with a 2025 paper ([Urdhwareshe, DOI 10.20944/preprints202504.1471.v1](https://www.preprints.org/manuscript/202504.1471)).

## Key findings

| Question | Result |
| --- | --- |
| Does qualifying matter more now? | Grid–finish correlation rose from 0.48 (1950s) to 0.62 (2010s and 2020s); pole sitters won 42% of races in the 1950s and 52% in the 2020s |
| Where do places change? | Drivers lose a net 2.6 places on pit-stop laps and gain 3.1 back in the middle stint; lap 1 nets +0.3 |
| Hardest circuits to overtake at (2010–2024) | Mexico City (2.54), Barcelona (2.61), Monaco (2.62); easiest: Sepang (3.84), Spa (3.43), Interlagos (3.31) |
| Can we predict the finish? | Random Forest cuts RMSE by 16% versus "finish = grid"; grid (0.34), team form (0.24) and qualifying gap (0.20) matter most |

![Grid-finish correlation by decade](output/chart_corr_by_decade.png)
![Net places gained by race phase](output/chart_phase_gains.png)
![Circuit overtaking index](output/chart_overtaking_index.png)

## Comparison with the paper

The paper reports RMSE 2.87 for its TabNet driver model on the 2023 season. Our pre-race model scores 4.33 on the same season, but the two setups differ. We tested each difference using the same split (train 2010–2022, test 2023):

| Model | Test rows | RMSE | MAE | R² |
| --- | --- | --- | --- | --- |
| Baseline: finish = grid | all | 5.15 | 3.70 | 0.20 |
| Random Forest, pre-race features (ours) | all | 4.33 | 3.35 | 0.44 |
| GBT, pre-race features (ours) | all | 4.67 | 3.59 | 0.34 |
| Random Forest, paper's features incl. laps completed | all | 3.52 | 2.63 | 0.63 |
| **Random Forest, pre-race features (ours)** | **finishers only** | **3.22** | **2.45** | **0.61** |
| Random Forest, paper's features incl. laps completed | finishers only | 3.29 | 2.52 | 0.60 |
| TabNet (paper, reported) | finishers only | 2.87 | 2.17 | 0.75 |

Two choices in the paper explain most of the gap:

- It uses "laps completed", which is only known after the race.
- It drops drivers with no finish position, i.e. retirements.

On the paper's terms, our pre-race model reaches 3.22, slightly ahead of the paper's own features (3.29). The remaining gap of about 0.4 places may come from TabNet with Optuna tuning and from the paper's unpublished data file; we did not test those.

![Predicted vs actual, 2023](output/chart_pred_vs_actual.png)

## Performance (from the notebook run: Spark 4.3 local mode, 2 vCPU, 7.8 GB RAM; median of 5 runs)

| Measure | Result |
| --- | --- |
| Overtaking-index Spark SQL query, uncached vs cached | 0.79 s vs 0.35 s (2.3× faster) |
| 589k-row lap aggregation, read from CSV vs Parquet | 1.07 s vs 0.22 s (4.8× faster) |
| Same aggregation on cached data, 1 / 2 / 4 / 8 partitions | 0.06 / 0.11 / 0.14 / 0.17 s |
| Same read + group-by in pandas (single-machine baseline) | 0.24 s |
| Correctness checks (Spark vs pandas, Spark SQL vs DataFrame API, lap deltas add up) | all 4 passed |

At about 21 MB on 2 cores, Spark's scheduling overhead outweighs parallelism, so adding partitions slows things down and pandas is faster. Caching and the Parquet format are the clear gains. Spark's value is that the same code scales to larger data, such as OpenF1 car telemetry, without a rewrite. Timings vary a little from run to run; the script's own measurements are in [`output/metrics.json`](output/metrics.json).

## Step-by-step notebook (evidence of every step)

[`notebook/F1_BDA_Analysis.ipynb`](notebook/F1_BDA_Analysis.ipynb) walks through the 8 points of the project form with code and outputs. A printable copy is [`notebook/F1_BDA_Analysis.pdf`](notebook/F1_BDA_Analysis.pdf), and [`notebook/F1_BDA_Analysis.html`](notebook/F1_BDA_Analysis.html) opens in any browser. It shows:

- the raw data before cleaning (`\N` text values, all-string columns) and every cleaning step with before/after rows;
- Parquet storage partitioned by decade, and Spark SQL queries;
- the physical query plan, showing partition pruning;
- lap-by-lap window functions on a real driver (Leclerc, Abu Dhabi 2024: P19 to P3);
- the MLlib models and every chart, generated from the cell above it;
- performance tests and correctness checks;
- a log of what was changed during development and why.

## How to run

**1. Get the data.** Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) (free login) and unzip the 14 CSVs into a folder named `data` next to the script. The CSVs are not included in this repository.

**2. Install.** Python 3.10+ and **Java 17 or 21** (needed by Spark 4).

```bash
pip install -r requirements.txt
```

For the notebook, also `pip install jupyter`.

**3. Run.** A full run takes about 3 minutes.

```bash
python f1_bda_pipeline.py --data ./data --out ./output
# add --skip-perf to skip the timing benchmarks
```

**On Windows:** install a Java 17 or 21 JDK (e.g. Eclipse Temurin) and make sure `java -version` works in a new terminal. If Spark cannot find Python, set `PYSPARK_PYTHON=python` before running.

**On Google Colab:** upload the CSVs to a `data` folder, then run `!pip install pyspark` followed by `!python f1_bda_pipeline.py --data data --out output`. Colab already has Java.

## Website

`website/index.html` is an interactive walkthrough of the project: the data, the cleaning steps, the Spark code, every finding as an interactive chart, a lap-by-lap explorer for all 2024 races, and the model results. Open it in a browser (it loads D3 and fonts from a CDN). To rebuild it after re-running the pipeline:

```bash
python website/build_site_data.py --data ./data --out ./output   # numbers from output/ + raw CSVs
python website/build_site.py                                     # writes website/index.html
```

## Workflow

```
Data (14 CSVs) → Cleaning → Processing → Analysis → Insights
```

1. **Cleaning:** `\N` markers become nulls; duplicate rows from shared drives, 1950–1978 (91) are removed; pit-lane starts (grid = 0) go to the back of the grid; qualifying times are parsed to seconds.
2. **Processing:** results are joined with races, circuits, status and qualifying. Window functions compute the gap to pole and each driver's and team's form over their previous 5 races.
3. **Analysis:** correlation by decade, circuit overtaking index, lap-by-lap gains by race phase (lap 1, pit-stop laps, middle stint, final third), and MLlib Random Forest and GBT models.
4. **Insights:** CSVs, charts and `metrics.json` are written to `output/`.

## Repository layout

| Path | Contents |
| --- | --- |
| `f1_bda_pipeline.py` | The whole pipeline |
| `output/` | Results from the full run: CSVs, `metrics.json`, charts, 2023 predictions |
| `website/` | Interactive project website (`index.html`) and the scripts that build it from `output/` |
| `notebook/` | Executed step-by-step notebook (`.ipynb`, `.html`, `.pdf`) and the script that builds it |
| `report/` | Scripts that build the print-sized figures and the 4-page report from `output/` |
| `requirements.txt` | Python dependencies |

## References

1. Urdhwareshe, A. (2025). *The Use of Machine Learning in Predicting Formula 1 Race Outcomes.* Preprints.org. [DOI 10.20944/preprints202504.1471.v1](https://www.preprints.org/manuscript/202504.1471)
2. Rao, R. *Formula 1 World Championship (1950–2024).* [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
3. Weissbock, J. & Mills, S. (2025). *Evaluating the Predictive Power of Qualifying Performance in Formula One Grand Prix.* [arXiv:2507.10966](https://arxiv.org/abs/2507.10966)
4. [Apache Spark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
5. [Jolpica-F1 API](https://github.com/jolpica/jolpica-f1), the source the dataset is built from

The data comes from the public Ergast/Jolpica database via Kaggle and is used here for academic, non-commercial work.
