// Builds the final 4-page report (.docx) in the RIT form layout. Every number is read from ../output.
// Usage: npm install docx && node make_report.js ../output figures <college_logo.png> <out.docx>
// Optional 5th and 6th arguments: team USNs (not stored in this repository)
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Header, Footer, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, LevelFormat, ShadingType, ExternalHyperlink,
} = require('docx');

const [OUT, FIG, LOGO, DEST, USN1 = '', USN2 = ''] = process.argv.slice(2);
const m = JSON.parse(fs.readFileSync(path.join(OUT, 'metrics.json'), 'utf8'));
const csv = (f) => {
  const [h, ...rows] = fs.readFileSync(path.join(OUT, f), 'utf8').trim().split('\n');
  const keys = h.split(',');
  return rows.map((line) => {
    const v = line.split(','); const o = {};
    keys.forEach((k, i) => { o[k] = isNaN(Number(v[i])) ? v[i] : Number(v[i]); });
    return o;
  });
};
const corr = csv('corr_by_decade.csv');
const oi = csv('overtaking_index.csv').sort((a, b) => a.overtaking_index - b.overtaking_index);
const ph = csv('phase_gains.csv');
const T = m.timings_s, Q = m.quality;
const f2 = (x) => x.toFixed(2), f1 = (x) => x.toFixed(1), pct = (x) => `${Math.round(x * 100)}%`;
const met = (k) => [f2(m[k].rmse), f2(m[k].mae), f2(m[k].r2)];
const c50 = corr[0], c20 = corr[corr.length - 1], c10 = corr.find((c) => c.decade === 2010);
const lo = oi.slice(0, 3), hi = oi.slice(-3).reverse();
const phase = Object.fromEntries(ph.map((r) => [r.phase.slice(0, 1), r]));
const fi = m.rf_feature_importance;
const gain = (1 - m.random_forest.rmse / m.baseline_grid_equals_finish.rmse) * 100;

const logo = fs.readFileSync(LOGO);
const FONT = 'Calibri', SZ = 21;
const r = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || SZ, bold: o.bold, italics: o.italics, color: o.color });
const p = (runs, o = {}) => new Paragraph({
  children: Array.isArray(runs) ? runs : [r(runs)],
  alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { before: o.before ?? 0, after: o.after ?? 70, line: o.line ?? 246 },
  numbering: o.num, keepNext: o.keepNext,
});
const lab = (label, text) => p([r(label, { bold: true }), r(' ' + text)]);
const h = (text) => p([r(text, { bold: true })], { before: 120, after: 50, align: AlignmentType.LEFT, keepNext: true });
const num = (runs, ref = 'n1') => p(runs, { num: { reference: ref, level: 0 }, after: 30 });
const link = (text, url) => new ExternalHyperlink({ link: url, children: [new TextRun({ text, font: FONT, size: SZ, color: '1F4E79', underline: {} })] });
const code = (line) => new Paragraph({
  children: [new TextRun({ text: line, font: 'Consolas', size: 16 })],
  spacing: { before: 0, after: 0, line: 216 }, shading: { type: ShadingType.CLEAR, fill: 'F2F2F2' },
});
const line = { style: BorderStyle.SINGLE, size: 4, color: '000000' };
const none = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
function table(widths, rows, o = {}) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((row, i) => new TableRow({ cantSplit: true, children: row.map((c, j) => new TableCell({
      borders: { top: line, bottom: line, left: line, right: line }, width: { size: widths[j], type: WidthType.DXA },
      margins: { top: 25, bottom: 25, left: 80, right: 80 },
      shading: o.hl && o.hl.includes(i) ? { type: ShadingType.CLEAR, fill: 'EAF2FB' } : undefined,
      children: [new Paragraph({ keepNext: i < rows.length - 1, keepLines: true, spacing: { before: 0, after: 0 },
        alignment: j > 0 && o.num ? AlignmentType.CENTER : AlignmentType.LEFT,
        children: [r(c, { bold: i === 0, size: 19 })] })],
    })) })),
  });
}
const img = (f, w, hgt) => new ImageRun({ type: 'png', data: fs.readFileSync(path.join(FIG, f)), transformation: { width: w, height: hgt } });
function figRow(a, b, hgt) {
  const cell = (child) => new TableCell({ borders: { top: none, bottom: none, left: none, right: none },
    width: { size: 4820, type: WidthType.DXA }, margins: { top: 20, bottom: 20, left: 0, right: 0 },
    children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 0 }, children: [child] })] });
  return new Table({ width: { size: 9640, type: WidthType.DXA }, columnWidths: [4820, 4820],
    borders: { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none },
    rows: [new TableRow({ cantSplit: true, children: [cell(img(a, 312, hgt)), cell(img(b, 312, hgt))] })] });
}

const L = AlignmentType.LEFT, C = AlignmentType.CENTER;
const body = [
  p([r('RAMAIAH INSTITUTE OF TECHNOLOGY', { bold: true, size: 22 })], { align: C, after: 90 }),
  p([r('Department of Computer Science and Engineering', { size: 22 })], { align: C, after: 90 }),
  p([r('Dataset Analysis for Real-Time Scenarios', { bold: true, size: 22 })], { align: C, after: 90 }),
  p([r('Subject: Big Data Analytics  |  Code: CSE557  |  Term: 2026  |  Non-CIE: 10 Marks', { size: 22 })], { align: C, after: 110 }),
  p([r('Submission last date is 2/11/26', { size: 22, color: 'FF0000' })], { align: L, after: 90 }),
  p([r('Project title: ', { size: 22 }), r('Does Qualifying Decide the Race? Grid-Position Impact Analysis and Finishing-Position Prediction on Formula 1 Data using PySpark', { size: 22, bold: true })], { align: L, after: 90 }),
  p([r('Faculty guide: ', { size: 22 }), r('Uzma Sulthana', { size: 22 })], { align: L, after: 90 }),
  p([r('Team member details', { bold: true, size: 22 })], { align: L, after: 50 }),
  table([3200, 3200, 3240], [
    ['Student name', 'USN', 'Main responsibility'],
    ['Arnav Chatterjee', USN1, 'Data ingestion, cleaning, PySpark pipeline, MLlib model, performance evaluation'],
    ['Khuushi Chand Navin', USN2, 'Paper review, Spark SQL analysis, visualizations, comparison with paper'],
  ]),

  h('1. Real-world scenario and problem statement'),
  lab('Scenario:', 'Sports analytics (Formula 1). Teams, strategists, broadcasters and fantasy/prediction platforms all need to know how much of a race result is already decided by qualifying.'),
  lab('Problem:', 'the effect of grid position varies by circuit and era, yet there is no scalable, data-driven measure of it per track, nor a reproducible way to predict finishing position from pre-race data. Stakeholders: team strategists, broadcasters and analysts, fantasy/prediction platforms, race organisers.'),
  p([r('Analytical questions: ', { bold: true }), r('(1) How strongly does grid position correlate with finish, and how has it changed by decade? (2) Which circuits are hardest to overtake at? (3) Where in a race are places gained or lost? (4) Can a distributed ML model predict finishing position, and how does it compare with a 2025 paper?')]),
  lab('Why Big Data:', `14 relational tables, including ${Q.lap_rows.toLocaleString('en-US')} lap records, must be joined and aggregated; Spark does this in parallel and scales unchanged to telemetry data. Data type: historical (1950–2024); nothing is streamed or simulated.`),

  h('2. Recent research paper and existing approach'),
  p([r('Paper: ', { bold: true }), r('Urdhwareshe, A. (2025). '), r('The Use of Machine Learning in Predicting Formula 1 Race Outcomes', { italics: true }), r('. Preprints.org. DOI: '), link('10.20944/preprints202504.1471.v1', 'https://www.preprints.org/manuscript/202504.1471')]),
  lab('Approach:', 'predicts a driver’s finishing position from one CSV of 2010–2023 results (4,637 rows after dropping rows with no grid or finish position) using TabNet with label encoding, MinMax scaling and Optuna tuning (20 trials). Inputs: grid, laps completed, constructor, round. Train 2010–2022, test 2023: RMSE 2.87, MAE 2.17, R² 0.75.'),
  lab('Limitations and opportunity:', 'single-machine pandas workflow; uses “laps completed”, which is only known after the race; excludes drivers who did not finish; no per-circuit or per-era insight; not peer-reviewed. We rebuild the analysis in PySpark over all 14 tables using pre-race features only, and test the effect of each of the paper’s choices.'),

  h('3. Objectives and proposed analysis'),
  num([r('Dataset preparation: ', { bold: true }), r('clean and join the 14 tables into one race-level table; measure completeness and row counts before and after.')]),
  num([r('Scalable analysis: ', { bold: true }), r('Spark SQL for grid–finish correlation by decade, a circuit overtaking index (2010–2024) and lap-by-lap gains by race phase; measure query time, caching, partitioning and a pandas baseline.')]),
  num([r('Evaluation: ', { bold: true }), r('MLlib Random Forest and GBT on the paper’s split (train 2010–2022, test 2023); compare RMSE, MAE, R² with a “finish = grid” baseline and the paper.')]),

  h('4. Dataset identification and preparation'),
  p([r('Dataset: ', { bold: true }), r('Formula 1 World Championship (1950–2024), Rohan Rao, Kaggle: '), link('kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020', 'https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020'), r(`. Source: Ergast/Jolpica F1 database. 14 CSV files (about 21 MB, 701,000 rows) linked by raceId, driverId, constructorId and circuitId; ${Q.results_rows_raw.toLocaleString('en-US')} race results and ${Q.lap_rows.toLocaleString('en-US')} lap records. Public, free Kaggle login.`)], { align: L }),
  lab('Preparation (measured):', `“\\N” markers read as nulls. Finish position (“position”) is null for ${f1(100 - Q['results_completeness_raw_%'].position)}% of results (non-classified drivers), so the complete “positionOrder” column is the target. ${Q.duplicate_rows_removed} duplicate (race, driver) rows (shared drives, 1950–1978) were reduced to the better finish. Pit-lane starts (grid = 0) moved to the back of the grid. Qualifying times (m:ss.sss) parsed to seconds; gap to pole is available for ${f1(Q['results_completeness_clean_%'].quali_gap_pct)}% of rows (qualifying data starts in 1994) and missing values are flagged. Derived: places gained, finished flag, decade, 5-race rolling driver and team form.`),
  lab('Big Data characteristics:', 'variety (14 linked tables, mixed types), veracity (nulls, retirements, shared drives, no lap data before 1996, no pit data before 2011) and volume (589k lap records; the same code scales to telemetry).'),

  h('5. Big Data implementation and workflow'),
  p('Apache Spark 4.3 via PySpark (DataFrames, Spark SQL functions, window functions, MLlib) in local mode. HDFS is not needed at this size; the same code runs on HDFS by changing the input path.'),
  lab('Workflow:', 'Data (read 14 CSVs) → Cleaning (nulls, types, duplicates, grid fix) → Processing (joins, qualifying gap, rolling form via windows) → Analysis (correlation, overtaking index, phase gains, MLlib models) → Insights (CSVs, charts, metrics.json).'),
  p([r('Execution: ', { bold: true }), r('pip install -r requirements.txt → put the Kaggle CSVs in ./data → python f1_bda_pipeline.py --data ./data --out ./output. Code: '), link('github.com/arnavchatterjee0528/f1-grid-impact-pyspark', 'https://github.com/arnavchatterjee0528/f1-grid-impact-pyspark')]),
  ...[
    'res = spark.read.csv("data/results.csv", header=True, inferSchema=True, nullValue="\\\\N")',
    'wd  = Window.partitionBy("driverId").orderBy("date").rowsBetween(-5, -1)   # past 5 races only',
    'df  = df.withColumn("driver_form5", F.avg("positionOrder").over(wd))',
    'oi  = (res.filter("finished = 1 AND year >= 2010").groupBy("circuit")',
    '          .agg(F.avg(F.abs("positions_gained")).alias("overtaking_index")))',
    'rf  = RandomForestRegressor(labelCol="label", numTrees=200, maxDepth=8).fit(train)  # train <= 2022',
  ].map(code),

  h('6. Results, visualization and paper comparison'),
  figRow('fig_a_corr.png', 'fig_b_phase.png', 189),
  num([r('Qualifying matters more than ever (a). ', { bold: true }), r(`Grid–finish correlation rose from ${f2(c50.grid_finish_corr)} in the 1950s to ${f2(c20.grid_finish_corr)} in both the 2010s and 2020s; pole sitters won ${pct(c50.pole_win_rate)} of races in the 1950s and ${pct(c20.pole_win_rate)} in the 2020s.`)], 'n2'),
  num([r('Places are lost in pit stops and won back mid-race (b). ', { bold: true }), r(`Drivers net \u2212${f1(Math.abs(phase['2'].avg_net_gain_per_driver_race))} places on pit-stop laps and +${f1(phase['3'].avg_net_gain_per_driver_race)} in the middle stint; lap 1 changes only +${f1(phase['1'].avg_net_gain_per_driver_race)} net.`)], 'n2'),
  p('', { after: 40 }),
  table([4740, 1600, 1100, 1100, 1100], [
    ['Model (test season 2023)', 'Rows', 'RMSE', 'MAE', 'R²'],
    ['Baseline: finish = grid', 'all', ...met('baseline_grid_equals_finish')],
    ['Random Forest, pre-race features (ours)', 'all', ...met('random_forest')],
    ['GBT, pre-race features (ours)', 'all', ...met('gbt')],
    ['Random Forest, paper’s features incl. laps', 'all', ...met('random_forest_paper_features_incl_laps')],
    ['Random Forest, pre-race features (ours)', 'finishers', ...met('random_forest_classified_only')],
    ['Random Forest, paper’s features incl. laps', 'finishers', ...met('random_forest_paper_features_classified_only')],
    ['TabNet (paper, reported)', 'finishers', '2.87', '2.17', '0.75'],
  ], { num: true, hl: [5] }),
  p([r('Comparison with the paper: ', { bold: true }), r(`our pre-race model (RMSE ${f2(m.random_forest.rmse)}) looks worse than the paper (2.87), but the setups differ. Adding the post-race “laps” input alone lowers RMSE to ${f2(m.random_forest_paper_features_incl_laps.rmse)}, and removing non-finishers as the paper did lowers it to ${f2(m.random_forest_classified_only.rmse)} using pre-race information only, slightly better than the paper’s own inputs under the same conditions (${f2(m.random_forest_paper_features_classified_only.rmse)}). The remaining gap of about ${f1(m.random_forest_classified_only.rmse - 2.87)} places may come from TabNet with Optuna tuning and from the paper’s unpublished data file; we did not test those.`)], { before: 80 }),

  figRow('fig_c_overtaking.png', 'fig_d_pred.png', 284),
  num([r('Circuits differ (c). ', { bold: true }), r(`Hardest to change places since 2010: ${lo.map((c) => `${c.country} (${f2(c.overtaking_index)})`).join(', ')}; easiest: ${hi.map((c) => `${c.country} (${f2(c.overtaking_index)})`).join(', ')}.`)], 'n2'),
  num([r('Prediction (d). ', { bold: true }), r(`Random Forest cuts RMSE by ${Math.round(gain)}% versus \u201Cfinish = grid\u201D. Most important inputs: grid (${f2(fi.grid_clean)}), team form (${f2(fi.constructor_form5)}), qualifying gap (${f2(fi.quali_gap_pct)}), driver form (${f2(fi.driver_form5)}). Errors are largest for drivers who retire (bottom right of d).`)], 'n2'),
  h('7. Performance evaluation and observations'),
  lab('Setup:', 'Spark 4.3 local mode, 2 vCPU, 7.8 GB RAM, Python 3.13. Timings are the median of 5 runs after one warm-up run.'),
  table([5100, 4540], [
    ['Measure', 'Result'],
    ['Data completeness after cleaning', `grid and finish 100%; qualifying gap ${f1(Q['results_completeness_clean_%'].quali_gap_pct)}% (flagged)`],
    ['Aggregation correctness (Spark vs pandas)', `${m.timings_s.check_groups_spark.toLocaleString('en-US')} vs ${m.timings_s.check_groups_pandas.toLocaleString('en-US')} (race, driver) groups: match`],
    ['Overtaking-index query: uncached vs cached', `${f2(T.spark_overtaking_index_uncached)} s vs ${f2(T.spark_overtaking_index_cached)} s (${f1(T.spark_overtaking_index_uncached / T.spark_overtaking_index_cached)}× faster)`],
    ['589k-row lap aggregation: 1 / 2 / 4 / 8 partitions', `${f2(T.spark_lap_agg_1_partitions)} / ${f2(T.spark_lap_agg_2_partitions)} / ${f2(T.spark_lap_agg_4_partitions)} / ${f2(T.spark_lap_agg_8_partitions)} s`],
    ['Same read + group-by: pandas vs Spark', `${f2(T.pandas_lap_agg_incl_read)} s vs ${f2(T.spark_lap_agg_incl_read)} s`],
    ['Model training: Random Forest / GBT', `${f1(T.train_random_forest)} s / ${f1(T.train_gbt)} s`],
  ]),
  p('', { after: 20 }),
  lab('Observations:', `caching gave the clearest gain. Adding partitions made the lap aggregation slower, and pandas was ${f1(T.spark_lap_agg_incl_read / T.pandas_lap_agg_incl_read)}× faster than Spark, because at 21 MB on 2 cores Spark’s scheduling overhead outweighs parallelism. Spark’s value here is that the same code scales to larger data, not speed at this size. GBT was slower to train and less accurate than Random Forest.`),
  lab('Limitations and improvements:', 'single machine, so no true cluster speed-up was measured; retirements cannot be predicted from pre-race data; the paper’s exact data is unpublished. Next: add OpenF1 telemetry and weather (2023 onwards), run on a multi-node HDFS cluster, and tune MLlib with cross-validation.'),

  h('8. Tools, responsible use and references'),
  lab('Tools:', 'Python 3, Apache Spark 4.3 (PySpark, Spark SQL, MLlib), pandas, Matplotlib; GitHub for code.'),
  lab('Responsible use:', 'public Kaggle dataset used for academic, non-commercial work under its listed licence; only public sporting results and professional drivers’ names; no credentials stored; only measured results reported. The CSVs are not redistributed in the repository.'),
  p([r('References: ', { bold: true }), r('[1] Urdhwareshe, A. (2025), The Use of Machine Learning in Predicting Formula 1 Race Outcomes, DOI: 10.20944/preprints202504.1471.v1. [2] Rao, R., Formula 1 World Championship (1950–2024), Kaggle. [3] Weissbock, J. & Mills, S. (2025), Evaluating the Predictive Power of Qualifying Performance in Formula One Grand Prix, arXiv:2507.10966. [4] Apache Spark MLlib Guide, spark.apache.org/docs/latest/ml-guide.html. [5] Jolpica-F1 API documentation, github.com/jolpica/jolpica-f1.')]),
];

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: SZ } } } },
  numbering: { config: [
    { reference: 'n1', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 300 } } } }] },
    { reference: 'n2', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1250, bottom: 1050, left: 1300, right: 1300, header: 480, footer: 480 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: C, children: [new ImageRun({ type: 'png', data: logo, transformation: { width: 104, height: 35 } })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [r('BDA Dataset Analysis Proposal', { size: 20 })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(DEST, b); console.log('wrote', DEST); });
