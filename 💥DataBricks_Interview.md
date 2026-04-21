<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 🗓️ DAY 1 — Databricks Foundation: Complete Study Guide

**TL;DR** — Databricks = Unified Lakehouse platform on top of Apache Spark. Day 1 covers the UI, compute basics, Spark internals, and hands-on Delta ops. This is your foundation for everything else. 🧱

***

## 🧭 1. Workspace Tour

The **Databricks Workspace** is your home screen — a collaborative environment where data engineers, scientists, and analysts share notebooks, jobs, clusters, and data assets in one platform.[^1_1]

Here's what each tab does:


| Tab | What it is | When you use it |
| :-- | :-- | :-- |
| 🏠 **Workspace** | File explorer for notebooks, folders, libraries | Organizing your code, sharing notebooks |
| 📁 **Repos** | Git-integrated folder (GitHub/GitLab/ADO) | Version control, CI/CD pipelines |
| 📊 **Data / Catalog** | Unity Catalog — browse databases, tables, schemas | Finding tables, checking schemas, data governance |
| ⚙️ **Compute** | Create/manage clusters \& SQL Warehouses | Before you run ANY code, you need compute here |
| 🔄 **Workflows** | Schedule jobs, pipelines, DLT | Production scheduling, DAG orchestration |
| 🔍 **SQL Editor** | Write and run SQL queries | Ad-hoc analysis, BI-style queries |

> 💡 **Why Repos over plain Workspace?** Repos support Git branching. In production, you NEVER want code sitting in Workspace only — it's unversioned, ungoverned, and untraceable. Always use Repos for team projects.

***

## ⚡ 2. Cluster Deep Dive: All-Purpose vs Job Cluster

This is one of the **most asked interview topics**. Know this cold. 🥶[^1_2]


| Feature | 🟢 All-Purpose Cluster | 🔵 Job Cluster |
| :-- | :-- | :-- |
| **Created by** | You (manually, via UI/CLI/API) | Databricks Job Scheduler (auto) |
| **Lifecycle** | Stays alive until you terminate | Spins up → job runs → auto-terminates |
| **Cost** | 💸 **Expensive** — you pay even when idle | ✅ **Cheaper** — pay only during job run |
| **Use case** | Dev, exploration, notebook work | Production ETL/ELT jobs |
| **Multi-user?** | Yes, shared cluster | No, isolated per job run |
| **Restartable?** | Yes | ❌ No — cannot restart a job cluster |

> 🚨 **Cost Warning (Senior-level flag):** All-purpose clusters run on DBUs (Databricks Units). They cost roughly **2x more DBUs** per hour than job clusters. If a junior DE leaves an all-purpose cluster running overnight = 💸💸💸. Always set **auto-termination** (30-60 mins idle) on all-purpose clusters.

> ❌ **Never use all-purpose clusters in production jobs.** This is a classic mistake. Interviewers will ask this.[^1_3]

***

## 📈 3. Autoscaling — Set It Right

Autoscaling lets Databricks automatically add/remove worker nodes based on load.[^1_4]

**How it works:**

- You set **Min workers** and **Max workers**
- Databricks monitors pending tasks in the Spark queue
- If tasks pile up → scale **out** (add workers)
- If tasks finish and nodes idle → scale **in** (remove workers)

**When to use autoscaling ✅:**

- Unpredictable/bursty workloads (e.g., event-driven pipelines, dashboards with variable user load)
- Development clusters (you don't know how big your dataset is yet)
- Cost-sensitive environments

**When NOT to use autoscaling ❌:**

- Streaming jobs (Spark Structured Streaming performs poorly with autoscaling — use fixed clusters)
- Jobs with very predictable, constant load (fixed cluster = faster, no scale-up lag)
- When you need deterministic SLA times (scaling latency adds unpredictability)

> ⚠️ **Tech Debt Flag:** Autoscaling with min=0 sounds great for cost but causes cold-start delays. In prod, min=1 is safer.

***

## 🚀 4. Photon — Vectorized Execution Engine

Photon is Databricks' **native query acceleration engine**, written from scratch in **C++** (not Java/JVM like regular Spark).[^1_5]

**Row-by-row vs Vectorized:**

```
🐢 Traditional Spark (JVM):   [row1] → process → [row2] → process → [row3] ...
🚀 Photon (Vectorized C++):   [row1, row2, row3, row4... batch of 1000s] → process all at once
```

Photon uses **CPU SIMD instructions** (Single Instruction, Multiple Data) to process batches of data in parallel at the hardware level.[^1_6]

**Photon speeds up:**

- ✅ SQL aggregations (`GROUP BY`, `SUM`, `COUNT`)
- ✅ Joins (hash joins, sort-merge joins)
- ✅ Delta Lake MERGE operations
- ✅ Parquet/Delta reads with predicate pushdown
- ✅ Window functions

**Photon does NOT help:**

- ❌ Python UDFs (they break out of JVM/C++ into Python process)
- ❌ Custom Scala/Java code outside SQL engine
- ❌ ML model training (different compute path)

> 💡 Photon is available on **Databricks Runtime 9.1+**. Enable it at cluster creation time — it's a checkbox. It costs slightly more DBUs but delivers up to **~12x price/performance improvement** for eligible workloads.[^1_5]

***

## 📓 5. Notebook Magic Commands

Magic commands are **cell-level language switchers** — they override the notebook's default language for just that one cell.[^1_7]

```python
# Default language of notebook is Python

# ---- Cell 1: Python (default, no magic needed) ----
df = spark.read.csv("/data/sales.csv", header=True)

# ---- Cell 2: SQL magic ----
# %sql
# SELECT product, SUM(revenue) FROM sales GROUP BY product

# ---- Cell 3: Scala magic ----
# %scala
# val rdd = sc.parallelize(List(1, 2, 3))

# ---- Cell 4: Shell command ----
# %sh
# ls /dbfs/data/

# ---- Cell 5: Install library ----
# %pip install great_expectations

# ---- Cell 6: Run another notebook ---- 
# %run ./utils/data_quality_checks
# After %run, all variables/functions from that notebook are available HERE
```

> 🔑 **`%run` is KEY for modular pipelines.** You put common functions (schema definitions, utility functions, connection strings) in a separate notebook and `%run` it at the top of every pipeline notebook. Think of it like Python `import` but for notebooks.

> 🚨 `%run` is **blocking and synchronous** — the called notebook must complete before the parent continues. Don't confuse it with `dbutils.notebook.run()` which is async and returns a string result — used for orchestration with timeouts.

***

## 🗂️ 6. DBFS — Databricks File System

DBFS is a **virtual filesystem abstraction layer** that sits on top of your actual cloud storage (S3 on AWS, ADLS Gen2 on Azure, GCS on GCP).[^1_8]

```
Your Code          DBFS Layer           Cloud Storage
-----------     ----------------     -----------------
spark.read     →  /dbfs/mnt/data  →   s3://my-bucket/data/
.csv("/dbfs/")     (abstraction)       or adls://...
```

**Key paths:**

- `/dbfs/` — root of DBFS (local file system path in notebooks/scripts)
- `dbfs:/` — same root but in Spark APIs and `dbutils`
- `/dbfs/FileStore/` — for file uploads from UI
- `/mnt/` — **mount points** (you mount S3/ADLS here using `dbutils.fs.mount()`)

```python
# List files
dbutils.fs.ls("dbfs:/mnt/raw/")

# Copy files
dbutils.fs.cp("dbfs:/mnt/raw/file.csv", "dbfs:/mnt/processed/file.csv")

# Check if path exists (useful in pipelines)
def path_exists(path):
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False
```

> 💡 **Modern best practice (2024+):** Unity Catalog + External Locations is replacing DBFS mounts. If your company uses UC, avoid `dbutils.fs.mount()` — use `abfss://` or `s3://` paths directly with proper IAM/credential passthrough.

***

## 🔄 7. Spark Execution Model — The Brain of Everything

This is the **most important conceptual topic** in all of Spark/Databricks. Every performance issue traces back to understanding this. 🧠

### The Pipeline: DAG → Stages → Tasks

```
Your Code (Transformations)
        ↓
    DAG (Directed Acyclic Graph)
    — logical plan of all operations
        ↓
    Stages (separated by "shuffles")
    — a shuffle = data moving across network
        ↓
    Tasks (1 task per partition per stage)
    — actual units of work running on executors
        ↓
    Action (.show(), .count(), .write())
    — TRIGGERS the whole thing to actually run
```


### 😴 Lazy Evaluation — Nothing Runs Until You Force It

```python
# These lines DO NOTHING — they just build a logical plan
df = spark.read.csv("/data/sales.csv", header=True)     # no execution
df_filtered = df.filter(df.revenue > 1000)              # no execution
df_grouped = df_filtered.groupBy("product").count()     # no execution

# THIS triggers execution — all the above runs NOW
df_grouped.show()   # ACTION → DAG executes!
df_grouped.count()  # another action → re-executes unless cached!
```

> 💡 **Why lazy evaluation?** Spark's optimizer (Catalyst) can look at your ENTIRE chain of transformations before running anything, and **reorder/optimize** them. Example: if you filter then join, Catalyst may push the filter BEFORE the join to reduce data size — saving massive shuffle cost. You can't do this optimization if you execute eagerly.

### Shuffles — The Performance Killer 💀

A **shuffle** occurs when data needs to move between executors (nodes). It causes:

- Network I/O
- Disk writes (spill to disk)
- Stage boundary in DAG

```python
# These operations CAUSE shuffles (= stage boundaries):
df.groupBy("country").count()          # shuffle to group same keys
df.join(other_df, on="id")             # shuffle to co-locate matching keys
df.orderBy("date")                     # shuffle to sort globally
df.repartition(200)                    # explicit shuffle

# These do NOT cause shuffles (within same stage):
df.filter(df.age > 25)                 # just skips rows
df.select("name", "age")               # just drops columns
df.withColumn("new", df.a + df.b)      # row-level transform
```


### Reading `df.explain()` Output

```python
df_result.explain(mode="formatted")
```

```
== Physical Plan ==
AdaptiveSparkPlan (1)          ← AQE wrapper (Adaptive Query Execution)
+- HashAggregate (2)           ← final aggregation (after shuffle)
   +- Exchange (3)             ← 🚨 SHUFFLE HAPPENS HERE
      +- HashAggregate (4)     ← partial aggregation (before shuffle)
         +- Filter (5)         ← filter pushed down ✅ (Catalyst optimization)
            +- FileScan (6)    ← reading from Delta/Parquet
```

Read it **bottom-up** — execution starts at the bottom (FileScan) and goes up.

***

## 💻 Hands-On Code

### Task 1: Multi-language Notebook Pattern

```python
# Cell 1 - %python
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

schema = StructType([
    StructField("order_id", IntegerType()),
    StructField("product", StringType()),
    StructField("category", StringType()),
    StructField("revenue", DoubleType()),
    StructField("country", StringType())
])

data = [
    (1, "Laptop", "Electronics", 1200.0, "India"),
    (2, "Phone", "Electronics", 800.0, "USA"),
    (3, "Desk", "Furniture", 350.0, "India"),
    (4, "Chair", "Furniture", 200.0, "UK"),
    (5, "Tablet", "Electronics", 600.0, "USA"),
    (6, "Monitor", "Electronics", 450.0, "India"),
]

df = spark.createDataFrame(data, schema)
df.createOrReplaceTempView("orders")  # expose to %sql cells
print("DataFrame created!")
```

```sql
-- Cell 2 - %sql
-- Now use the temp view created above
SELECT 
    category,
    country,
    COUNT(*) as order_count,
    ROUND(SUM(revenue), 2) as total_revenue
FROM orders
GROUP BY category, country
ORDER BY total_revenue DESC
```


### Task 2: CSV → Parquet → Delta

```python
# Read CSV (or create sample)
df = spark.createDataFrame(data, schema)

# Write as Parquet
df.write.mode("overwrite").parquet("/tmp/orders_parquet")

# Read parquet back and write as Delta
df_parquet = spark.read.parquet("/tmp/orders_parquet")
df_parquet.write.format("delta").mode("overwrite").save("/tmp/orders_delta")

# Register as Delta table in catalog
spark.sql("""
    CREATE TABLE IF NOT EXISTS orders_delta
    USING DELTA
    LOCATION '/tmp/orders_delta'
""")
```


### Task 3: filter(), groupBy().agg(), join()

```python
# filter() - push predicates early, reduces data in pipeline
high_value = df.filter(F.col("revenue") > 500)

# groupBy().agg() - multi-metric aggregation
summary = df.groupBy("category").agg(
    F.count("order_id").alias("total_orders"),
    F.sum("revenue").alias("total_revenue"),
    F.avg("revenue").alias("avg_revenue"),
    F.max("revenue").alias("max_revenue")
)

# join() - always filter BEFORE joining to reduce shuffle size
country_info = spark.createDataFrame([
    ("India", "Asia"), ("USA", "Americas"), ("UK", "Europe")
], ["country", "region"])

# filter first, THEN join (Catalyst does this anyway, but explicit = readable)
enriched = high_value.join(country_info, on="country", how="left")
enriched.show()
```


### Task 4: explain() + Spark UI

```python
# Read the physical plan - bottom up!
enriched.explain(mode="formatted")

# Cache to avoid re-computation if used multiple times
enriched.cache()
enriched.count()  # materializes cache

# Check partitions (related to parallelism)
print(f"Number of partitions: {enriched.rdd.getNumPartitions()}")

# Repartition if too many small files
enriched_repartitioned = enriched.repartition(4)
```

> 💡 **Spark UI** is at `https://<your-cluster-url>/sparkui` — click on a job → see stages → click a stage → see task distribution. Look for **skewed tasks** (one task takes 10x longer than others = data skew problem 🎯).

***

## 🎯 Interview Questions — Day 1

**Q1.** What is the difference between an All-Purpose cluster and a Job cluster? When would you use each?
> ✅ **Answer:** All-purpose: interactive/dev, multi-user, manually managed, expensive. Job: prod ETL, auto-created/terminated by scheduler, cheaper, isolated. Never use all-purpose in prod — you pay for idle time.[^1_2]

**Q2.** What is lazy evaluation in Spark? Why does it exist?
> ✅ **Answer:** Transformations build a logical plan (DAG) but don't execute. Only an **action** (`.show()`, `.count()`, `.write()`) triggers execution. This allows Catalyst optimizer to reorder/prune operations before running — enabling predicate pushdown, column pruning, join reordering for better performance.

**Q3.** What is a shuffle in Spark? Which operations cause it?
> ✅ **Answer:** Shuffle = moving data across network between executors. Caused by: `groupBy`, `join`, `orderBy`, `repartition`, `distinct`. It's expensive (network + disk I/O). Minimizing shuffles is key to performance optimization.

**Q4.** What is Photon? When does it NOT help?
> ✅ **Answer:** C++ vectorized execution engine that processes data in batches instead of row-by-row. Speeds up SQL, aggregations, joins, Delta merges up to 12x. Does NOT help with Python UDFs (they exit the C++ engine back into Python process).[^1_5]

**Q5.** What is the difference between `%run` and `dbutils.notebook.run()`?
> ✅ **Answer:** `%run` is synchronous — imports variables/functions into current notebook's scope, used for sharing utilities. `dbutils.notebook.run()` runs a notebook as a separate job asynchronously, returns a string value, has a timeout parameter — used for orchestration.

**Q6.** What is DBFS? Is it recommended to use in 2024+?
> ✅ **Answer:** Virtual filesystem abstraction over cloud storage (S3/ADLS/GCS). For new Databricks setups with Unity Catalog, DBFS mounts (`/mnt/`) are being deprecated in favor of External Locations and direct cloud paths (`abfss://`, `s3://`) for better governance and security.

**Q7.** What is a DAG in Spark?
> ✅ **Answer:** Directed Acyclic Graph — Spark's logical execution plan of all transformations. It's "directed" (has flow direction), "acyclic" (no loops). DAG is split into stages at shuffle boundaries. Each stage has tasks (1 per partition).

**Q8.** When should you NOT use autoscaling?
> ✅ **Answer:** Structured Streaming jobs (autoscaling interferes with micro-batch timing), jobs with predictable/constant load (fixed cluster is faster, no scale-up latency), when you need deterministic SLAs.

**Q9.** How do you read a Spark physical plan from `explain()`?
> ✅ **Answer:** Read **bottom-up**. Bottom = data source (FileScan). `Exchange` = shuffle boundary. `HashAggregate` appearing twice = partial agg before shuffle + final agg after shuffle (optimization). Look for wide operations that cause stage splits.

**Q10.** What are DBUs?
> ✅ **Answer:** Databricks Units — the billing metric for compute. All-purpose clusters cost more DBUs/hour than job clusters. Photon-enabled clusters cost slightly more DBUs but deliver better performance per dollar for eligible workloads.

***

## 🗺️ Mental Model: Day 1 Summary Map

```
Databricks Workspace
├── 📁 Repos (Git)
├── ⚙️ Compute
│   ├── All-Purpose (dev) — 💸 expensive, interactive
│   └── Job Cluster (prod) — ✅ cheap, auto-terminate
│       └── Photon enabled? → vectorized C++ engine 🚀
├── 📓 Notebooks
│   ├── %python / %sql / %scala / %sh / %pip
│   └── %run → modular pipeline pattern
├── 🗂️ DBFS → abstraction over S3/ADLS
└── Apache Spark Engine
    ├── Lazy eval → Catalyst optimizer → DAG
    ├── DAG → Stages (split at shuffles)
    └── Stages → Tasks (1 per partition)
```


***

> 
---

# 🗓️ DAY 2 — Delta Lake Deep Dive: Complete Guide

**TL;DR** — Delta Lake = Parquet files + a transaction log (`_delta_log`) that gives you ACID on a data lake. Every feature (time travel, MERGE, OPTIMIZE) is powered by that single log. Understand the log → understand everything. 🧠

***

## 🔬 1. Delta Table Anatomy — What's Actually on Disk

When you create a Delta table, Delta doesn't just write Parquet files. It creates **two things** side-by-side:[^2_1]

```
my_delta_table/
├── _delta_log/                    ← THE BRAIN 🧠
│   ├── 00000000000000000000.json  ← version 0 (CREATE TABLE)
│   ├── 00000000000000000001.json  ← version 1 (first INSERT)
│   ├── 00000000000000000002.json  ← version 2 (UPDATE/MERGE)
│   └── 00000000000000000010.checkpoint.parquet  ← snapshot every 10 versions
│
├── part-00001-abc123.parquet      ← actual data files
├── part-00002-def456.parquet
└── part-00003-ghi789.parquet      ← old version file (not yet VACUUMed)
```

Each JSON file in `_delta_log` is a **commit** that contains:[^2_2]

- `add` — new Parquet files added
- `remove` — old Parquet files logically deleted (tombstoned, not physically deleted yet)
- `metaData` — schema changes
- `protocol` — Delta protocol version
- `commitInfo` — who did what, when (timestamp, operation type)

> 💡 **Key insight:** Delta never immediately deletes Parquet files. It just writes a `remove` action in the log. The physical files stick around until you run `VACUUM`. This is exactly WHY time travel works — old files are still there![^2_2]

### 📸 Checkpoints — Every 10 Versions

Reading 1000 JSON files to reconstruct table state would be slow. So every **10 commits**, Delta writes a **checkpoint** `.parquet` file — a full snapshot of the table state. Spark reads the latest checkpoint + only the JSON files after it.[^2_3]

***

## 🔐 2. ACID Transactions — How Delta Actually Does It

ACID = **A**tomicity, **C**onsistency, **I**solation, **D**urability. Here's exactly how Delta delivers each:[^2_1][^2_2]


| Property | What it means | How Delta does it |
| :-- | :-- | :-- |
| ⚛️ **Atomicity** | All or nothing — no partial writes | Writes go to new Parquet files; only committed to `_delta_log` when fully done. If writer crashes mid-write → no commit entry → transaction never happened |
| ✅ **Consistency** | Data always valid, schema enforced | Schema validation on every write; `_delta_log` only accepts valid commits |
| 🔒 **Isolation** | Concurrent writes don't corrupt each other | **Optimistic Concurrency Control** (see below) |
| 💾 **Durability** | Committed data survives crashes | `_delta_log` on durable cloud storage (S3/ADLS) — once commit JSON written, it's permanent |

### ⚔️ Optimistic Concurrency Control (OCC) — The Isolation Mechanism

This is the most interview-tested part of ACID in Delta:[^2_1]

```
Writer A                    Writer B
  |                            |
  | 1. Record start version=5  | 1. Record start version=5
  | 2. Read + process data     | 2. Read + process data
  | 3. Try commit as v6 ✅     | 3. Try commit as v6 ❌ (A already wrote v6!)
  |                            | 4. Check: did A's commit affect MY reads?
  |                            |    YES → retry / raise conflict error
  |                            |    NO → commit as v7 ✅
```

> Delta assumes conflicts are **rare** (optimistic), so it doesn't lock the table upfront. It only checks for conflicts at commit time. If a conflict is detected, it retries or raises a `ConcurrentModificationException`. This gives high concurrency without heavy locking.[^2_1]

***

## ⏳ 3. Time Travel — Querying the Past

Delta keeps all old Parquet files (until VACUUMed), so you can literally **query any past version** of your table.[^2_4][^2_5]

**Two ways to time travel:**

```sql
-- METHOD 1: By version number (precise, deterministic)
SELECT * FROM orders VERSION AS OF 0;   -- very first version
SELECT * FROM orders VERSION AS OF 3;   -- after 3rd commit

-- METHOD 2: By timestamp (useful for business logic)
SELECT * FROM orders TIMESTAMP AS OF '2026-04-10 09:00:00';
SELECT * FROM orders TIMESTAMP AS OF date_sub(current_date(), 7);  -- 7 days ago
```

```python
# PySpark equivalent
df_v0 = spark.read.format("delta").option("versionAsOf", 0).load("/tmp/orders_delta")
df_ts = spark.read.format("delta").option("timestampAsOf", "2026-04-10").load("/tmp/orders_delta")
```

**RESTORE — Going Back Permanently:**

```sql
-- Time travel only READS old data. RESTORE makes old version the new current:
RESTORE TABLE orders TO VERSION AS OF 2;
RESTORE TABLE orders TO TIMESTAMP AS OF '2026-04-10';
```

> 🔑 **Interview distinction:** `TIME TRAVEL` = read-only, current version unchanged. `RESTORE` = makes an old version the new HEAD. Know this difference.[^2_4]

***

## 🗑️ 4. VACUUM — The Physical File Cleaner

`VACUUM` physically deletes Parquet files that have been `remove`d in the transaction log AND are older than the retention period.[^2_6]

```sql
-- Default retention = 7 days (168 hours)
VACUUM orders;

-- Custom retention (dangerous!)
VACUUM orders RETAIN 24 HOURS;

-- DRY RUN — see what WOULD be deleted without actually deleting
VACUUM orders DRY RUN;
```


### 🚨 The Streaming Interview Trap — Know This Cold!

> **Never set retention below 7 days (168 hours) if you have active streaming readers.**[^2_6]

Here's why:

```
Streaming job checkpoint says: "I last read version 15"
Someone runs:  VACUUM orders RETAIN 2 HOURS;
                   → deletes all files older than 2 hours
                   → version 15's Parquet files are GONE 💀

Next streaming micro-batch:
   "Let me read from version 15..."
   → FileNotFoundException 💥
   → Streaming job crashes!
```

Delta's safety check: it **blocks** `VACUUM` below 7 days by default. To override (dangerous!):

```python
# Override the safety check — only do this in controlled scenarios
spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
spark.sql("VACUUM orders RETAIN 1 HOURS")
```

> 💸 **Cost note:** Running VACUUM too infrequently = storage costs pile up. Running it too aggressively = breaks streaming + loses time travel. Sweet spot: VACUUM weekly (7-day default), UNLESS your time travel SLA is longer.

***

## ⚡ 5. OPTIMIZE + Z-Ordering

### The Small File Problem First 🗂️

Every `INSERT`, `MERGE`, or streaming micro-batch writes **new small Parquet files**. Over time:

```
After 1000 micro-batches:
/orders_delta/part-0001.parquet  (5 KB)
/orders_delta/part-0002.parquet  (5 KB)
/orders_delta/part-0003.parquet  (5 KB)
... × 997 more tiny files

→ Spark opens 1000 files just to read your table
→ Massive overhead in file listing, metadata ops = SLOW 🐢
```


### OPTIMIZE — File Compaction

`OPTIMIZE` compacts many small files into fewer large files (target: ~256MB each):[^2_7]

```sql
-- Basic optimize: compacts small files
OPTIMIZE orders;

-- Optimize only a partition (faster, less resource)
OPTIMIZE orders WHERE date = '2026-04-14';
```

What happens internally:

1. Delta reads all small files in scope
2. Rewrites them as larger, right-sized Parquet files
3. Writes `add` entries for new big files + `remove` entries for old small files in `_delta_log`
4. Old small files still physically exist until VACUUM runs ← this is WHY run VACUUM after OPTIMIZE!

### Z-Ordering — Smart Data Co-location

Z-ORDER reorganizes data within Parquet files so that **related rows are physically stored together**, based on columns you commonly filter on:[^2_7]

```sql
-- OPTIMIZE + Z-ORDER together (most common in prod)
OPTIMIZE orders ZORDER BY (country, category);
```

**Visual analogy:**

```
Without Z-ORDER:                With Z-ORDER (country, category):
File 1: [USA, India, UK, ...]   File 1: [India+Electronics, India+Furniture]
File 2: [Electronics, Furn..]   File 2: [USA+Electronics, USA+Furniture]
File 3: [mixed countries...]    File 3: [UK+Electronics, UK+Furniture]

Query: WHERE country='India' AND category='Electronics'
→ Must scan ALL 3 files         → Can skip File 2, File 3 entirely ✅
```

**What Z-ORDER actually does mathematically:** It interleaves the bits of multiple column values (Z-curve/Morton curve) to assign each row a single number such that rows with similar multi-dimensional coordinates cluster together.[^2_6]

> 🚨 **Difference between OPTIMIZE and ZORDER (most asked!):**
> - `OPTIMIZE` alone = **compaction only** (size problem)
> - `OPTIMIZE + ZORDER BY` = **compaction + data co-location** (size + query speed problem)
> - Z-ORDER without OPTIMIZE doesn't exist as a standalone command — they always run together

**When NOT to use Z-ORDER:**

- High-cardinality columns that users never filter on (waste of compute)
- Columns with very low cardinality (boolean, flag columns — use partitioning instead)
- Tables that are appended-only and always read fully (no selective filtering)

***

## 🔀 6. MERGE INTO — The Most Asked Hands-On Question

`MERGE INTO` is how you do **upserts** (update if exists, insert if new) + deletes in one atomic operation.[^2_8][^2_9]

### Anatomy of MERGE:

```sql
MERGE INTO target_table AS target      -- the Delta table you're updating
USING source_table AS source           -- incoming data (DataFrame or table)
ON target.id = source.id               -- match condition (join key)

WHEN MATCHED AND source.is_deleted = true THEN DELETE    -- delete condition
WHEN MATCHED THEN UPDATE SET                             -- update condition
    target.name = source.name,
    target.revenue = source.revenue,
    target.updated_at = source.updated_at

WHEN NOT MATCHED THEN INSERT (                           -- insert new rows
    id, name, revenue, updated_at
) VALUES (
    source.id, source.name, source.revenue, source.updated_at
)
```

**Real-world MERGE scenarios:**


| Use Case | WHEN clause to use |
| :-- | :-- |
| CDC (Change Data Capture) | MATCHED + UPDATE, NOT MATCHED + INSERT |
| SCD Type 1 (overwrite) | MATCHED + UPDATE only |
| Deduplication | MATCHED (do nothing) + NOT MATCHED + INSERT |
| GDPR delete | MATCHED AND flagged → DELETE |
| Full upsert | All three clauses |


***

## 💻 Hands-On Code — Everything in One Place

### Task 1: Create Delta Table + DESCRIBE

<span style="display:none">[^2_10][^2_11][^2_12][^2_13][^2_14][^2_15]</span>

<div align="center">⁂</div>

## 💻 Hands-On Code — Everything in One Place

### Task 1: Create Delta Table + All DESCRIBE Variants

```python
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
from delta.tables import DeltaTable

schema = StructType([
    StructField("order_id",   IntegerType(), False),
    StructField("customer",   StringType(),  True),
    StructField("product",    StringType(),  True),
    StructField("category",   StringType(),  True),
    StructField("revenue",    DoubleType(),  True),
    StructField("country",    StringType(),  True),
    StructField("order_date", StringType(),  True),
])

initial_data = [
    (1, "Alice",   "Laptop",  "Electronics", 1200.0, "India", "2026-04-01"),
    (2, "Bob",     "Phone",   "Electronics",  800.0, "USA",   "2026-04-02"),
    (3, "Charlie", "Desk",    "Furniture",    350.0, "India", "2026-04-03"),
    (4, "Diana",   "Chair",   "Furniture",    200.0, "UK",    "2026-04-04"),
    (5, "Eve",     "Tablet",  "Electronics",  600.0, "USA",   "2026-04-05"),
]

df = spark.createDataFrame(initial_data, schema)

DELTA_PATH = "/tmp/day2_orders_delta"

# Write as Delta — this creates version 0
df.write.format("delta").mode("overwrite").save(DELTA_PATH)

# Register as a table in the metastore (so SQL works cleanly)
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS day2_orders
    USING DELTA
    LOCATION '{DELTA_PATH}'
""")
```

```sql
-- %sql

-- DESCRIBE: shows column names + data types
DESCRIBE TABLE day2_orders;

-- DESCRIBE EXTENDED: adds storage info, location, table type, provider
DESCRIBE TABLE EXTENDED day2_orders;

-- DESCRIBE DETAIL: Delta-specific — shows numFiles, sizeInBytes, partitionColumns, etc.
DESCRIBE DETAIL day2_orders;

-- DESCRIBE HISTORY: every commit ever made — operation, timestamp, user, version
DESCRIBE HISTORY day2_orders;
```

> 🔑 **Interview tip:** Know all 4 DESCRIBE variants. `DESCRIBE DETAIL` and `DESCRIBE HISTORY` are Delta-specific — they don't work on non-Delta tables. Interviewers love asking "how do you check how many files a Delta table has?" → `DESCRIBE DETAIL`.[^3_1][^3_2]

***

### Task 2: MERGE INTO — Full Upsert + Delete

```python
# Simulate incoming CDC data (update existing + add new + delete one)
updates_data = [
    (2, "Bob",     "Phone",   "Electronics", 950.0, "USA",   "2026-04-10"),  # revenue updated
    (4, "Diana",   "Chair",   "Furniture",   200.0, "UK",    "2026-04-04"),  # no change
    (6, "Frank",   "Monitor", "Electronics", 430.0, "India", "2026-04-10"),  # NEW row
    (7, "Grace",   "Sofa",    "Furniture",   890.0, "UK",    "2026-04-10"),  # NEW row
]

df_updates = spark.createDataFrame(updates_data, schema)
df_updates.createOrReplaceTempView("orders_updates")
```

```sql
-- %sql — MERGE INTO (version 1 created after this)
MERGE INTO day2_orders AS target
USING orders_updates AS source
ON target.order_id = source.order_id

-- If order exists AND revenue changed → update it
WHEN MATCHED AND target.revenue != source.revenue THEN
  UPDATE SET
    target.revenue    = source.revenue,
    target.order_date = source.order_date

-- If order doesn't exist in target → insert it
WHEN NOT MATCHED THEN
  INSERT (order_id, customer, product, category, revenue, country, order_date)
  VALUES (source.order_id, source.customer, source.product,
          source.category, source.revenue, source.country, source.order_date);
```

```python
# PySpark API equivalent — same MERGE but using DeltaTable builder
delta_table = DeltaTable.forPath(spark, DELTA_PATH)

(delta_table.alias("target")
    .merge(
        df_updates.alias("source"),
        "target.order_id = source.order_id"
    )
    .whenMatchedUpdate(
        condition="target.revenue != source.revenue",
        set={
            "revenue":    "source.revenue",
            "order_date": "source.order_date"
        }
    )
    .whenNotMatchedInsertAll()  # insert all columns from source
    .execute()
)
```

> ⚠️ **Performance trap:** MERGE does a full table scan on the target by default. For large tables, always add a **partition filter** in the ON condition to limit scanning: `ON target.order_id = source.order_id AND target.order_date >= '2026-04-01'`. This is a real prod optimization.[^3_3]

***

### Task 3: Time Travel — Query v0 vs Current

```python
# Make one more change to create version 2
spark.sql("""
    INSERT INTO day2_orders
    VALUES (8, 'Heidi', 'Keyboard', 'Electronics', 120.0, 'India', '2026-04-11')
""")

# Now we have: v0 (initial 5 rows), v1 (after MERGE), v2 (after INSERT)
```

```sql
-- %sql

-- Check current version (should show all rows including v2 insert)
SELECT COUNT(*) as row_count, 'CURRENT' as version FROM day2_orders

UNION ALL

-- Query version 0 — original 5 rows only
SELECT COUNT(*) as row_count, 'VERSION 0' as version
FROM day2_orders VERSION AS OF 0

UNION ALL

-- Query by timestamp — goes back to state at that exact time
SELECT COUNT(*) as row_count, 'TIMESTAMP TRAVEL' as version
FROM day2_orders TIMESTAMP AS OF '2026-04-14 00:00:00';
```

```python
# PySpark time travel
df_v0 = (spark.read.format("delta")
              .option("versionAsOf", 0)
              .load(DELTA_PATH))

df_v1 = (spark.read.format("delta")
              .option("versionAsOf", 1)
              .load(DELTA_PATH))

print(f"v0 rows: {df_v0.count()}")   # 5
print(f"v1 rows: {df_v1.count()}")   # 7 (5 original + 2 new from MERGE)
print(f"current rows: {spark.read.format('delta').load(DELTA_PATH).count()}")  # 8

# RESTORE — make v0 the new current HEAD (careful in prod!)
spark.sql(f"RESTORE TABLE day2_orders TO VERSION AS OF 0")
```


***

### Task 4: DESCRIBE HISTORY — Read the Transaction Log

```sql
-- %sql
DESCRIBE HISTORY day2_orders;
```

Output you'll see (read this in interviews! 👇):


| version | timestamp | operation | operationParameters |
| :-- | :-- | :-- | :-- |
| 3 | 2026-04-14 ... | RESTORE | `{"version": "0"}` |
| 2 | 2026-04-14 ... | WRITE | `{"mode": "Append"}` |
| 1 | 2026-04-14 ... | MERGE | `{"predicate": "..."}` |
| 0 | 2026-04-14 ... | WRITE | `{"mode": "Overwrite"}` |

```python
# Get history as a DataFrame for programmatic use
history_df = spark.sql("DESCRIBE HISTORY day2_orders")

# Filter only MERGE operations — useful for audit logs
history_df.filter(F.col("operation") == "MERGE").select(
    "version", "timestamp", "operation", "operationMetrics"
).show(truncate=False)
```

> 💡 `operationMetrics` in history contains gold: `numTargetRowsUpdated`, `numTargetRowsInserted`, `numTargetRowsDeleted`, `executionTimeMs`. In prod you log these to a monitoring table to track pipeline health.[^3_4]

***

### Task 5: OPTIMIZE + ZORDER BY

```python
# First, create small file problem by doing many small appends
for i in range(10, 20):
    small_df = spark.createDataFrame(
        [(i, f"User{i}", "Widget", "Electronics", float(i * 100), "India", "2026-04-14")],
        schema
    )
    small_df.write.format("delta").mode("append").save(DELTA_PATH)

# Check file count BEFORE optimize
spark.sql("DESCRIBE DETAIL day2_orders").select("numFiles", "sizeInBytes").show()
```

```sql
-- %sql

-- OPTIMIZE alone: just compacts small files into ~256MB files
OPTIMIZE day2_orders;

-- Check file count AFTER optimize — should be 1-2 files now
DESCRIBE DETAIL day2_orders;

-- OPTIMIZE + ZORDER: compact + co-locate data by country and category
-- Run this when you frequently filter on these columns
OPTIMIZE day2_orders ZORDER BY (country, category);

-- Partition-level optimize (faster for large tables — only touches one partition)
OPTIMIZE day2_orders WHERE order_date = '2026-04-14' ZORDER BY (country);
```

> 🔑 **How to verify Z-ORDER worked:** Run a query with `WHERE country = 'India'` **before** and **after** ZORDER, then compare the Spark UI metrics — specifically `number of files read` and `bytes read`. After ZORDER, fewer files should be scanned. This data-skipping is tracked in Delta's **data skipping statistics** stored in `_delta_log`.[^3_5]

***

### Task 6: VACUUM — Safe Cleanup

```sql
-- %sql

-- STEP 1: Always dry run first — see what WOULD be deleted
VACUUM day2_orders DRY RUN;

-- STEP 2: Run with default 7-day retention (safe)
VACUUM day2_orders;

-- STEP 3: Verify — try time travel to v0 after vacuum (should fail if old files gone)
-- This will throw AnalysisException if those files were vacuumed
SELECT * FROM day2_orders VERSION AS OF 0;
```

```python
# If you MUST reduce retention (non-streaming tables only!):
spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
spark.sql("VACUUM day2_orders RETAIN 0 HOURS")  # deletes everything not in current version
# Re-enable safety check immediately after!
spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "true")
```

> 🚨 **Post-VACUUM pro tip:** After VACUUM, time travel to old versions is **gone forever** for vacuumed files. Always communicate this with your team — never VACUUM a table that your data science team is using for reproducible model training on historical snapshots.[^3_6]

***

## 🎯 Interview Questions — Day 2 (Complete Set)

**Q1.** How does Delta Lake achieve ACID transactions?
> ✅ **Answer:** Via the `_delta_log` — every write first creates new Parquet files, then atomically commits a JSON entry to the log. If the commit fails, no entry exists = transaction never happened (Atomicity). Concurrent writers use Optimistic Concurrency Control — they check if another writer touched the same files since they started; if yes, conflict error; if no, both commits succeed.[^3_7]

**Q2.** What is in the `_delta_log` directory?
> ✅ **Answer:** JSON commit files (one per transaction) containing `add`/`remove` file actions, schema metadata, and commit info. Every 10 commits, a checkpoint `.parquet` file is written as a full state snapshot to avoid reading all JSONs from the start.[^3_8]

**Q3.** What is the difference between OPTIMIZE and ZORDER BY?
> ✅ **Answer:** `OPTIMIZE` = file **compaction** — solves the small files problem by merging many small Parquet files into fewer large (~256MB) ones. `ZORDER BY` = **data co-location** — reorganizes rows within files using a Z-curve so rows with similar column values are physically adjacent, enabling data skipping for range queries. They always run together (`OPTIMIZE ... ZORDER BY`).[^3_5]

**Q4.** Why should you never VACUUM below 7 days if you have streaming?
> ✅ **Answer:** Streaming jobs use checkpoints that track which Delta version they last read. If VACUUM deletes files that the streaming job's checkpoint points to, the next micro-batch throws `FileNotFoundException` and the stream crashes. 7 days gives streaming enough buffer to lag and catch up.[^3_6]

**Q5.** What's the difference between `VERSION AS OF` and `RESTORE`?
> ✅ **Answer:** `VERSION AS OF` is **read-only time travel** — reads old data but current table version is unchanged. `RESTORE TABLE TO VERSION AS OF` **changes the table HEAD** — it makes the old version the new current state (creates a new commit in the log).[^3_9]

**Q6.** What does `DESCRIBE HISTORY` show? What's in `operationMetrics`?
> ✅ **Answer:** Every commit: version number, timestamp, operation type (WRITE/MERGE/OPTIMIZE/VACUUM), user, cluster ID. `operationMetrics` has row-level counts: `numOutputRows`, `numTargetRowsUpdated`, `numTargetRowsInserted`, `numTargetRowsDeleted`, `executionTimeMs` — critical for pipeline monitoring.[^3_4]

**Q7.** What is Optimistic Concurrency Control in Delta?
> ✅ **Answer:** Delta assumes conflicts are rare. Writers don't lock the table upfront. At commit time, Delta checks if any other writer modified files you read since you started. If YES → conflict exception (retry logic handles this). If NO → both commits succeed at different version numbers. This allows high concurrency without table-level locking.

**Q8.** How does Delta's data skipping work with Z-ORDER?
> ✅ **Answer:** Delta stores min/max statistics for each Parquet file in the `_delta_log`. For a `WHERE country = 'India'` query, Delta checks the stats — if a file's `country` range is `[UK, USA]`, it **skips that file entirely** without opening it. Z-ORDER physically clusters similar values together, making these min/max stats more effective (smaller ranges per file = more files skipped).[^3_5]

**Q9.** What happens physically when you run MERGE INTO?
> ✅ **Answer:** Delta reads matching files from the target, applies the merge logic, writes new Parquet files with the results, then commits `add` entries for new files + `remove` entries for old files to `_delta_log`. Old files remain physically until VACUUM. The whole operation is atomic — either all changes commit or none do.

**Q10.** Can you run time travel after VACUUM? Why or why not?
> ✅ **Answer:** Only to versions **within the retention window**. VACUUM physically deletes old Parquet files. Time travel needs those physical files — if they're gone, Delta throws `FileNotFoundException`. Time travel to recent versions (within retention period) still works because their files haven't been vacuumed yet.

***

## 🗺️ Day 2 Mental Model — How It All Connects

```
Every write ──────────────────────────────────────────────┐
                                                           ▼
INSERT / MERGE / UPDATE  →  new Parquet files  →  _delta_log commit JSON
                                                     (add/remove entries)
                                   │
              ┌────────────────────┼───────────────────────┐
              ▼                    ▼                        ▼
        TIME TRAVEL           ACID guarantee          DESCRIBE HISTORY
    (old files still exist    (OCC + atomic           (audit trail of
     until VACUUMed)           commits)                every version)
              │
              ▼
         VACUUM ──── removes old files (respect 7-day rule ⚠️)
         OPTIMIZE ── compact small files (run after VACUUM)
         ZORDER ──── co-locate data for query speed 🚀
```


***

> 🔥 **Tonight's challenge Sagar:** After running all tasks, open `_delta_log/` using `dbutils.fs.ls("/tmp/day2_orders_delta/_delta_log/")` and actually **read one of the JSON commit files** with `dbutils.fs.head(...)`. You'll see the raw `add`/`remove` entries with file stats. Seeing it once makes the theory stick forever. This also makes a great interview story — "I've actually read the transaction log by hand." 💪
<span style="display:none">[^3_10][^3_11][^3_12][^3_13][^3_14][^3_15][^3_16][^3_17][^3_18][^3_19][^3_20][^3_21][^3_22]</span>

<div align="center">⁂</div>


---

# 🗓️ DAY 3 — Spark Performance \& Optimization ⚡

**TL;DR** — Spark performance = minimize shuffles, right-size partitions, cache smart, let AQE do the heavy lifting. Every interview question on this day maps back to one root cause: **unnecessary data movement across the network**. 🌐

***

## 🔀 1. Narrow vs Wide Transformations

This is the **fundamental split** that determines whether Spark crosses a stage boundary or not.[^4_1]

**Narrow Transformation** — each input partition maps to **exactly one** output partition. No data moves across the network. Fast, parallelizable, failure-recoverable per partition.[^4_2]

**Wide Transformation** — one input partition feeds **multiple** output partitions. Spark must **shuffle** data across executors. This costs: network I/O + disk spill + serialization. It creates a **stage boundary** in the DAG.[^4_1]

```
NARROW (no shuffle — same stage ✅)        WIDE (shuffle — new stage 🚨)
─────────────────────────────────         ──────────────────────────────
filter()       → just drops rows           groupBy().agg()  → keys scatter
select()       → drops columns             join()           → co-locate keys
map()          → row-level transform       orderBy()        → global sort
withColumn()   → add/modify a column       distinct()       → cross-partition dedup
union()        → stack partitions (✅!)    repartition()    → full reshuffle
flatMap()      → 1 row → many rows         intersection()   → compare all partitions

⚠️ union() is narrow — just stacks. intersection() is WIDE — needs shuffle!
```

> 💡 **Why this matters for performance:** Every wide transformation = a stage boundary = Spark **writes shuffle data to disk** before the next stage reads it. Disk I/O is the enemy. If you chain 5 wide transforms, that's 5 disk write-read cycles. Minimize wide transforms or batch them where possible.[^4_3]

***

## 🗂️ 2. Partitions: `repartition()` vs `coalesce()`

Both change the number of partitions — but their internals are completely different.[^4_4]


| Feature | `repartition(n)` | `coalesce(n)` |
| :-- | :-- | :-- |
| **Direction** | ↑ increase OR ↓ decrease | ↓ decrease ONLY |
| **Shuffle?** | ✅ Yes — full shuffle | ❌ No — just merges existing partitions |
| **Data distribution** | Evenly balanced ✅ | Can be unbalanced ⚠️ |
| **Speed** | Slower (network cost) | Much faster (local merge) |
| **Use case** | Before a big join/group, after filter that killed partitions | Before writing output, reduce file count cheaply |

```python
# repartition: use when you NEED even distribution + don't mind shuffling
# e.g. before a big join to ensure even key distribution
df_big = df.repartition(200, "country")  # hash-partition by country

# coalesce: use when reducing for output (e.g. write 1 file per partition)
# no shuffle = fast, but partitions may be unequal in size
df_small = df.coalesce(4)  # merge 200 partitions down to 4, no shuffle

# RULE OF THUMB:
# Going DOWN and don't care about balance → coalesce()
# Going DOWN and need balance → repartition()
# Going UP → always repartition() (coalesce going up = same as repartition)
```

> 🚨 **Common mistake:** Using `repartition(1)` to write a single file in production. Yes it works, but it **forces all data to ONE executor** — this is a single point of failure and bottleneck on large datasets. Instead use `coalesce(1)` (less shuffle) OR better — write with `maxRecordsPerFile` and let Delta handle file management.[^4_5]

***

## 🧠 3. `cache()` vs `persist()` — Storage Levels

`cache()` is just `persist()` with a default storage level. But the **storage level** is what actually matters in production.[^4_6]

```python
from pyspark import StorageLevel

# cache() = shorthand for MEMORY_AND_DISK (Spark 3.x default for DataFrames)
df.cache()

# persist() = full control over storage level
df.persist(StorageLevel.MEMORY_ONLY)          # fastest but OOM risk
df.persist(StorageLevel.MEMORY_AND_DISK)      # spills to disk if RAM full ✅ prod default
df.persist(StorageLevel.MEMORY_ONLY_SER)      # serialized in memory — less RAM, CPU cost
df.persist(StorageLevel.DISK_ONLY)            # only on disk — slow, but massive datasets
df.persist(StorageLevel.OFF_HEAP)             # off-heap memory — avoids GC pressure

# Always unpersist when done — Spark won't auto-evict until OOM pressure!
df.unpersist()
```


### When to cache ✅ vs when NOT to ❌

```
CACHE WHEN:                              DON'T CACHE WHEN:
─────────────────────────────────        ──────────────────────────────────
DataFrame used 3+ times in pipeline      Used only once (wasted memory)
Iterative ML algorithms (loop reads)     Bigger than available executor RAM
Expensive join result reused             Write-once-read-once ETL pipeline
After a heavy transformation chain       Streaming DataFrames (unsupported)
Shared across multiple downstream ops    Data changes between uses (stale!)
```

> 💸 **Cost flag:** Caching on Databricks consumes executor memory = keeps cluster alive longer = more DBUs. Always `unpersist()` explicitly after use. For Databricks specifically, Delta caching (disk-level SSD cache) is often better than Spark caching for tables — it persists across jobs.

***

## 📡 4. Broadcast Join — Eliminate the Shuffle Entirely

In a regular join, Spark shuffles BOTH tables to co-locate matching keys. Broadcast join sends the **entire small table to every executor** — so the big table never moves.[^4_7]

```
SHUFFLE JOIN (both tables move 🐢):
Executor 1: [big_table partition 1] ←→ [small_table partition 1] ← network!
Executor 2: [big_table partition 2] ←→ [small_table partition 2] ← network!

BROADCAST JOIN (small table broadcast, big table stays put 🚀):
Driver: reads entire small_table → sends copy to EVERY executor
Executor 1: [big_table partition 1] ← looks up in local copy of small_table
Executor 2: [big_table partition 2] ← looks up in local copy of small_table
No shuffle! ✅
```

```python
from pyspark.sql.functions import broadcast

# METHOD 1: Explicit broadcast hint (override Spark's decision)
result = big_df.join(broadcast(small_df), on="country", how="left")

# METHOD 2: Let Spark auto-broadcast (default threshold = 10MB)
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 10 * 1024 * 1024)  # 10MB

# METHOD 3: Increase threshold for slightly larger lookup tables
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 50 * 1024 * 1024)  # 50MB

# METHOD 4: Disable broadcast entirely (useful for testing)
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
```

```python
# Verify in explain() — look for "BroadcastHashJoin" vs "SortMergeJoin"
result.explain(mode="formatted")
# With broadcast → BroadcastHashJoin ✅
# Without → SortMergeJoin (shuffle happens) ❌
```

**Broadcast join limits \& tradeoffs:**


| Aspect | Detail |
| :-- | :-- |
| Default threshold | 10MB (configurable) |
| Max safe size | ~200-300MB (beyond this, driver OOM risk) |
| Join types supported | Inner, Left (broadcast right side) — NOT right outer |
| Failure mode | Driver OOM if table too large → whole job fails |
| When AQE helps | AQE can **auto-convert** SortMergeJoin → BroadcastHashJoin at runtime |


***

## ⚖️ 5. Data Skew — The Silent Performance Killer

**Data skew** = some partitions have WAY more data than others. One "hot key" drowns one executor while others sit idle.[^4_8]

### How to Spot Skew in Spark UI:

```
Stage 5 — 200 tasks:
Task  1: 2 sec   ████
Task  2: 3 sec   ██████
Task  3: 2 sec   ████
...
Task 167: 4 min  ████████████████████████████████████  ← THIS IS SKEW 🚨
Task 168: 2 sec  ████
```

One task takes 60x longer than others = one partition has 60x more data.

### Fix 1: Salting — Break Up the Hot Key

```python
import random
from pyspark.sql import functions as F

# PROBLEM: 70% of data has country='India' — one executor gets crushed
# SOLUTION: Add a random "salt" to split India into N buckets

SALT_BUCKETS = 20  # tune based on skew severity (see table below)

# Step 1: Salt the BIG table — add random salt 0–19
df_big_salted = df_big.withColumn(
    "salt",
    (F.rand() * SALT_BUCKETS).cast("int")
).withColumn(
    "join_key_salted",
    F.concat(F.col("country"), F.lit("_"), F.col("salt"))
)

# Step 2: EXPLODE the small table — replicate for all salt values
# This is the cost: small table grows 20x in memory
salt_values = spark.range(SALT_BUCKETS).withColumnRenamed("id", "salt")
df_small_salted = df_small.crossJoin(salt_values).withColumn(
    "join_key_salted",
    F.concat(F.col("country"), F.lit("_"), F.col("salt"))
)

# Step 3: Join on salted key — India now split across 20 partitions
result = df_big_salted.join(df_small_salted, on="join_key_salted", how="inner")

# Step 4: Drop the artificial salt columns
result = result.drop("salt", "join_key_salted")
```

**Salt bucket guide:**[^4_8]


| Skew Severity | Example | Salt Buckets |
| :-- | :-- | :-- |
| Mild | 1 key = 5–10% of data | 5 |
| Moderate | 1 key = 10–30% | 10–20 |
| Severe | 1 key = 30–60% | 20–50 |
| Extreme | 1 key = >60% | 50–100 |

### Fix 2: AQE Skew Join (Spark 3.x — Automatic!)

If AQE is enabled, Spark can **automatically detect and split skewed partitions** at runtime without you writing salting code:

```python
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
# AQE detects tasks > 5x median AND > 256MB → splits them automatically
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")
```


***

## 🤖 6. AQE — Adaptive Query Execution (Spark 3.2+ Default)

AQE is Spark's **runtime self-optimizer** — it re-plans the query MID-EXECUTION using actual runtime statistics instead of static estimates. Enabled by default from Spark 3.2.[^4_9][^4_6]

AQE has **3 superpowers**:

### Superpower 1: Coalesce Post-Shuffle Partitions

```
BEFORE execution: spark.sql.shuffle.partitions = 200 (your setting)
AFTER shuffle stage: AQE sees only 5MB of actual shuffle data
AQE says: "200 partitions for 5MB is insane — let me coalesce to 4"
Result: 4 tasks instead of 200 ✅ (massive overhead reduction)
```


### Superpower 2: Convert SortMerge → Broadcast Join

```
Spark's static estimate: "table_B = 500MB, use SortMergeJoin"
Runtime reality: after filters, table_B = 8MB
AQE says: "8MB < broadcast threshold — switch to BroadcastHashJoin!"
Result: shuffle eliminated at runtime ✅
```


### Superpower 3: Skew Join Optimization

```
Runtime: AQE detects partition 47 has 10GB, others have 200MB each
AQE says: "split partition 47 into 50 sub-tasks"
Result: no single executor stuck on one hot partition ✅
```

```python
# Full AQE config in production
spark.conf.set("spark.sql.adaptive.enabled", "true")                           # master switch
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")        # superpower 1
spark.conf.set("spark.sql.adaptive.coalescePartitions.minPartitionSize", "64m") # min size per coalesced partition
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")      # target size
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")                  # superpower 3

# Verify AQE is active — look for "AdaptiveSparkPlan" at the top of explain()
df.explain(mode="formatted")
```

> 💡 Even with AQE ON, you should still set `spark.sql.shuffle.partitions` sensibly. AQE coalesces DOWN from your setting — it doesn't increase it. So if you set 10 and actually need 200, AQE can't help you go up.[^4_9]

***

## ⚙️ 7. `spark.sql.shuffle.partitions` — Tune This Every Job

Default is **200** — a completely arbitrary number from legacy Hadoop days.[^4_6]

```python
# Default 200 — almost always wrong
spark.conf.set("spark.sql.shuffle.partitions", "200")

# Rule of thumb: target ~128MB–256MB per partition after shuffle
# Formula: shuffle_data_size / target_partition_size
# With AQE: set to max you might need, AQE will coalesce down

# Example tuning:
# Small dataset (< 1GB) → 4-8 partitions
spark.conf.set("spark.sql.shuffle.partitions", "8")

# Medium dataset (1-50GB) → num_cores × 2 or 4
# If cluster has 20 cores: 40–80 partitions
spark.conf.set("spark.sql.shuffle.partitions", "40")

# Large dataset (100GB+) → data_size_MB / 200 (target 200MB/partition)
# 500GB / 200MB = 2500 partitions
spark.conf.set("spark.sql.shuffle.partitions", "2500")
```

> ✅ **With AQE enabled:** Set shuffle.partitions to the **maximum** you might need. AQE will automatically coalesce down at runtime. This is the modern recommended approach — one setting, AQE handles the rest.[^4_6]

***

## 💻 Hands-On Code — All Tasks

### Task 1: explain() — With vs Without Broadcast

```python
from pyspark.sql.functions import broadcast, col, rand
from pyspark.sql.types import *

# Create sample DataFrames
orders_data = [(i, f"product_{i % 50}", ["India","USA","UK","Germany","Brazil"][i % 5], float(i * 10))
               for i in range(1, 100001)]
orders_schema = StructType([
    StructField("order_id", IntegerType()),
    StructField("product",  StringType()),
    StructField("country",  StringType()),
    StructField("revenue",  DoubleType()),
])

country_data = [("India","Asia","INR"), ("USA","Americas","USD"),
                ("UK","Europe","GBP"), ("Germany","Europe","EUR"), ("Brazil","Americas","BRL")]
country_schema = StructType([
    StructField("country",  StringType()),
    StructField("region",   StringType()),
    StructField("currency", StringType()),
])

df_orders  = spark.createDataFrame(orders_data, orders_schema)
df_country = spark.createDataFrame(country_data, country_schema)

# WITHOUT broadcast — Spark may choose SortMergeJoin
print("=== WITHOUT BROADCAST ===")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")  # disable auto-broadcast
df_orders.join(df_country, on="country").explain(mode="formatted")
# Look for: SortMergeJoin ← shuffle on both sides

# WITH broadcast — explicit hint
print("=== WITH BROADCAST ===")
df_orders.join(broadcast(df_country), on="country").explain(mode="formatted")
# Look for: BroadcastHashJoin ← no shuffle!

# Re-enable auto-broadcast
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", str(10 * 1024 * 1024))
```


### Task 2: Force + Observe Data Skew

```python
# Create HEAVILY skewed data — 80% of rows have country='India'
import random
random.seed(42)

skewed_data = []
for i in range(1, 50001):
    country = "India" if random.random() < 0.8 else random.choice(["USA", "UK", "Germany"])
    skewed_data.append((i, country, float(i * 10)))

df_skewed = spark.createDataFrame(skewed_data, ["order_id", "country", "revenue"])

# Disable AQE to see RAW skew
spark.conf.set("spark.sql.adaptive.enabled", "false")
spark.conf.set("spark.sql.shuffle.partitions", "10")

# This groupBy will show massive skew in Spark UI
df_skewed.groupBy("country").count().show()
# → Open Spark UI → Stages → look at task duration bar chart
# You'll see 1 task bar WAY longer than others

# Now enable AQE and re-run — compare
spark.conf.set("spark.sql.adaptive.enabled", "true")
df_skewed.groupBy("country").count().show()
# → Spark UI now shows balanced tasks (AQE split the India partition)
```


### Task 3: Salting — Manual Skew Fix

```python
SALT_BUCKETS = 10

# Salt the large skewed table
df_salted = df_skewed.withColumn(
    "salt", (rand() * SALT_BUCKETS).cast("int")
).withColumn(
    "country_salted", col("country").cast("string").concat(
        F.lit("_"), col("salt").cast("string"))
)

# Explode the small lookup table across salt values
salt_df = spark.range(SALT_BUCKETS).withColumnRenamed("id", "salt")
df_country_salted = df_country.crossJoin(salt_df).withColumn(
    "country_salted", col("country").cast("string").concat(
        F.lit("_"), col("salt").cast("string"))
)

# Join on salted key — India now split across 10 partitions
result = (df_salted
    .join(df_country_salted, on="country_salted", how="left")
    .drop("salt", "country_salted")
)

result.explain(mode="formatted")
```


### Task 4: Benchmark repartition() vs coalesce()

```python
import time

# Generate a medium dataset
df_large = spark.range(5_000_000).withColumn("val", rand())
df_large.cache()
df_large.count()  # materialize cache

# Benchmark repartition() — triggers FULL shuffle
t1 = time.time()
df_large.repartition(10).write.format("noop").mode("overwrite").save()
t2 = time.time()
print(f"repartition(10):  {t2-t1:.2f} sec")  # slower — full shuffle

# Benchmark coalesce() — NO shuffle, just merges
t3 = time.time()
df_large.coalesce(10).write.format("noop").mode("overwrite").save()
t4 = time.time()
print(f"coalesce(10):     {t4-t3:.2f} sec")  # faster — no shuffle

# Check partition sizes after coalesce (may be unbalanced!)
df_coalesced = df_large.coalesce(4)
print("Rows per partition after coalesce:")
df_coalesced.rdd.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()

df_large.unpersist()
```


### Task 5: AQE Full Config + Verify

```python
# Full AQE setup for production
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.minPartitionSize", "64m")
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
spark.conf.set("spark.sql.shuffle.partitions", "200")

# With AQE — explain shows "AdaptiveSparkPlan isFinalPlan=false" before execution
# After execution — shows "AdaptiveSparkPlan isFinalPlan=true" with ACTUAL plan
df_orders.join(broadcast(df_country), on="country") \
         .groupBy("region") \
         .count() \
         .explain(mode="formatted")
```

> 💡 **Spark UI trick for AQE:** In the SQL tab, click a query and look at the Physical Plan. You'll see two plans: **initial plan** (static estimate) and **final plan** (what AQE actually ran). Comparing these is powerful for understanding what AQE changed. This makes a great interview story! 🎯

***

## 🎯 Interview Questions — Day 3 (Complete Set)

**Q1.** What is the difference between a narrow and wide transformation?
> ✅ **Answer:** Narrow = each input partition maps to one output partition, no shuffle, same DAG stage (filter, select, map). Wide = input partitions map to multiple output partitions, requires shuffle across network, creates stage boundary (groupBy, join, orderBy). Shuffles cause disk I/O + network cost = main performance concern.[^4_1]

**Q2.** When would you use `repartition()` vs `coalesce()`?
> ✅ **Answer:** `coalesce()` when reducing partitions — it merges partitions locally, no shuffle, faster but may be unbalanced. `repartition()` when you need even distribution OR increasing partitions — full shuffle but balanced. Classic use: `coalesce(1)` before writing output files, `repartition(200, "key")` before a big join.[^4_4]

**Q3.** What is AQE and what are its three main features?
> ✅ **Answer:** Adaptive Query Execution — re-plans query at runtime using actual statistics (not static estimates). Three features: 1) Coalesce post-shuffle partitions (200 → 4 if data is small), 2) Convert SortMergeJoin → BroadcastHashJoin if runtime table size < threshold, 3) Skew join optimization — splits oversized partitions. Default ON in Spark 3.2+.[^4_9][^4_6]

**Q4.** What is data skew? How do you fix it?
> ✅ **Answer:** Data skew = some partitions have far more data than others (e.g., 80% rows share one key). One executor task runs 60x longer than others, wasting cluster. Fixes: 1) Salting — add random suffix to hot keys, explode small table to match, join on salted key, 2) AQE skew join (automatic in Spark 3.2+), 3) Broadcast join if small table is involved.[^4_8]

**Q5.** When should you NOT use a broadcast join?
> ✅ **Answer:** When the "small" table isn't actually small — broadcasting a 500MB table means every executor gets a 500MB copy = driver/executor OOM risk. Also avoid broadcasting in joins where the driver needs to collect the entire table (memory bottleneck). Right outer joins can't broadcast the right side. If AQE is on, let it decide dynamically — it's safer.[^4_7]

**Q6.** What is `spark.sql.shuffle.partitions` and how do you tune it?
> ✅ **Answer:** Controls number of partitions after a shuffle (groupBy, join). Default 200 (legacy Hadoop number). Too high = too many small tasks + overhead. Too low = too few tasks, data spills to disk. Formula: target 128-256MB per partition. With AQE enabled, set to the MAX you might need — AQE will coalesce down automatically.

**Q7.** What is the difference between `cache()` and `persist()`?
> ✅ **Answer:** `cache()` = `persist(StorageLevel.MEMORY_AND_DISK)` — it's a convenience shorthand. `persist()` gives you full control over storage level: MEMORY_ONLY (fastest but OOM risk), MEMORY_AND_DISK (safe default), DISK_ONLY (slow, huge datasets), OFF_HEAP (avoids JVM GC). Always call `unpersist()` when done — Spark won't auto-free until memory pressure.

**Q8.** When would you NOT use autoscaling? (Day 1 callback ♻️)
> ✅ **Answer:** Structured Streaming (autoscaling messes with micro-batch timing), fine-tuned shuffle partition jobs (adding nodes mid-job can cause partition plan mismatch), when you need deterministic SLAs (scale-up latency adds variance). Fixed job clusters with tuned `shuffle.partitions` = predictable cost + runtime.[^4_6]

**Q9.** What's the difference between a sort-merge join and a broadcast hash join in Spark?
> ✅ **Answer:** Sort-merge join: both tables shuffled + sorted by join key, then merged — works for any size, but 2 full shuffles. Broadcast hash join: small table broadcast to all executors as a hash map, big table scans locally — zero shuffle, much faster, but limited by small table size (memory). AQE can convert between them at runtime.

**Q10.** How does AQE decide to coalesce shuffle partitions?
> ✅ **Answer:** After a shuffle stage completes, AQE reads the **actual byte size** of each partition from shuffle metadata. It then greedily merges consecutive small partitions until the merged size reaches `advisoryPartitionSizeInBytes` (default 64MB). So 200 partitions with 1MB each → coalesced to ~13 partitions of ~15MB. This happens between stages, not within.[^4_9]

***

## 🗺️ Day 3 Mental Model

```
EVERY PERF PROBLEM IN SPARK
          │
    Is it a shuffle problem?
    ┌─────┴──────┐
   YES            NO
    │              └──► Is it a partition size problem?
    ▼                    ┌────────────┴──────────────┐
Minimize shuffles       Too many (small)        Too few (large)
├── Use narrow ops         coalesce() ✅          repartition() ✅
├── Broadcast small        AQE coalesces          Increase shuffle.partitions
│   tables (<200MB)        automatically
├── Salt hot keys     
└── Enable AQE             Is the same DF computed multiple times?
    (auto-handles                    └─→ cache() / persist() it!
     coalesce +
     skew + 
     broadcast)
```


***

---

# 🗓️ DAY 4 — Incremental Ingestion \& Structured Streaming ⚡

**TL;DR** — Auto Loader = scalable incremental file ingestion with schema intelligence. Structured Streaming = continuous micro-batch processing with exactly-once guarantees. Checkpointing + watermarking = fault tolerance + late data handling. This is the backbone of every real-time lakehouse pipeline. 🏗️

***

## 🔄 1. Auto Loader (`cloudFiles`) — Incremental File Ingestion

Auto Loader is Databricks' managed incremental ingestion engine. It **tracks which files have already been processed** so every new pipeline run only picks up new files — not the whole bucket.[^5_1]

### How Auto Loader Tracks Files — Two Modes

```
MODE 1: Directory Listing (default)
─────────────────────────────────
Auto Loader lists S3/ADLS bucket at every trigger
Compares against what it already processed (stored in checkpoint)
Cheap for small buckets, scales poorly for millions of files

MODE 2: File Notification (recommended for scale 🚀)
──────────────────────────────────────────────────
S3 → SNS → SQS  (AWS)
ADLS → Event Grid → Event Hub  (Azure)
Auto Loader subscribes to queue — gets PUSH notification per new file
No directory listing needed → scales to millions of files easily
Set via: .option("cloudFiles.useNotifications", "true")
```

> 💡 **When to switch to File Notification mode:** When your landing zone gets **10,000+ files/hour** or directory listing starts taking more than a few seconds per trigger. This is a real prod flag — directory listing on a bucket with 5M files is catastrophically slow.[^5_2]

### Schema Inference \& the `_schemas` Directory

When Auto Loader first runs, it **samples up to 50GB or 1000 files** (whichever comes first) to infer schema. It stores that inferred schema in a `_schemas` subdirectory of your `schemaLocation`.[^5_1]

```python
df_stream = (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.schemaLocation", "/checkpoints/orders/_schemas")  # where to store inferred schema
    .option("header", "true")
    .option("inferSchema", "true")
    .load("s3://my-bucket/landing/orders/")
)
```

> 🔑 `schemaLocation` ≠ `checkpointLocation`. Schema location stores the **inferred/evolved schema**. Checkpoint location stores **streaming progress** (which files processed, offsets). Two separate paths, both required![^5_3]

***

## 🔬 2. Schema Evolution Modes — Know All 4

This is a **key interview topic** — interviewers ask "what happens when a new column appears in your incoming files?"[^5_4][^5_1]


| Mode | What happens on new column | Use case |
| :-- | :-- | :-- |
| `addNewColumns` ✅ | Stream fails → restarts → adds new col to schema | Most common prod setting — graceful evolution |
| `rescue` | Unknown columns go into a special `_rescued_data` JSON column | When you want to capture unexpected data without failing |
| `none` | New columns silently ignored / dropped | Strict schema contracts (you control the source) |
| `failOnNewColumns` | Stream **hard fails**, won't restart automatically | Quality enforcement — force human review on change |

```python
# addNewColumns: auto-evolve (stream restarts once per schema change)
df_stream = (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", "/checkpoints/events/_schemas")
    .option("cloudFiles.schemaEvolutionMode", "addNewColumns")   # 👈 key option
    .option("cloudFiles.inferColumnTypes", "true")               # infer int/double vs string
    .load("s3://my-bucket/landing/events/")
)
```

```python
# rescue mode: catch unknown columns in _rescued_data col
df_stream = (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", "/checkpoints/events/_schemas")
    .option("cloudFiles.schemaEvolutionMode", "rescue")
    .option("rescuedDataColumn", "_rescued_data")               # catches new/unexpected columns as JSON string
    .load("s3://my-bucket/landing/events/")
)
# _rescued_data looks like: '{"new_field": "value", "another_new": 123}'
```

> ⚠️ **Schema evolution flow with `addNewColumns`:** New col detected → stream **intentionally fails with `UnknownFieldException`** → Databricks Job auto-restarts → on restart, new schema (with new col) is loaded from `_schemas` dir → stream continues. Configure your Databricks Job with `"onFailure": "restart"` for this to be seamless.[^5_4]

***

## ⚖️ 3. Auto Loader vs COPY INTO — Side-by-Side

| Feature | 🔵 Auto Loader | 🟢 COPY INTO |
| :-- | :-- | :-- |
| **Scale** | ✅ Millions of files (file notification mode) | ⚠️ Thousands of files max |
| **Schema evolution** | ✅ Native — 4 modes | ❌ Manual — you manage it |
| **API type** | Structured Streaming (readStream) | SQL command |
| **File tracking** | Checkpoint + `_schemas` dir | Internal metadata table |
| **Idempotent?** | ✅ Yes | ✅ Yes |
| **Reprocess specific files** | ❌ Hard — need to reset checkpoint | ✅ Easy — just specify FILES= |
| **Incremental?** | ✅ Streaming (always incremental) | ✅ Batch incremental |
| **When to use** | Continuous/scheduled large-scale ingestion | Simple one-time or small-batch loads |
| **Complexity** | Medium | Low |

```sql
-- COPY INTO: simple, SQL-based, idempotent
COPY INTO orders_delta
FROM 's3://my-bucket/landing/orders/'
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
COPY_OPTIONS ('mergeSchema' = 'true');
-- Re-run this 10 times → only processes NEW files each time ✅
-- Files already loaded are tracked in Delta's internal metadata
```

> 💡 **COPY INTO `FORCE` option:** `COPY_OPTIONS ('force' = 'true')` reprocesses ALL files even if already loaded. Use for backfill scenarios only — it breaks idempotency. This is a prod trap interviewers love.[^5_5]

***

## 🌊 4. Structured Streaming — The Full Mental Model

Structured Streaming treats a **live data source as an infinite, ever-growing table**. You write normal DataFrame operations on it — Spark handles the micro-batch execution loop.[^5_6]

```
Infinite Input Table (S3, Kafka, Delta)
          │
          │  new rows arrive every micro-batch
          ▼
   ┌──────────────────┐
   │  Your transforms │  filter, join, agg, withColumn...
   └──────────────────┘
          │
          ▼
   Result Table (updated every trigger)
          │
          ▼
     Sink (Delta, Kafka, console...)
```


### The Three Output Modes

```python
# APPEND: only new rows written to sink — most common for raw ingestion
.writeStream.outputMode("append")

# COMPLETE: entire result table rewritten every batch — only for aggregations
.writeStream.outputMode("complete")

# UPDATE: only changed rows written — efficient for aggregations with Delta sink
.writeStream.outputMode("update")
```

> 🚨 **Output mode trap:** `append` mode **cannot be used** with aggregations that don't have watermarks (Spark doesn't know when an aggregate is "final" without a watermark). Use `update` or `complete` for aggregations, or add a watermark first.[^5_7]

***

## ⏰ 5. Triggers — Control When Batches Fire

Four trigger types — interviewers ask the difference between `Once` and `availableNow` constantly.[^5_8][^5_6]

```python
from pyspark.sql.trigger import Trigger

# TRIGGER 1: Default micro-batch — runs as fast as possible (next batch starts when prev ends)
.trigger()  # no argument = default

# TRIGGER 2: Fixed interval — wait N time between batch starts
.trigger(processingTime="1 minute")   # fire every 60s
.trigger(processingTime="30 seconds") # fire every 30s

# TRIGGER 3: Once — process ALL available data in ONE batch, then stop
# ⚠️ DEPRECATED in newer Spark — use availableNow instead
.trigger(once=True)

# TRIGGER 4: AvailableNow — process all available data in MULTIPLE batches, then stop ✅
# Respects rate limits, better for large backlogs, supports checkpointing per batch
.trigger(availableNow=True)

# TRIGGER 5: Continuous — experimental, millisecond latency (not for production)
.trigger(continuous="1 second")  # ⚠️ very limited operator support
```


### `trigger(once=True)` vs `trigger(availableNow=True)` — The Key Difference[^5_9]

```
trigger(once=True):
──────────────────
All pending data → ONE giant micro-batch → done
Problem: if 10GB of data pending, one 10GB task
         OOM risk, no checkpoint granularity per file
         DEPRECATED in Spark 3.3+

trigger(availableNow=True):
───────────────────────────
All pending data → multiple right-sized micro-batches → done
Respects maxFilesPerTrigger / maxBytesPerTrigger limits
Each batch checkpointed separately → fault tolerant
If job dies midway, resumes from last checkpoint ✅
✅ USE THIS instead of trigger(once=True) always!
```


***

## 🔒 6. Checkpointing — Fault Tolerance \& Exactly-Once

Checkpointing is how Structured Streaming achieves **exactly-once** end-to-end semantics. It stores two things:[^5_6]

```
checkpoint_path/
├── offsets/          ← what data was READ (Kafka offset, file list, Delta version)
│   ├── 0             ← batch 0 offsets
│   ├── 1             ← batch 1 offsets
│   └── 2             ← batch 2 offsets
├── commits/          ← what data was WRITTEN SUCCESSFULLY
│   ├── 0             ← batch 0 committed
│   └── 1             ← batch 1 committed
├── sources/          ← source state (Auto Loader file tracking)
└── state/            ← stateful operation state (aggregations, dedup)
```

**Exactly-once guarantee mechanism:**

```
Batch 3 starts:
1. Write offset (what we're about to read) → offsets/3
2. Read + transform data
3. Write to Delta sink (Delta's transaction log = idempotent)
4. Write commit → commits/3

If crash between step 3 and 4:
→ On restart: offset/3 exists but commits/3 doesn't
→ Spark replays batch 3 from offsets
→ Delta's idempotent write deduplicates → no duplicate data ✅

If crash between step 1 and 2:
→ On restart: offset/3 exists, commits/3 doesn't
→ Replay batch 3 ✅
```

> 🚨 **Never share checkpoint directories between two streaming queries!** They'll corrupt each other's offset tracking. One stream = one checkpoint path, always. And never delete a checkpoint unless you're intentionally doing a full reprocess from scratch.[^5_10]

***

## 🌊 7. Watermarking — Handling Late-Arriving Data

**Event time** = timestamp when event happened (in the data). **Processing time** = when Spark processes it. These are rarely the same — network lag, retries, or offline devices cause **late data**.[^5_11][^5_7]

```
Timeline:
10:00 AM — Event happens on mobile device
10:00–10:45 AM — Device is offline
10:45 AM — Device reconnects, event reaches Kafka
10:45 AM — Spark processes it... but it BELONGS to the 10:00 window!
```

Without watermarking, Spark keeps state for **all windows forever** → memory explodes 💥

**Watermark = "I'll accept late data up to X time behind the latest event I've seen"**

```python
from pyspark.sql import functions as F

# withWatermark tells Spark: "event_time is the event-time field, accept up to 10 min late"
df_watermarked = (df_stream
    .withWatermark("event_time", "10 minutes")  # discard anything > 10min late
    .groupBy(
        F.window("event_time", "5 minutes"),   # 5-minute tumbling window
        F.col("country")
    )
    .agg(
        F.count("*").alias("event_count"),
        F.sum("revenue").alias("total_revenue")
    )
)
```

**How watermark advances:**

```
Batch 1 max event_time seen: 10:30 → watermark = 10:30 - 10min = 10:20
  → Accept events with event_time >= 10:20 ✅
  → Drop events with event_time < 10:20 ❌

Batch 2 max event_time seen: 10:45 → watermark = 10:45 - 10min = 10:35
  → Now drop events < 10:35

Late event arrives: event_time = 10:18
  → 10:18 < 10:35 watermark → DROPPED ❌ (too late!)
```

> ⚠️ **Watermark trade-off:** Larger watermark = more late data accepted = more state kept in memory = higher latency before results are emitted. Smaller watermark = faster results, less memory, but more data dropped. Tune based on your SLA and source reliability.[^5_7]

***

## 💻 Hands-On Code — All Tasks

### Task 1: Auto Loader Pipeline — Incremental CSV Ingestion

```python
# Setup: create sample landing files in DBFS
import json
from pyspark.sql import functions as F

LANDING_PATH   = "/tmp/day4_landing/orders/"
CHECKPOINT_PATH = "/tmp/day4_checkpoints/orders_autoloader/"
SCHEMA_PATH    = "/tmp/day4_checkpoints/orders_schema/"
OUTPUT_PATH    = "/tmp/day4_delta/orders/"

# Create initial batch of files to ingest
batch1 = spark.createDataFrame([
    (1, "Alice",   "Laptop",  1200.0, "2026-04-14T10:00:00"),
    (2, "Bob",     "Phone",    800.0, "2026-04-14T10:05:00"),
    (3, "Charlie", "Desk",     350.0, "2026-04-14T10:10:00"),
], ["order_id", "customer", "product", "revenue", "event_time"])

# Write as CSV files to landing zone (simulating upstream drop)
batch1.write.mode("overwrite").option("header", True).csv(f"{LANDING_PATH}/batch1/")

# Auto Loader stream — reads incrementally from LANDING_PATH
df_stream = (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.schemaLocation", SCHEMA_PATH)
    .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
    .option("cloudFiles.inferColumnTypes", "true")
    .option("header", "true")
    .option("maxFilesPerTrigger", "10")   # process max 10 files per micro-batch
    .load(LANDING_PATH)
)

# Add metadata columns — crucial for lineage in production
df_enriched = df_stream.withColumn("_ingest_time", F.current_timestamp()) \
                       .withColumn("_source_file", F.input_file_name())

# Write to Delta with checkpoint
query = (df_enriched.writeStream
    .format("delta")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .option("mergeSchema", "true")       # handle schema evolution on write side too
    .outputMode("append")
    .trigger(availableNow=True)          # process all pending, then stop ✅
    .start(OUTPUT_PATH)
)

query.awaitTermination()
print(f"Streaming complete. Batches run: {query.lastProgress['batchId'] + 1}")
```


### Task 2: Add a New File → Prove Incremental Behavior

```python
# Simulate new files arriving AFTER first run
batch2 = spark.createDataFrame([
    (4, "Diana",  "Monitor", 450.0, "2026-04-14T11:00:00"),
    (5, "Eve",    "Tablet",  600.0, "2026-04-14T11:05:00"),
], ["order_id", "customer", "product", "revenue", "event_time"])

batch2.write.mode("overwrite").option("header", True).csv(f"{LANDING_PATH}/batch2/")

# Re-run same stream with SAME checkpoint — only batch2 files processed!
query2 = (df_enriched.writeStream   # same stream definition
    .format("delta")
    .option("checkpointLocation", CHECKPOINT_PATH)  # same checkpoint = knows batch1 done
    .outputMode("append")
    .trigger(availableNow=True)
    .start(OUTPUT_PATH)
)

query2.awaitTermination()

# Verify — should have 5 rows total (3 from batch1 + 2 from batch2)
spark.read.format("delta").load(OUTPUT_PATH).show()
```


### Task 3: Continuous Stream with ProcessingTime Trigger

```python
# For continuous ingestion (not availableNow), use processingTime
# This keeps running until manually stopped

query_continuous = (df_enriched.writeStream
    .format("delta")
    .option("checkpointLocation", "/tmp/day4_checkpoints/continuous/")
    .outputMode("append")
    .trigger(processingTime="30 seconds")  # check for new files every 30s
    .start(OUTPUT_PATH)
)

# In a Databricks Job, this runs forever until cluster terminates
# Monitor via:
print(query_continuous.status)
print(query_continuous.lastProgress)

# Stop manually (or use Job timeout)
query_continuous.stop()
```


### Task 4: Watermarking — Late Data Simulation

```python
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

# Create streaming source from a rate stream (built-in, no files needed)
# Simulate event stream with event_time column
events_data = [
    ("India", 100.0, "2026-04-14T10:00:00"),
    ("USA",   200.0, "2026-04-14T10:02:00"),
    ("India", 150.0, "2026-04-14T10:08:00"),
    ("UK",     80.0, "2026-04-14T10:12:00"),
    # Late events (arrive late but event_time is old)
    ("India",  50.0, "2026-04-14T09:55:00"),  # 5min late — within watermark ✅
    ("USA",    90.0, "2026-04-14T09:40:00"),  # 20min late — DROPPED by watermark ❌
]

events_schema = StructType([
    StructField("country",    StringType()),
    StructField("revenue",    DoubleType()),
    StructField("event_time", StringType()),
])

df_events = spark.createDataFrame(events_data, events_schema) \
                 .withColumn("event_time", F.to_timestamp("event_time"))

# Write to Delta as a streaming source
EVENTS_PATH = "/tmp/day4_events_stream/"
df_events.write.format("delta").mode("overwrite").save(EVENTS_PATH)

# Read as stream
df_event_stream = (spark.readStream
    .format("delta")
    .load(EVENTS_PATH)
)

# Apply watermark + window aggregation
df_windowed = (df_event_stream
    .withWatermark("event_time", "10 minutes")            # accept up to 10min late
    .groupBy(
        F.window("event_time", "5 minutes"),              # 5-min tumbling window
        F.col("country")
    )
    .agg(
        F.count("*").alias("event_count"),
        F.round(F.sum("revenue"), 2).alias("total_revenue")
    )
    .select(
        F.col("window.start").alias("window_start"),
        F.col("window.end").alias("window_end"),
        "country", "event_count", "total_revenue"
    )
)

# Write to console to observe (use Delta in prod)
watermark_query = (df_windowed.writeStream
    .outputMode("update")     # update mode with watermark + aggregation ✅
    .option("checkpointLocation", "/tmp/day4_checkpoints/watermark/")
    .trigger(availableNow=True)
    .format("console")
    .start()
)

watermark_query.awaitTermination()
```


### Task 5: COPY INTO — Simple Batch Alternative

```sql
-- %sql — COPY INTO: simpler, SQL-native, idempotent

-- First create target Delta table
CREATE TABLE IF NOT EXISTS copy_into_orders (
    order_id  INT,
    customer  STRING,
    product   STRING,
    revenue   DOUBLE,
    event_time STRING
) USING DELTA;

-- COPY INTO: only loads NEW files, skips already loaded ones
COPY INTO copy_into_orders
FROM '/tmp/day4_landing/orders/'
FILEFORMAT = CSV
FORMAT_OPTIONS (
    'header'      = 'true',
    'inferSchema' = 'true'
)
COPY_OPTIONS (
    'mergeSchema' = 'true'
);

-- Run again — no duplicate rows! Idempotent ✅
COPY INTO copy_into_orders
FROM '/tmp/day4_landing/orders/';

SELECT COUNT(*) FROM copy_into_orders;  -- same count both times
```


***

## 🎯 Interview Questions — Day 4 (Complete Set)

**Q1.** What is Auto Loader and how does it track which files have already been processed?
> ✅ **Answer:** Auto Loader (`cloudFiles`) is Databricks' incremental file ingestion engine. It tracks processed files via a **checkpoint directory** (streaming offsets) + `_schemas` directory (inferred schema). Two detection modes: **directory listing** (polls for new files, simpler) and **file notification** (S3/SNS/SQS or ADLS/Event Grid, scales to millions of files). On restart with same checkpoint, only new files are processed.[^5_2][^5_1]

**Q2.** What is the difference between `trigger(once=True)` and `trigger(availableNow=True)`?
> ✅ **Answer:** Both process all pending data and stop. But `once` processes everything in **one giant batch** — OOM risk on large backlogs, no per-batch fault tolerance. `availableNow` processes in **multiple right-sized batches**, respects `maxFilesPerTrigger`, checkpoints after each batch (fault tolerant). `once` is deprecated in Spark 3.3+ — always use `availableNow`.[^5_9][^5_6]

**Q3.** What are the four schema evolution modes in Auto Loader?
> ✅ **Answer:** `addNewColumns` — stream intentionally fails, restarts with new schema (most common). `rescue` — unknown columns go to `_rescued_data` JSON column (no failure). `none` — new columns silently dropped. `failOnNewColumns` — hard fail, requires human intervention to update schema.[^5_1]

**Q4.** How does Structured Streaming achieve exactly-once semantics?
> ✅ **Answer:** Via checkpointing (stores what was read in `offsets/`, what was committed in `commits/`) + idempotent sinks (Delta Lake's transactional writes). If a batch fails mid-write, on restart Spark replays from the last saved offset. Delta's idempotency ensures no duplicate rows even on replay.[^5_10]

**Q5.** What is a watermark? What's the trade-off in setting it?
> ✅ **Answer:** Watermark = max tolerance for late-arriving data. `withWatermark("event_time", "10 minutes")` tells Spark: accept events up to 10 min behind the latest event seen, drop anything older. Trade-off: **larger watermark** = more data accepted, more state in memory, higher output latency. **Smaller watermark** = faster results, less memory, more data dropped as "too late."[^5_11][^5_7]

**Q6.** When would you use COPY INTO over Auto Loader?
> ✅ **Answer:** COPY INTO when: fewer than thousands of files total, simple SQL-based workflow preferred, you need easy reprocessing of specific files (`FILES=` clause), or no schema evolution needed. Auto Loader when: millions of files, schema evolves frequently, need continuous/near-real-time ingestion, or running at enterprise scale.[^5_5]

**Q7.** What is the difference between `schemaLocation` and `checkpointLocation` in Auto Loader?
> ✅ **Answer:** `schemaLocation` = where Auto Loader stores the **inferred/evolved schema** (`_schemas/` directory) — persisted across restarts. `checkpointLocation` = where Structured Streaming stores **streaming progress** (offsets, commits, source state). Both are required, both must be separate paths, both must persist across job restarts.

**Q8.** What output modes does Structured Streaming support and when do you use each?
> ✅ **Answer:** `append` — only new rows to sink, most common for raw ingestion, **cannot be used for aggregations without watermarks**. `complete` — entire result table rewritten every batch, only for aggregations, expensive. `update` — only changed/new rows, most efficient for aggregation with Delta sink. Delta sink supports all three modes.

**Q9.** How do you handle schema changes in a running Auto Loader stream?
> ✅ **Answer:** Set `cloudFiles.schemaEvolutionMode = addNewColumns`. When new column arrives: stream **intentionally fails** with `UnknownFieldException`. Configure your Databricks Job to auto-restart on failure. On restart, Auto Loader reads the updated schema from `_schemas` dir and continues. Downstream Delta table needs `mergeSchema = true` on the write side too.[^5_4]

**Q10.** What happens if you delete the checkpoint directory of a running stream?
> ✅ **Answer:** Catastrophic — the stream loses all record of what was processed. On restart it reprocesses ALL files from the beginning → **duplicate data in sink**. If sink is Delta, you'll need deduplication (`dropDuplicates` or MERGE). Never delete checkpoints unless intentionally doing full historical reprocess. Always back up checkpoint dirs in prod.

***

## 🗺️ Day 4 Mental Model

```
Cloud Storage (S3/ADLS)
       │ new files land here
       ▼
  Auto Loader (cloudFiles)          COPY INTO (SQL)
  ─────────────────────────         ──────────────
  Streaming, millions of files      Batch, thousands of files
  Schema evolution (4 modes)        Manual schema management
  checkpoint + _schemas             Internal metadata table
       │
       ▼
  readStream → transforms → writeStream
       │                         │
       │                    checkpointLocation
       │                    (offsets/ + commits/)
       │                         │
       ▼                         ▼
  Delta Table Sink ← exactly-once via checkpoint + Delta idempotency
       │
  Trigger controls WHEN:
  ├── processingTime  → continuous, every N seconds
  ├── availableNow   → batch-style, all pending → stop ✅
  └── (default)      → as fast as possible

  Watermark controls WHAT (for stateful ops):
  └── withWatermark("event_time", "10 min") → drop data > 10min late
                                            → bound memory state size
```


***

> 

---

# 🗓️ DAY 5 — Medallion Architecture \& ETL Design 🏅

**TL;DR** — Medallion = progressively refine data quality across 3 layers. Bronze = raw truth, Silver = clean \& enriched, Gold = business-ready aggregates. SCD2 + window functions are the most interview-tested SQL skills in this entire study plan. 🎯

***

## 🥉🥈🥇 1. Medallion Architecture — The Full Picture

Medallion is a **data design pattern** that organizes data into layers of progressively increasing quality, not just a folder structure. Each layer has a clear contract — what goes in, what comes out, who owns it.[^6_1][^6_2]

```
SOURCE SYSTEMS
(APIs, DBs, Kafka, Files)
        │
        ▼
🥉 BRONZE  ──── Raw, as-is, append-only, Delta
        │        "What did we receive?" 
        ▼
🥈 SILVER  ──── Cleansed, typed, deduplicated, conformed
        │        "What is true?"
        ▼
🥇 GOLD    ──── Business aggregates, star schema, KPIs
                 "What does it mean?"
```

Each layer answers a **different question** and serves a different consumer:[^6_3]


| Layer | Quality | Consumers | Retention |
| :-- | :-- | :-- | :-- |
| 🥉 **Bronze** | Raw (no transform) | Data Engineers (debugging, replay) | Long — 1–7 years |
| 🥈 **Silver** | Validated, typed | Data Engineers, Analysts, DS | Medium — 1–3 years |
| 🥇 **Gold** | Business-aggregated | BI, Dashboards, Stakeholders | Short — 1 year (recomputable) |

> 🔑 **Bronze is sacred** — never transform or delete original Bronze data. It's your source of truth for replays, audits, and debugging upstream issues. If Silver/Gold has a bug, you re-derive from Bronze — not from the source system.[^6_4]

> ⚠️ **Overkill flag for juniors:** For tiny projects (1 source, 1 team, simple transforms), strict 3-layer medallion adds overhead. It shines when you have **multiple sources, multiple consumers, and evolving schemas**. Don't over-engineer for a single table pipeline.[^6_3]

***

## 🥉 2. Bronze Layer — Land Everything, Transform Nothing

Bronze = **raw data vault**. Write exactly what came from the source, plus metadata.[^6_2]

**Bronze rules:**

- ✅ Always **append-only** (never update/delete Bronze)
- ✅ Add ingestion metadata: `_ingest_time`, `_source_file`, `_source_system`
- ✅ Keep original column names, original data types (even if messy strings)
- ✅ Store as Delta (gives you ACID + time travel for replay)
- ❌ Never cast types, filter rows, or apply business rules
- ❌ Never deduplicate (duplicates from source = valid Bronze data)

```
Schema drift at Bronze → cascade failure at Silver/Gold 🚨
Fix: use Auto Loader with "rescue" mode on Bronze to capture unexpected columns
```


***

## 🥈 3. Silver Layer — Clean, Conform, Enrich

Silver = **single source of truth** for each business entity. This is where most ETL logic lives.[^6_1]

**Silver operations:**

- Cast types: `string → int`, `string → timestamp`, `"Y"/"N" → boolean`
- Deduplicate: `ROW_NUMBER()` to keep latest record per business key
- Null handling: fill defaults, flag nulls, drop critical-null rows
- Standardize: `UPPER(country)`, trim whitespace, normalize codes
- Apply SCD2 for slowly changing dimensions (see section 5)
- Join reference/lookup data to enrich

```
Silver is NOT a warehouse. No aggregations. No metrics.
Silver = clean facts and dimensions at their most granular level.
```


***

## 🥇 4. Gold Layer — Business Ready

Gold = **aggregated, business-aligned data products** ready for dashboards, ML, and stakeholder reporting.[^6_2]

**Gold patterns:**

- Star schema: fact tables + dimension tables
- Pre-aggregated KPIs: daily sales by region, weekly active users
- Feature store tables for ML models
- Denormalized tables for BI tools (Tableau, Power BI)

```
Gold tables are RECOMPUTABLE from Silver.
If Gold has a bug → truncate + rerun. Never patch Gold manually.
```


***

## 🔄 5. SCD Type 2 — Slowly Changing Dimensions

SCD2 = **track historical changes** to dimension data. Instead of overwriting old values, you expire the old row and insert a new one. Interviewers LOVE this.[^6_5][^6_6]

**The pattern:**

```
Customer changes city: India → USA

BEFORE (only SCD1 / overwrite):
id=1 | Alice | USA | ← India is LOST FOREVER

AFTER (SCD2 / keep history):
id=1 | Alice | India | is_active=FALSE | end_date=2026-04-20
id=1 | Alice | USA   | is_active=TRUE  | end_date=NULL      ← current
```

**SCD2 columns you MUST add to dimension tables:**

- `is_current` (BOOLEAN) — is this the latest version?
- `start_date` (TIMESTAMP) — when this record became active
- `end_date` (TIMESTAMP, nullable) — when it was expired (NULL = still active)
- `surrogate_key` (INT/BIGINT) — system-generated unique key per row version


### SCD2 MERGE Pattern — Two-Step Process

SCD2 with MERGE requires **two operations**:

1. **Expire** old active rows that have changed
2. **Insert** new rows as the current version
```python
from delta.tables import DeltaTable
from pyspark.sql import functions as F

# ──────────────────────────────────────────────────
# Step 1: Build the incoming change data
# ──────────────────────────────────────────────────
updates_data = [
    (1, "Alice",   "USA",   "alice@email.com"),   # changed country
    (3, "Charlie", "India", "charlie@new.com"),   # changed email
    (6, "Frank",   "UK",    "frank@email.com"),   # brand new customer
]
updates_schema = ["customer_id", "name", "country", "email"]
df_updates = spark.createDataFrame(updates_data, updates_schema) \
                  .withColumn("start_date", F.current_timestamp()) \
                  .withColumn("end_date",   F.lit(None).cast("timestamp")) \
                  .withColumn("is_current", F.lit(True))

SILVER_CUSTOMERS_PATH = "/tmp/day5_silver/customers/"
```

```sql
-- %sql STEP 1: Expire old active rows that have changed
-- Use MERGE to UPDATE is_current=FALSE + set end_date on changed rows

MERGE INTO silver_customers AS target
USING updates AS source
ON target.customer_id = source.customer_id
   AND target.is_current = TRUE  -- only match current rows

-- Row EXISTS and something CHANGED → expire it
WHEN MATCHED AND (
    target.country != source.country OR
    target.email   != source.email
) THEN UPDATE SET
    target.is_current = FALSE,
    target.end_date   = current_timestamp()

-- No match (new customer) already handled by step 2 below
;
```

```sql
-- %sql STEP 2: Insert new rows (updated versions + brand new customers)
-- Insert all rows from source that either:
-- a) Are new customers (NOT MATCHED)
-- b) Had changes (just expired in step 1, now re-insert as current)

INSERT INTO silver_customers
SELECT
    customer_id,
    name,
    country,
    email,
    current_timestamp() AS start_date,
    NULL                AS end_date,
    TRUE                AS is_current
FROM updates
WHERE NOT EXISTS (
    -- Don't re-insert unchanged rows
    SELECT 1 FROM silver_customers t
    WHERE t.customer_id = updates.customer_id
      AND t.is_current  = TRUE
      AND t.country     = updates.country
      AND t.email       = updates.email
);
```

```python
# PySpark DeltaTable API — full SCD2 in one block
delta_customers = DeltaTable.forPath(spark, SILVER_CUSTOMERS_PATH)

# STEP 1: Expire changed active rows
(delta_customers.alias("target")
    .merge(
        df_updates.alias("source"),
        "target.customer_id = source.customer_id AND target.is_current = true"
    )
    .whenMatchedUpdate(
        condition = "target.country != source.country OR target.email != source.email",
        set = {
            "is_current": "false",
            "end_date":   "current_timestamp()"
        }
    )
    .execute()
)

# STEP 2: Insert new current rows (new + updated records)
# Only insert rows that don't already exist as current + unchanged
df_to_insert = df_updates.alias("source").join(
    spark.read.format("delta").load(SILVER_CUSTOMERS_PATH)
         .filter("is_current = true")
         .alias("target"),
    on="customer_id", how="left_anti"  # rows in source NOT in current active target
).union(
    # Also re-insert rows that were just expired (they changed)
    df_updates.join(
        spark.read.format("delta").load(SILVER_CUSTOMERS_PATH)
             .filter("is_current = false AND end_date >= current_timestamp() - interval 1 second"),
        on="customer_id", how="inner"
    )
)

df_to_insert.write.format("delta").mode("append").save(SILVER_CUSTOMERS_PATH)
```

> 💡 **Simpler production SCD2 pattern:** Many teams skip the two-step MERGE and just use a **hash column** to detect changes. Add a `row_hash = MD5(concat_ws('|', col1, col2, col3))` to both source and target. MERGE condition becomes `target.row_hash != source.row_hash` — much cleaner. [^6_6]

***

## 🪟 6. Window Functions — Asked in EVERY SQL Round

Window functions perform calculations **across a set of rows related to the current row**, without collapsing the result like GROUP BY.[^6_7][^6_8]

### The Window Spec — Anatomy

```python
from pyspark.sql.window import Window
from pyspark.sql import functions as F

# Window spec = PARTITION BY + ORDER BY + optional FRAME
window_spec = (Window
    .partitionBy("department")   # like GROUP BY — defines the group
    .orderBy(F.desc("salary"))   # row ordering within the group
    # optional frame: .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)
```


### ROW_NUMBER vs RANK vs DENSE_RANK — The Classic Trio

This is the \#1 most asked SQL interview question in 2026.[^6_8]

```python
data = [
    ("Engineering", "Alice",   100000),
    ("Engineering", "Bob",      90000),
    ("Engineering", "Charlie",  90000),  # tie with Bob
    ("Engineering", "Diana",    80000),
    ("Marketing",   "Eve",      75000),
    ("Marketing",   "Frank",    70000),
]
df = spark.createDataFrame(data, ["dept", "name", "salary"])

w = Window.partitionBy("dept").orderBy(F.desc("salary"))

df.withColumn("row_number", F.row_number().over(w)) \
  .withColumn("rank",       F.rank().over(w)) \
  .withColumn("dense_rank", F.dense_rank().over(w)) \
  .show()
```

```
dept         name     salary  row_number  rank  dense_rank
──────────────────────────────────────────────────────────
Engineering  Alice    100000       1        1        1
Engineering  Bob       90000       2        2        2    ← tie
Engineering  Charlie   90000       3        2        2    ← tie (same rank)
Engineering  Diana     80000       4        4        3    ← RANK skips 3, DENSE_RANK doesn't
Marketing    Eve        75000       1        1        1
Marketing    Frank      70000       2        2        2
```

| Function | Ties get same rank? | Skips next rank? | Use case |
| :-- | :-- | :-- | :-- |
| `ROW_NUMBER()` | ❌ always unique | N/A | Deduplication, pick top-1 per group |
| `RANK()` | ✅ yes | ✅ yes (gap) | Olympic-style rankings, leaderboards |
| `DENSE_RANK()` | ✅ yes | ❌ no gap | Grouping peers, percentile buckets |

### LAG() and LEAD() — Time-Series \& Comparisons

```python
# LAG: look BACK n rows within the partition
# LEAD: look FORWARD n rows within the partition

w_time = Window.partitionBy("product").orderBy("order_date")

df_sales = spark.createDataFrame([
    ("Laptop", "2026-01-01", 1200.0),
    ("Laptop", "2026-02-01", 1350.0),
    ("Laptop", "2026-03-01", 1100.0),
    ("Laptop", "2026-04-01", 1500.0),
], ["product", "order_date", "revenue"])

df_sales.withColumn(
    "prev_month_revenue", F.lag("revenue",  1, 0).over(w_time)   # 0 = default if no prev row
).withColumn(
    "next_month_revenue", F.lead("revenue", 1, 0).over(w_time)
).withColumn(
    "mom_change_pct",     # Month-over-month % change
    F.round(
        (F.col("revenue") - F.lag("revenue", 1).over(w_time)) /
        F.lag("revenue", 1).over(w_time) * 100, 2
    )
).show()
```


### Running Total + Moving Average

```python
# Frame = how many rows to include in calculation
w_running = (Window
    .partitionBy("dept")
    .orderBy("order_date")
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)  # all rows up to current
)

w_moving_3 = (Window
    .partitionBy("product")
    .orderBy("order_date")
    .rowsBetween(-2, 0)   # current row + 2 rows before = 3-period moving avg
)

df_sales \
    .withColumn("running_total", F.sum("revenue").over(w_running)) \
    .withColumn("moving_avg_3",  F.avg("revenue").over(w_moving_3)) \
    .show()
```


### NTILE — Bucket Rows into Percentile Groups

```python
# NTILE(4) → split into quartiles (Q1, Q2, Q3, Q4)
w_ntile = Window.partitionBy("dept").orderBy(F.desc("salary"))
df.withColumn("salary_quartile", F.ntile(4).over(w_ntile)).show()
# Q1 = top 25% earners, Q4 = bottom 25%
```


***

## 🗄️ 7. Managed vs External Tables

The most practically important table architecture decision in any Databricks project.[^6_9][^6_10]


| Feature | 🔵 Managed Table | 🟢 External Table |
| :-- | :-- | :-- |
| **Data lives in** | Databricks-managed storage (Unity Catalog default location) | Your S3/ADLS path — you specify it |
| **`DROP TABLE`** | ❌ **Deletes metadata AND data files** | ✅ Deletes metadata only — data stays in S3 |
| **Who owns data?** | Databricks/Unity Catalog | You — your cloud account |
| **Cross-tool access?** | Hard — tied to Databricks | Easy — Athena, Glue, Synapse can also read |
| **VACUUM behavior** | Databricks manages | You must run VACUUM yourself |
| **Create syntax** | No `LOCATION` clause | Requires `LOCATION 's3://...'` |
| **Best for** | Internal analytics, single-platform | Shared data, multi-tool, existing S3 data |

```sql
-- MANAGED table (no LOCATION = Databricks owns the data)
CREATE TABLE gold_daily_sales (
    sale_date  DATE,
    country    STRING,
    total_rev  DOUBLE
)
USING DELTA;
-- Data stored at: dbfs:/user/hive/warehouse/gold_daily_sales/

-- EXTERNAL table (you own the data at this S3 path)
CREATE TABLE gold_daily_sales_ext (
    sale_date  DATE,
    country    STRING,
    total_rev  DOUBLE
)
USING DELTA
LOCATION 's3://my-data-bucket/gold/daily_sales/';
-- DROP TABLE only removes catalog entry — data in S3 untouched ✅
```

```python
# PySpark — create external Delta table
df_gold.write \
    .format("delta") \
    .mode("overwrite") \
    .option("path", "s3://my-data-bucket/gold/daily_sales/") \
    .saveAsTable("gold_daily_sales_ext")  # registers in metastore with LOCATION
```

> 🚨 **Real prod war story pattern:** Teams accidentally `DROP TABLE` a managed table and lose the data. In production, **always use External tables for anything critical**. This protects you from accidental drops. It also lets other tools (Athena for ad-hoc, Glue for ETL) read the same S3 data without going through Databricks.[^6_9]

***

## 💻 Hands-On Code — Full Bronze → Silver → Gold Pipeline

### Setup: Sample Dataset

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
from delta.tables import DeltaTable

# Paths
BRONZE_PATH = "/tmp/day5/bronze/orders/"
SILVER_PATH = "/tmp/day5/silver/orders/"
GOLD_PATH   = "/tmp/day5/gold/daily_sales/"

# Raw source data (simulating CSV landing)
raw_data = [
    (1, "alice",   "Laptop",  "electronics", "1200.0", "IN",  "2026-04-14 10:00:00"),
    (2, "bob",     "Phone",   "electronics", "800.0",  "US",  "2026-04-14 10:05:00"),
    (3, "charlie", "Desk",    "furniture",   "350.0",  "IN",  "2026-04-14 11:00:00"),
    (4, "alice",   "Laptop",  "electronics", "1200.0", "IN",  "2026-04-14 10:00:00"),  # duplicate
    (5, "diana",   "Chair",   "furniture",   None,     "UK",  "2026-04-14 12:00:00"),  # null revenue
    (6, "eve",     "Tablet",  "electronics", "600.0",  "US",  "2026-04-14 13:00:00"),
    (7, "frank",   "Monitor", "electronics", "UNKNOWN","IN",  "2026-04-15 09:00:00"),  # bad revenue
    (8, "grace",   "Sofa",    "furniture",   "890.0",  "UK",  "2026-04-15 10:00:00"),
]

raw_schema = StructType([
    StructField("order_id",   IntegerType()),
    StructField("customer",   StringType()),
    StructField("product",    StringType()),
    StructField("category",   StringType()),
    StructField("revenue",    StringType()),   # intentionally STRING (raw = messy)
    StructField("country",    StringType()),
    StructField("order_time", StringType()),   # intentionally STRING
])

df_raw = spark.createDataFrame(raw_data, raw_schema)
```


### 🥉 Bronze Layer — Land As-Is

```python
# Bronze: append raw data + add metadata. NO business transforms.
df_bronze = df_raw \
    .withColumn("_ingest_time",   F.current_timestamp()) \
    .withColumn("_source_file",   F.lit("orders_20260414.csv")) \
    .withColumn("_source_system", F.lit("orders_api"))

df_bronze.write \
    .format("delta") \
    .mode("append") \
    .save(BRONZE_PATH)

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS bronze_orders
    USING DELTA LOCATION '{BRONZE_PATH}'
""")

print("Bronze row count:", spark.read.format("delta").load(BRONZE_PATH).count())
# 8 rows — including duplicates, nulls, bad values (INTENTIONAL for Bronze!)
```


### 🥈 Silver Layer — Cleanse, Type, Deduplicate

```python
df_bronze_read = spark.read.format("delta").load(BRONZE_PATH)

# Step 1: Cast types, standardize values
df_typed = df_bronze_read \
    .withColumn("revenue_clean",
        F.when(F.col("revenue").rlike(r"^\d+\.?\d*$"),  # only if numeric string
               F.col("revenue").cast(DoubleType()))
         .otherwise(F.lit(None))                         # "UNKNOWN" → NULL
    ) \
    .withColumn("order_ts",     F.to_timestamp("order_time")) \
    .withColumn("customer",     F.upper(F.trim("customer"))) \
    .withColumn("country_code", F.upper(F.trim("country"))) \
    .withColumn("category",     F.lower(F.trim("category"))) \
    .drop("revenue", "order_time", "_ingest_time", "_source_file", "_source_system")

# Step 2: Flag and filter nulls
df_typed = df_typed.withColumn(
    "has_revenue",
    F.col("revenue_clean").isNotNull()
)

# Step 3: Deduplicate — keep latest row per order_id
w_dedup = Window.partitionBy("order_id").orderBy(F.desc("order_ts"))

df_silver = df_typed \
    .withColumn("row_num", F.row_number().over(w_dedup)) \
    .filter(F.col("row_num") == 1) \
    .drop("row_num") \
    .filter(F.col("has_revenue") == True)   # drop rows with bad/null revenue

df_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .save(SILVER_PATH)

spark.sql(f"CREATE TABLE IF NOT EXISTS silver_orders USING DELTA LOCATION '{SILVER_PATH}'")

print("Silver row count:", df_silver.count())
# 6 rows — deduplicated, typed, bad revenue rows dropped
df_silver.show()
```


### 🥇 Gold Layer — Business Aggregates

```python
df_silver_read = spark.read.format("delta").load(SILVER_PATH)

# Gold 1: Daily sales by country + category
df_gold_daily = df_silver_read \
    .withColumn("sale_date", F.to_date("order_ts")) \
    .groupBy("sale_date", "country_code", "category") \
    .agg(
        F.count("order_id")           .alias("order_count"),
        F.round(F.sum("revenue_clean"), 2) .alias("total_revenue"),
        F.round(F.avg("revenue_clean"), 2) .alias("avg_revenue"),
        F.round(F.max("revenue_clean"), 2) .alias("max_revenue")
    ) \
    .orderBy("sale_date", "country_code")

df_gold_daily.write \
    .format("delta") \
    .mode("overwrite") \
    .save(GOLD_PATH)

spark.sql(f"CREATE TABLE IF NOT EXISTS gold_daily_sales USING DELTA LOCATION '{GOLD_PATH}'")
df_gold_daily.show()
```


### Window Functions — All Patterns in One Block

```python
df_silver_read = spark.read.format("delta").load(SILVER_PATH)

# 1. ROW_NUMBER — deduplication / top-N per group
w_rank = Window.partitionBy("country_code").orderBy(F.desc("revenue_clean"))

df_ranked = df_silver_read \
    .withColumn("rank_in_country",       F.rank().over(w_rank)) \
    .withColumn("dense_rank_in_country", F.dense_rank().over(w_rank)) \
    .withColumn("row_num_in_country",    F.row_number().over(w_rank))

# Top-1 revenue order per country
df_ranked.filter(F.col("row_num_in_country") == 1) \
         .select("country_code", "product", "revenue_clean") \
         .show()

# 2. Running total per category over time
w_running = (Window
    .partitionBy("category")
    .orderBy("order_ts")
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)

df_silver_read \
    .withColumn("running_revenue", F.round(F.sum("revenue_clean").over(w_running), 2)) \
    .select("category", "product", "revenue_clean", "running_revenue") \
    .orderBy("category", "order_ts") \
    .show()

# 3. LAG/LEAD — compare with adjacent rows
w_time = Window.partitionBy("country_code").orderBy("order_ts")

df_silver_read \
    .withColumn("prev_order_revenue", F.lag("revenue_clean",  1).over(w_time)) \
    .withColumn("next_order_revenue", F.lead("revenue_clean", 1).over(w_time)) \
    .withColumn("revenue_vs_prev",
        F.round(F.col("revenue_clean") - F.col("prev_order_revenue"), 2)
    ) \
    .select("country_code", "product", "revenue_clean",
            "prev_order_revenue", "revenue_vs_prev") \
    .show()

# 4. NTILE — split into revenue quartiles
w_ntile = Window.partitionBy("category").orderBy(F.desc("revenue_clean"))
df_silver_read \
    .withColumn("revenue_quartile", F.ntile(4).over(w_ntile)) \
    .select("category", "product", "revenue_clean", "revenue_quartile") \
    .show()
```


### SCD2 Full Implementation

```python
# Build customer dimension table
customers_v1 = spark.createDataFrame([
    (1, "Alice",   "India", "alice@old.com",  True,  "2026-01-01", None),
    (2, "Bob",     "USA",   "bob@email.com",  True,  "2026-01-01", None),
    (3, "Charlie", "UK",    "charlie@co.com", True,  "2026-01-01", None),
], ["customer_id", "name", "country", "email", "is_current", "start_date", "end_date"])

DIM_PATH = "/tmp/day5/silver/customers/"
customers_v1.write.format("delta").mode("overwrite").save(DIM_PATH)

spark.sql(f"CREATE TABLE IF NOT EXISTS dim_customers USING DELTA LOCATION '{DIM_PATH}'")

# Incoming changes: Alice moved countries, Charlie changed email, Dave is new
incoming = spark.createDataFrame([
    (1, "Alice",   "USA",   "alice@new.com"),    # changed country + email
    (3, "Charlie", "UK",    "charlie@new.com"),  # only email changed
    (4, "Dave",    "India", "dave@email.com"),   # new customer
], ["customer_id", "name", "country", "email"])

delta_dim = DeltaTable.forPath(spark, DIM_PATH)

# STEP 1: Expire changed active rows
(delta_dim.alias("target")
    .merge(
        incoming.alias("source"),
        "target.customer_id = source.customer_id AND target.is_current = true"
    )
    .whenMatchedUpdate(
        condition = "target.country != source.country OR target.email != source.email",
        set = {
            "is_current": F.lit(False),
            "end_date":   F.current_timestamp()
        }
    )
    .execute()
)

# STEP 2: Insert new current rows for changed + brand new customers
df_to_insert = incoming \
    .withColumn("is_current", F.lit(True)) \
    .withColumn("start_date", F.current_timestamp()) \
    .withColumn("end_date",   F.lit(None).cast("timestamp"))

# Only insert if: new customer OR row was just expired (changed record)
current_unchanged = (spark.read.format("delta").load(DIM_PATH)
    .filter("is_current = true")
    .select("customer_id", "country", "email"))

df_final_insert = df_to_insert.join(
    current_unchanged,
    on=["customer_id", "country", "email"],
    how="left_anti"   # rows NOT in unchanged current = must insert
)

df_final_insert.write.format("delta").mode("append").save(DIM_PATH)

# Verify SCD2 result
spark.read.format("delta").load(DIM_PATH) \
    .orderBy("customer_id", "is_current") \
    .show(truncate=False)
# Alice: 2 rows (India=expired + USA=current)
# Charlie: 2 rows (old email=expired + new email=current)
# Bob: 1 row (unchanged, still current)
# Dave: 1 row (new, current)
```


***

## 🎯 Interview Questions — Day 5 (Complete Set)

**Q1.** What are the three layers of Medallion architecture and what happens in each?
> ✅ **Answer:** Bronze = raw data landed as-is from source, append-only, Delta format, no transforms — preserves source truth for replay. Silver = cleansed, typed, deduplicated, business-rule-validated facts and dimensions — single source of truth. Gold = pre-aggregated business metrics, star schema, KPIs for BI/dashboards. Each layer serves different consumers and has different retention requirements.[^6_1][^6_2]

**Q2.** How do you implement SCD Type 2 in Delta Lake?
> ✅ **Answer:** Two-step MERGE: Step 1 — MERGE to expire changed active rows: set `is_current=FALSE`, `end_date=current_timestamp()` WHERE record exists AND data changed. Step 2 — INSERT new rows for changed + new records with `is_current=TRUE`, `start_date=now`, `end_date=NULL`. Often simplified using a hash column (`MD5(concat_ws(...))`) on tracked fields to detect changes.[^6_5][^6_6]

**Q3.** What is the difference between ROW_NUMBER, RANK, and DENSE_RANK?
> ✅ **Answer:** `ROW_NUMBER` — always unique, no ties, arbitrary tiebreaker. `RANK` — tied rows get same rank, SKIPS next rank (1,2,2,4). `DENSE_RANK` — tied rows get same rank, NO skip (1,2,2,3). Use ROW_NUMBER for deduplication (need unique per group). RANK for Olympic-style leaderboards. DENSE_RANK for contiguous grouping/percentile buckets.[^6_8]

**Q4.** When would you use an External table vs a Managed table?
> ✅ **Answer:** External when: data must survive `DROP TABLE` (prod safety net), multiple tools access the same S3 data (Athena, Glue, Synapse), data existed before Databricks, or you need to manage your own data lifecycle. Managed when: pure Databricks workloads, you want Databricks to handle storage cleanup, temporary/internal tables. In production, prefer External for anything business-critical.[^6_9][^6_10]

**Q5.** What are LAG and LEAD window functions used for?
> ✅ **Answer:** `LAG(col, n)` — accesses a value n rows **before** the current row within the partition. `LEAD(col, n)` — accesses a value n rows **after**. Classic uses: month-over-month revenue change (`current - LAG(revenue, 1)`), detect consecutive events, calculate time between events, compare to previous/next record in a time series.

**Q6.** Why should Bronze layer be append-only? What's the risk of deleting from Bronze?
> ✅ **Answer:** Bronze is your **only replay mechanism**. If Silver/Gold has a bug, you re-derive from Bronze. If you delete/modify Bronze, you've lost the raw source truth — you can't reconstruct "what the source actually sent." It also enables audit trails: you can prove exactly what data you received at what time. Deleting from Bronze = destroying evidence.[^6_4]

**Q7.** What is a running total and how do you compute it in Spark?
> ✅ **Answer:** Running total = cumulative sum of a column up to the current row, within a partition. Use `Window.partitionBy(...).orderBy(...).rowsBetween(Window.unboundedPreceding, Window.currentRow)` then `sum(col).over(window_spec)`. Without the frame clause, Spark defaults to `rangeBetween` which can give unexpected results with ties in ordering.

**Q8.** What's the difference between RANK and NTILE?
> ✅ **Answer:** `RANK` assigns rank based on ORDER BY values — tied rows share rank. `NTILE(n)` divides rows into n equal buckets (like percentiles) — ignores ties, just splits by row count. NTILE(4) = quartiles, NTILE(100) = percentiles. Use NTILE when you need to segment users/products into equal-sized groups for cohort analysis or A/B grouping.

**Q9.** What's the difference between Gold layer and a data warehouse?
> ✅ **Answer:** Conceptually similar — both serve pre-aggregated, business-friendly data. Difference: Gold layer sits in your Lakehouse (Delta on S3/ADLS), supports raw SQL + DataFrame ops, is queryable by ML/Python tools too. A traditional data warehouse (Snowflake, Redshift) is a separate system with proprietary storage and separate cost. Databricks Gold can serve as a **logical data warehouse** without data movement.

**Q10.** How would you handle a schema change between Bronze and Silver?
> ✅ **Answer:** Bronze (with Auto Loader rescue mode) captures new/unexpected columns in `_rescued_data`. In the Silver transformation job: detect new columns by comparing Bronze schema to Silver schema. Use `mergeSchema=true` on Silver write. Add the new column to the Silver transform logic. For breaking changes (column removed/renamed), add a Silver reconciliation step. This is why Bronze is immutable — you can always re-derive Silver with updated logic.

***

## 🗺️ Day 5 Mental Model

```
RAW FILES / APIs / DBs
        │
        ▼
🥉 BRONZE ─── append-only, Delta, raw types, + metadata cols
        │      (_ingest_time, _source_file, _source_system)
        │      ← Never modify. Never delete.
        ▼
🥈 SILVER ─── cast types, dedup (ROW_NUMBER), nulls, SCD2
        │      window functions live here for enrichment
        │      ← Most ETL logic. Single source of truth.
        ▼
🥇 GOLD ──── groupBy + agg, star schema, KPIs
               ← Recomputable from Silver. BI/ML reads here.

TABLE STRATEGY:
├── Critical prod data    → External Table (S3 path, survive DROP)
├── Internal analytics   → Managed Table (Databricks manages)
└── DROP TABLE behavior  → External=safe, Managed=💀 data gone
```


***

> 

---

# 🗓️ DAY 6 — Workflows, Unity Catalog \& Governance 🔐

**TL;DR** — Workflows = Databricks' native orchestrator (think Airflow but deeply integrated with Spark). Unity Catalog = centralized governance layer for all data assets. DLT = declarative pipeline framework on top of Structured Streaming. These three together are the production-grade data platform story. 🏗️

***

## ⚙️ 1. Databricks Workflows (Lakeflow Jobs) — Deep Dive

A Databricks **Job** is a collection of **Tasks** with dependency relationships. It's your production orchestration layer — replaces Airflow for Databricks-native workloads in most cases.[^7_1]

### Task Types — What Can Run in a Workflow

```
TASK TYPES (each runs independently with its own cluster if needed):
──────────────────────────────────────────────────────────────────
🗒️  Notebook          → most common, runs a .ipynb/.py notebook
🐍  Python Script     → runs a .py file directly (better for CI/CD)
🛢️  SQL              → runs SQL queries against a SQL Warehouse
🌊  DLT Pipeline     → triggers a Delta Live Tables pipeline
🔧  dbt              → runs dbt models/tests
🔄  Spark Submit     → submits a JAR/Python file to Spark
📦  Python Wheel     → runs a packaged Python library entry point
📋  Run Job          → triggers ANOTHER job (job orchestration)
```


### Task Dependencies — DAG of Tasks

```
          ┌──────────────────┐
          │  TASK 1: Bronze  │
          │  (ingest raw)    │
          └────────┬─────────┘
                   │
          ┌────────▼─────────┐
          │  TASK 2: Silver  │
          │  (cleanse)       │
          └────────┬─────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼──────┐   ┌──────────▼──────┐
│ TASK 3: Gold │   │ TASK 4: Quality │
│ (aggregate)  │   │ (DQ checks)     │
└───────┬──────┘   └──────────┬──────┘
        └──────────┬──────────┘
                   │
          ┌────────▼─────────┐
          │ TASK 5: Notify   │
          │ (send report)    │
          └──────────────────┘
```

Tasks 3 and 4 run **in parallel** because they both only depend on Task 2 — Databricks handles this automatically. You just declare the dependency edges.[^7_2][^7_1]

### Retry Policy — How to Configure It

```json
// Job config JSON (also settable via UI)
{
  "task_key": "silver_task",
  "depends_on": [{"task_key": "bronze_task"}],
  "retry_on_timeout": true,
  "max_retries": 3,
  "min_retry_interval_millis": 60000,   // wait 1 min between retries
  "notebook_task": {
    "notebook_path": "/Repos/pipelines/silver_notebook"
  }
}
```

> 🔑 **Retry is per-task, not per-job.** Set retry on Silver but not Bronze if Bronze failure = data unavailability (no point retrying). Set retry on Silver if it fails due to transient cluster issues or schema evolution stream restarts.[^7_3]

> ⚠️ **Interview trap:** Timeout applies to **each retry attempt**, not the whole job. If timeout=5min and retries=3, the total possible runtime is 5×3 = 15 min. This surprises people.[^7_3]

### Email Alerts — Configuration

```json
{
  "email_notifications": {
    "on_start":   ["sagar@company.com"],
    "on_success": ["data-team@company.com"],
    "on_failure": ["sagar@company.com", "oncall@company.com"],
    "no_alert_for_skipped_runs": true
  }
}
```


***

## 🔀 2. Passing Parameters Between Tasks — `taskValues`

This is the **most asked Workflows interview question**. Multiple ways to pass data — know all three.[^7_4][^7_5]

### Method 1: `dbutils.jobs.taskValues` (Most Powerful)

```python
# ── TASK 1: Bronze notebook ──────────────────────────────────────────
from datetime import datetime

rows_ingested = df_bronze.count()
run_date      = datetime.now().strftime("%Y-%m-%d")

# SET values — downstream tasks can read these
dbutils.jobs.taskValues.set(key="rows_ingested", value=rows_ingested)
dbutils.jobs.taskValues.set(key="run_date",      value=run_date)
dbutils.jobs.taskValues.set(key="status",        value="success")
print(f"Bronze complete: {rows_ingested} rows ingested for {run_date}")
```

```python
# ── TASK 2: Silver notebook (depends on TASK 1) ──────────────────────

# GET values from upstream task
# Use taskKey reference — NOT task name string (name can change!)
rows_from_bronze = dbutils.jobs.taskValues.get(
    taskKey   = "bronze_task",   # task key defined in Job config
    key       = "rows_ingested",
    default   = 0,               # fallback if not set (useful for debugging)
    debugValue= 100              # value used when running notebook interactively
)

run_date = dbutils.jobs.taskValues.get(
    taskKey = "bronze_task",
    key     = "run_date",
    default = "2026-04-21"
)

print(f"Processing Silver for run_date={run_date}, bronze rows={rows_from_bronze}")

# Do silver work...
rows_silver = df_silver.count()

# Pass forward to gold task
dbutils.jobs.taskValues.set(key="silver_rows", value=rows_silver)
dbutils.jobs.taskValues.set(key="run_date",    value=run_date)  # forward along
```

```python
# ── TASK 3: Gold/Notify notebook ─────────────────────────────────────

bronze_rows = dbutils.jobs.taskValues.get(taskKey="bronze_task", key="rows_ingested")
silver_rows = dbutils.jobs.taskValues.get(taskKey="silver_task", key="silver_rows")
run_date    = dbutils.jobs.taskValues.get(taskKey="silver_task", key="run_date")

print(f"""
Pipeline Run Summary [{run_date}]:
  Bronze rows: {bronze_rows}
  Silver rows: {silver_rows}
  Drop rate:   {round((1 - silver_rows/bronze_rows)*100, 1)}%
""")
# Can send this to Slack/email/monitoring table
```

> ⚠️ **`taskValues` only works in Python cells.** For SQL/Scala tasks, use **Dynamic Value References** in the task config: `{{tasks.bronze_task.values.run_date}}` in the task parameters field.[^7_4]

### Method 2: Job Parameters (Simpler, Static)

```python
# Set at job level — same value for ALL tasks
# Access in any notebook via:
run_date    = dbutils.widgets.get("run_date")      # passed as job parameter
environment = dbutils.widgets.get("environment")   # "dev", "staging", "prod"

# Set via UI: Job → Configure → Parameters → Add {"run_date": "2026-04-21"}
# Or via API when triggering the job
```


### Method 3: Delta Tables as "Task Handoff" (Real Production Pattern)

```python
# Write a metadata/audit table that downstream tasks read
# Useful when passing large data structures (not just scalar values)
audit_data = [(run_date, "bronze", rows_ingested, "success")]
spark.createDataFrame(audit_data, ["run_date","layer","row_count","status"]) \
     .write.format("delta").mode("append").save("/audit/pipeline_runs/")

# Silver task reads from this
last_run = spark.read.format("delta").load("/audit/pipeline_runs/") \
               .filter(f"run_date = '{run_date}'").collect()[^7_0]
```


***

## 🌊 3. Delta Live Tables (DLT) — Declarative Pipelines

DLT is a **framework on top of Structured Streaming** where you declare WHAT you want (tables + quality rules), not HOW to execute it. Databricks handles the orchestration, dependency resolution, and restarts.[^7_6][^7_7]

### DLT vs Regular Notebooks/Jobs

| Feature | 🔵 Regular Notebooks + Jobs | 🟢 DLT |
| :-- | :-- | :-- |
| **Dependency management** | Manual (you order tasks) | Automatic (reads `dlt.read()` graph) |
| **Data quality** | Custom code (`filter`, `assert`) | Declarative `@dlt.expect` decorators |
| **Auto-restart on schema change** | Manual retry config | Built-in, automatic |
| **Lineage** | Manual tracking | Automatic, visualized in UI |
| **Monitoring UI** | Spark UI / Job runs | DLT Pipeline UI (data flow graph) |
| **Streaming + batch** | Separate logic | Unified — same code, DLT handles |
| **Overkill for?** | ❌ never overkill | ✅ for simple 2-table pipelines |

### DLT Syntax — Python API

```python
import dlt
from pyspark.sql import functions as F

# ─── BRONZE: Raw ingestion via Auto Loader ────────────────────────────
@dlt.table(
    name    = "bronze_orders",
    comment = "Raw orders from S3 landing zone",
    table_properties = {"quality": "bronze"}
)
def bronze_orders():
    return (spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.schemaLocation", "/dlt/_schemas/orders/")
        .option("header", "true")
        .load("s3://my-bucket/landing/orders/")
        .withColumn("_ingest_time", F.current_timestamp())
    )

# ─── SILVER: Cleanse + quality expectations ───────────────────────────
@dlt.expect("valid_order_id",    "order_id IS NOT NULL")          # WARN: keep bad rows, count them
@dlt.expect_or_drop("valid_revenue", "revenue > 0")              # DROP rows with revenue <= 0
@dlt.expect_or_fail("valid_customer", "customer IS NOT NULL")    # FAIL pipeline if customer null
@dlt.table(
    name    = "silver_orders",
    comment = "Cleansed orders with type casting"
)
def silver_orders():
    return (dlt.read_stream("bronze_orders")   # reads from bronze_orders above
        .withColumn("revenue",    F.col("revenue").cast("double"))
        .withColumn("order_ts",   F.to_timestamp("order_time"))
        .withColumn("customer",   F.upper(F.trim("customer")))
        .withColumn("country",    F.upper("country"))
        .dropDuplicates(["order_id"])
    )

# ─── GOLD: Business aggregations ──────────────────────────────────────
@dlt.table(
    name    = "gold_daily_sales",
    comment = "Daily revenue by country and category"
)
def gold_daily_sales():
    return (dlt.read("silver_orders")   # batch read (not stream) for aggregations
        .withColumn("sale_date", F.to_date("order_ts"))
        .groupBy("sale_date", "country", "category")
        .agg(
            F.count("order_id")              .alias("order_count"),
            F.round(F.sum("revenue"), 2)     .alias("total_revenue"),
            F.round(F.avg("revenue"), 2)     .alias("avg_revenue")
        )
    )
```


### DLT Expectation Violation Actions — The 3 Modes

```
@dlt.expect("name", "condition")           → WARN  — keep rows, log violation count 📊
@dlt.expect_or_drop("name", "condition")   → DROP  — silently remove bad rows ❌
@dlt.expect_or_fail("name", "condition")   → FAIL  — halt entire pipeline 🚨
```

```python
# Multiple expectations on same table
@dlt.expect_all({
    "valid_id":      "order_id IS NOT NULL",
    "positive_rev":  "revenue > 0",
    "valid_country": "LENGTH(country) = 2"
})
@dlt.expect_all_or_drop({
    "non_empty_customer": "customer != ''",
    "valid_date":         "order_ts >= '2020-01-01'"
})
@dlt.table(name="silver_orders_strict")
def silver_orders_strict():
    return dlt.read_stream("bronze_orders")
```

> 💡 **DLT quality metrics** are automatically logged — you can query them via `SELECT * FROM event_log('/pipeline_path/')` to see row counts per expectation per run. This is how you build a data quality dashboard.[^7_8]

***

## 🏛️ 4. Unity Catalog — Centralized Governance

Unity Catalog (UC) is Databricks' **governance layer** that provides a single place to manage data access, lineage, and discovery across ALL workspaces in your organization.[^7_9]

### 3-Level Namespace

```
catalog.schema.table
   │       │      │
   │       │      └── orders, customers, dim_date (tables/views/functions)
   │       └───────── bronze, silver, gold, raw (schemas = namespaces)
   └───────────────── my_company, finance_prod, ml_platform (catalogs = orgs/domains)

Example:
  finance_prod.gold.daily_revenue   ← the production gold table
  finance_dev.gold.daily_revenue    ← the dev copy (same schema, different catalog)
  ml_platform.features.customer_ltv ← feature store table

Legacy (Hive metastore): schema.table  ← only 2 levels, no catalog isolation
```


### Why Unity Catalog Over Legacy Hive Metastore

| Feature | Legacy Hive Metastore | Unity Catalog |
| :-- | :-- | :-- |
| **Scope** | Single workspace | All workspaces in account |
| **Fine-grained access** | Table-level only | Column + row level |
| **Data lineage** | ❌ Not available | ✅ Automatic, cross-workspace |
| **Audit logs** | Limited | Full — who read what, when |
| **External locations** | Manual mount points | Managed, governed |
| **Delta Sharing** | ❌ | ✅ Share data across orgs |
| **AI/ML assets** | ❌ | ✅ Models, vectors, features |

### RBAC — Role-Based Access Control

```sql
-- Grant and revoke at any level of the 3-tier hierarchy

-- CATALOG level (all schemas + tables in catalog)
GRANT USE CATALOG ON CATALOG finance_prod TO `data-analysts@company.com`;

-- SCHEMA level (all tables in schema)
GRANT USE SCHEMA, SELECT ON SCHEMA finance_prod.gold TO `bi-team@company.com`;

-- TABLE level
GRANT SELECT ON TABLE finance_prod.gold.daily_revenue TO `analyst_john@company.com`;

-- COLUMN level — restrict specific columns
-- Unity Catalog uses column masks (dynamic masking functions)
GRANT SELECT ON TABLE finance_prod.silver.customers TO `intern@company.com`;
-- But column mask applied to 'email' and 'phone' columns (below)

-- Revoke
REVOKE SELECT ON TABLE finance_prod.gold.daily_revenue FROM `analyst_john@company.com`;

-- Check what a user has access to
SHOW GRANTS ON TABLE finance_prod.gold.daily_revenue;
```


### Row-Level Security — Via Row Filters

```sql
-- Step 1: Create a row filter function
CREATE FUNCTION finance_prod.security.region_filter(region STRING)
RETURNS BOOLEAN
RETURN region = current_user() OR is_account_group_member('data-admin');

-- Step 2: Apply to table — users only see rows matching their region
ALTER TABLE finance_prod.silver.sales
SET ROW FILTER finance_prod.security.region_filter ON (region);

-- Now: analyst from India only sees India rows
-- Admin sees all rows
```


### Column-Level Security — Via Column Masks

```sql
-- Step 1: Create a masking function
CREATE FUNCTION finance_prod.security.mask_email(email STRING)
RETURNS STRING
RETURN CASE
    WHEN is_account_group_member('hr-team') THEN email  -- HR sees full email
    ELSE CONCAT(LEFT(email, 2), '****@****.com')        -- others see masked
END;

-- Step 2: Apply mask to column
ALTER TABLE finance_prod.silver.customers
ALTER COLUMN email SET MASK finance_prod.security.mask_email;

-- Normal user: al****@****.com
-- HR user:     alice@company.com
```

> 🔑 **Key point for interviews:** UC's row and column security is enforced **at the query engine level** — not in application code. Even if someone connects via JDBC, the security still applies within Databricks.[^7_10][^7_9]

***

## 🔍 5. Data Lineage — Automatic in Unity Catalog

Lineage is automatic — no configuration needed. UC tracks table-to-table dependencies when you: read a table, transform it, and write to another table.[^7_11]

```
bronze_orders ──TRANSFORM──► silver_orders ──AGGREGATE──► gold_daily_sales
                                   │
                              ──JOIN──► dim_customers ──► gold_customer_kpis

Unity Catalog shows:
  gold_daily_sales
    ├── upstream: silver_orders (via notebook X, job Y, run Z)
    ├── upstream: dim_date (via notebook X)
    └── downstream: powerbi_dashboard (via SQL query by user@co.com)
```

> 💡 **Why lineage matters in prod:** When Silver schema changes, you instantly know which Gold tables and BI dashboards will break. Without lineage, you find out when the dashboard is red at 9am Monday morning. 😬

***

## 🔑 6. Secrets — Never Hardcode Credentials

```python
# ──────────────────────────────────────────────────
# WRONG — NEVER DO THIS (will appear in git, logs, notebooks)
# ──────────────────────────────────────────────────
conn_string = "jdbc:postgresql://prod-db:5432/orders?user=admin&password=SuperSecret123"

# ──────────────────────────────────────────────────
# RIGHT — Use Databricks Secrets
# ──────────────────────────────────────────────────
# First create secret scope + secret via CLI:
# databricks secrets create-scope --scope prod-secrets
# databricks secrets put --scope prod-secrets --key db-password

# In notebook:
db_password = dbutils.secrets.get(scope="prod-secrets", key="db-password")
db_user     = dbutils.secrets.get(scope="prod-secrets", key="db-user")
db_host     = dbutils.secrets.get(scope="prod-secrets", key="db-host")

# Secrets are REDACTED in notebook output — you'll see [REDACTED]
print(db_password)   # → [REDACTED] ✅ never exposed even in output

# Use in JDBC connection
jdbc_url = f"jdbc:postgresql://{db_host}:5432/orders"
df = spark.read.format("jdbc") \
    .option("url",      jdbc_url) \
    .option("user",     db_user) \
    .option("password", db_password) \
    .option("dbtable",  "orders") \
    .load()
```

**Secret scope types:**

```
DATABRICKS-BACKED SCOPE:
  - Secrets stored in Databricks' encrypted store
  - Create via CLI or API
  - Best for: Databricks-specific secrets

AZURE KEY VAULT-BACKED SCOPE (Azure only):
  - Secrets stored in Azure Key Vault
  - Databricks reads from AKV at runtime
  - Best for: enterprise Azure shops (single secret store)

AWS SECRETS MANAGER (via environment variables):
  - Use IAM instance profile on cluster → no secret needed
  - Best for: AWS-native shops using IAM roles
```


***

## 💻 Hands-On Code — All Tasks

### Task 1: Workflow via JSON API (Production Pattern)

```python
# Create a 3-task Bronze → Silver → Gold workflow via Databricks Jobs API
# In production, this JSON lives in your Git repo and gets deployed via CI/CD

job_config = {
    "name": "medallion_pipeline_daily",
    "schedule": {
        "quartz_cron_expression": "0 0 6 * * ?",   # 6:00 AM UTC daily
        "timezone_id":             "Asia/Kolkata"
    },
    "email_notifications": {
        "on_failure": ["sagar@company.com"],
        "no_alert_for_skipped_runs": True
    },
    "tasks": [
        {
            "task_key": "bronze_ingest",
            "description": "Ingest raw CSV files to Bronze Delta",
            "notebook_task": {
                "notebook_path": "/Repos/pipelines/day5/bronze_notebook",
                "base_parameters": {
                    "run_date":     "{{job.start_time.iso_date}}",  # dynamic param
                    "environment":  "prod"
                }
            },
            "job_cluster_key": "pipeline_cluster",
            "max_retries":     2,
            "min_retry_interval_millis": 60000,
            "retry_on_timeout": True,
            "timeout_seconds":  3600
        },
        {
            "task_key": "silver_cleanse",
            "description": "Cleanse Bronze → Silver",
            "depends_on": [{"task_key": "bronze_ingest"}],  # DAG dependency
            "notebook_task": {
                "notebook_path": "/Repos/pipelines/day5/silver_notebook"
            },
            "job_cluster_key": "pipeline_cluster",
            "max_retries":     3,
            "min_retry_interval_millis": 30000
        },
        {
            "task_key": "gold_aggregate",
            "description": "Aggregate Silver → Gold",
            "depends_on": [{"task_key": "silver_cleanse"}],
            "notebook_task": {
                "notebook_path": "/Repos/pipelines/day5/gold_notebook"
            },
            "job_cluster_key": "pipeline_cluster",
            "max_retries": 1
        }
    ],
    "job_clusters": [{
        "job_cluster_key": "pipeline_cluster",
        "new_cluster": {
            "spark_version":  "15.4.x-scala2.12",
            "node_type_id":   "i3.xlarge",
            "num_workers":    2,
            "spark_conf": {
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.shuffle.partitions": "auto"
            }
        }
    }]
}

# Deploy via Databricks REST API (from CI/CD pipeline or local)
import requests

DATABRICKS_HOST  = dbutils.secrets.get("dev-secrets", "databricks-host")
DATABRICKS_TOKEN = dbutils.secrets.get("dev-secrets", "databricks-token")

response = requests.post(
    f"{DATABRICKS_HOST}/api/2.1/jobs/create",
    headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
    json=job_config
)
print(f"Job created: {response.json()}")
```


### Task 2: taskValues — Bronze → Silver → Gold Handoff

```python
# ── bronze_notebook.py ────────────────────────────────────────────────
from datetime import datetime
from pyspark.sql import functions as F

run_date = dbutils.widgets.get("run_date") if "run_date" in [w.name for w in dbutils.widgets.getAll()] \
           else datetime.now().strftime("%Y-%m-%d")

# Do bronze work
df_bronze = spark.createDataFrame([
    (1, "Alice",  "Laptop", 1200.0),
    (2, "Bob",    "Phone",   800.0),
    (3, "Alice",  "Laptop", 1200.0),  # duplicate
], ["order_id", "customer", "product", "revenue"])

BRONZE_PATH = f"/tmp/day6/bronze/{run_date}/"
df_bronze.write.format("delta").mode("overwrite").save(BRONZE_PATH)

rows = df_bronze.count()
dbutils.jobs.taskValues.set(key="rows_ingested", value=rows)
dbutils.jobs.taskValues.set(key="run_date",      value=run_date)
dbutils.jobs.taskValues.set(key="bronze_path",   value=BRONZE_PATH)
print(f"Bronze: {rows} rows → {BRONZE_PATH}")
```

```python
# ── silver_notebook.py ────────────────────────────────────────────────
from pyspark.sql.window import Window
from pyspark.sql import functions as F

# Get from upstream bronze task
bronze_rows  = dbutils.jobs.taskValues.get(taskKey="bronze_ingest", key="rows_ingested",  debugValue=3)
run_date     = dbutils.jobs.taskValues.get(taskKey="bronze_ingest", key="run_date",       debugValue="2026-04-21")
bronze_path  = dbutils.jobs.taskValues.get(taskKey="bronze_ingest", key="bronze_path",    debugValue="/tmp/day6/bronze/2026-04-21/")

df_bronze = spark.read.format("delta").load(bronze_path)

# Deduplicate
w = Window.partitionBy("order_id").orderBy(F.desc(F.lit(1)))
df_silver = df_bronze.withColumn("rn", F.row_number().over(w)) \
                     .filter("rn = 1").drop("rn")

SILVER_PATH = f"/tmp/day6/silver/{run_date}/"
df_silver.write.format("delta").mode("overwrite").save(SILVER_PATH)

silver_rows = df_silver.count()
dbutils.jobs.taskValues.set(key="silver_rows", value=silver_rows)
dbutils.jobs.taskValues.set(key="silver_path", value=SILVER_PATH)
dbutils.jobs.taskValues.set(key="run_date",    value=run_date)

print(f"Silver: {silver_rows} rows (dropped {bronze_rows - silver_rows} dupes)")
```

```python
# ── gold_notify_notebook.py ───────────────────────────────────────────
bronze_rows = dbutils.jobs.taskValues.get(taskKey="bronze_ingest", key="rows_ingested", debugValue=3)
silver_rows = dbutils.jobs.taskValues.get(taskKey="silver_cleanse", key="silver_rows",  debugValue=2)
run_date    = dbutils.jobs.taskValues.get(taskKey="silver_cleanse", key="run_date",     debugValue="2026-04-21")
silver_path = dbutils.jobs.taskValues.get(taskKey="silver_cleanse", key="silver_path",  debugValue="/tmp/day6/silver/2026-04-21/")

df_silver = spark.read.format("delta").load(silver_path)

df_gold = df_silver.groupBy("product") \
                   .agg(F.count("order_id").alias("orders"),
                        F.sum("revenue").alias("total_rev"))

GOLD_PATH = f"/tmp/day6/gold/{run_date}/"
df_gold.write.format("delta").mode("overwrite").save(GOLD_PATH)

# Write audit record for pipeline observability
audit_row = [(run_date, bronze_rows, silver_rows, df_gold.count(), "success")]
spark.createDataFrame(audit_row, ["run_date","bronze_rows","silver_rows","gold_rows","status"]) \
     .write.format("delta").mode("append").save("/tmp/day6/audit/pipeline_runs/")

print(f"""
✅ Pipeline Complete [{run_date}]
   Bronze rows:  {bronze_rows}
   Silver rows:  {silver_rows}
   Gold rows:    {df_gold.count()}
   Drop rate:    {round((1 - silver_rows/bronze_rows)*100, 1)}%
""")
```


### Task 3: Unity Catalog Table + RBAC

```sql
-- %sql

-- Create catalog (done by admin, once)
CREATE CATALOG IF NOT EXISTS finance_prod;

-- Create schemas per medallion layer
CREATE SCHEMA IF NOT EXISTS finance_prod.bronze;
CREATE SCHEMA IF NOT EXISTS finance_prod.silver;
CREATE SCHEMA IF NOT EXISTS finance_prod.gold;

-- Create external Delta table in UC
CREATE TABLE IF NOT EXISTS finance_prod.gold.daily_sales (
    sale_date   DATE,
    country     STRING,
    category    STRING,
    order_count BIGINT,
    total_rev   DOUBLE
)
USING DELTA
LOCATION 's3://my-bucket/finance/gold/daily_sales/';

-- Grant access (RBAC)
GRANT USE CATALOG ON CATALOG finance_prod TO `bi-analysts`;
GRANT USE SCHEMA  ON SCHEMA finance_prod.gold TO `bi-analysts`;
GRANT SELECT      ON TABLE finance_prod.gold.daily_sales TO `bi-analysts`;

-- Grant write to DE team
GRANT ALL PRIVILEGES ON TABLE finance_prod.gold.daily_sales TO `data-engineers`;

-- Check effective grants
SHOW GRANTS ON TABLE finance_prod.gold.daily_sales;
```


### Task 4: Secrets Usage Pattern

```python
# Pattern: load all secrets at top of notebook, use throughout
# Never pass secrets as function arguments (they'd appear in logs)

class PipelineConfig:
    def __init__(self):
        self.db_host     = dbutils.secrets.get("prod-secrets", "postgres-host")
        self.db_user     = dbutils.secrets.get("prod-secrets", "postgres-user")
        self.db_password = dbutils.secrets.get("prod-secrets", "postgres-password")
        self.api_key     = dbutils.secrets.get("prod-secrets", "external-api-key")
        self.s3_bucket   = dbutils.secrets.get("prod-secrets", "output-s3-bucket")

cfg = PipelineConfig()

# Use config throughout — never expose raw secrets
def read_postgres(table_name: str):
    return (spark.read.format("jdbc")
        .option("url",      f"jdbc:postgresql://{cfg.db_host}:5432/proddb")
        .option("dbtable",  table_name)
        .option("user",     cfg.db_user)
        .option("password", cfg.db_password)
        .option("driver",   "org.postgresql.Driver")
        .load()
    )

df_orders = read_postgres("orders")
```


***

## 🎯 Interview Questions — Day 6 (Complete Set)

**Q1.** How do you pass parameters between tasks in a Databricks Workflow?
> ✅ **Answer:** Three ways: 1) `dbutils.jobs.taskValues.set/get()` — key-value store per run, task-scoped, most powerful, Python only (use dynamic references for SQL/Scala). 2) Job-level parameters via `dbutils.widgets.get()` — same value for all tasks, set at job config time. 3) Writing to a shared Delta table — best for complex objects or audit trails.[^7_5][^7_4]

**Q2.** What is Unity Catalog and why is it better than the legacy Hive metastore?
> ✅ **Answer:** Unity Catalog is a centralized governance layer with 3-level namespace (`catalog.schema.table`) that spans ALL workspaces in an organization. Hive metastore: workspace-scoped, table-level ACLs only, no lineage, no audit. UC adds: cross-workspace governance, column/row-level security, automatic data lineage, full audit logs, Delta Sharing, and ML/AI asset management.[^7_11]

**Q3.** What are DLT expectations and what are the three violation actions?
> ✅ **Answer:** Expectations are declarative data quality rules attached to DLT tables via decorators. Three violation actions: `@dlt.expect` = WARN (keep bad rows, log metrics), `@dlt.expect_or_drop` = DROP (silently remove violating rows), `@dlt.expect_or_fail` = FAIL (halt entire pipeline, requires human intervention). Choose based on whether bad data is tolerable or catastrophic.[^7_7][^7_6]

**Q4.** What is the difference between DLT and regular Databricks Jobs?
> ✅ **Answer:** DLT is **declarative** — you define tables and quality rules, DLT resolves dependencies and handles restarts automatically. Regular Jobs are **imperative** — you explicitly order tasks and handle errors. DLT has built-in quality metrics UI, automatic lineage, and handles streaming + batch in one framework. Jobs are better for heterogeneous tasks (SQL Warehouse + DLT + Python scripts) and when you need fine-grained retry control per task.

**Q5.** How does row-level security work in Unity Catalog?
> ✅ **Answer:** Via **Row Filter functions** — you create a SQL function that returns a boolean based on `current_user()` or group membership. You attach this function to a table via `ALTER TABLE ... SET ROW FILTER`. At query time, UC evaluates the function for each row — users only see rows where the function returns TRUE. Enforced at engine level, not application level.[^7_9]

**Q6.** What happens to data when you `DROP TABLE` on a managed vs external table in Unity Catalog?
> ✅ **Answer:** Managed table: `DROP TABLE` deletes **both** the catalog metadata AND the physical data files. External table: `DROP TABLE` removes only the catalog metadata — physical data in S3/ADLS is untouched. In production, always use external tables for business-critical data as a safety net against accidental drops.

**Q7.** Why should you never hardcode secrets in notebooks? What's the risk?
> ✅ **Answer:** Notebooks are stored in version control (Repos/Git), displayed in UI, logged in job run history, and shareable. Hardcoded credentials appear in all these places. Use `dbutils.secrets.get()` — secrets are stored encrypted, and are **redacted** (`[REDACTED]`) even in print/log output. For AWS, use IAM instance profiles instead of any credentials at all.

**Q8.** What is the difference between `trigger(once=True)` in a DLT pipeline vs a Streaming query?
> ✅ **Answer:** In Structured Streaming: `trigger(once=True)` processes all available data in one batch, deprecated in Spark 3.3+. In DLT: the equivalent is **Triggered mode** (vs Continuous mode) — processes new data and stops. DLT Triggered mode uses `availableNow` semantics internally (multi-batch, fault tolerant). DLT Continuous mode runs the pipeline indefinitely like a long-running stream.

**Q9.** How does automatic lineage work in Unity Catalog?
> ✅ **Answer:** UC intercepts query execution at the engine level — when you `SELECT` from Table A and `INSERT INTO` Table B, UC records the relationship automatically. No annotation needed. Lineage is queryable via the UC UI (Data → [table] → Lineage tab) or via system tables. It tracks column-level lineage too — which source columns contributed to which target columns.

**Q10.** What's the `debugValue` parameter in `dbutils.jobs.taskValues.get()` for?
> ✅ **Answer:** When you run a notebook interactively (not as a job task), there's no upstream task that set the value — `taskValues.get()` would fail. `debugValue` provides a **static fallback** value for interactive development/testing only. It's ignored when running inside an actual job. Essential for developing and testing notebook code locally before deploying to a job.[^7_5]

***

## 🗺️ Day 6 Mental Model

```
DATABRICKS PLATFORM GOVERNANCE
────────────────────────────────────────────────────────────────────────
UNITY CATALOG  (catalog.schema.table)
  ├── RBAC: GRANT/REVOKE at catalog/schema/table/column/row level
  ├── Auto Lineage: tracks all reads/writes automatically
  └── Audit Logs: who read what, when (compliance ✅)

WORKFLOWS (Lakeflow Jobs)
  ├── Tasks: Notebook | Python | SQL | DLT | dbt | Run Job
  ├── DAG dependencies: parallel + sequential
  ├── Retry policies: per-task, timeout per retry
  ├── Task Values: dbutils.jobs.taskValues.set/get → pass data between tasks
  └── Alerts: on_start / on_success / on_failure → email/webhook

DELTA LIVE TABLES (DLT)
  ├── @dlt.table → declare table
  ├── dlt.read / dlt.read_stream → auto-resolves dependencies
  ├── @dlt.expect* → data quality (warn/drop/fail)
  └── Continuous vs Triggered mode

SECRETS
  └── dbutils.secrets.get(scope, key)
      ├── Databricks-backed scope (encrypted store)
      ├── Azure Key Vault-backed scope (enterprise)
      └── AWS IAM roles (best practice, no secret at all)
```


***

> 