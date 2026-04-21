<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>



# 🟢 Phase 1 — PySpark Fresher Foundation

## 🧠 Core Concepts


***

**Q1. What is PySpark, and how is it different from regular Python Pandas?**

**A:** PySpark is the Python API for Apache Spark — a distributed computing engine built for processing massive datasets across a cluster of machines. Pandas runs on a single machine, in-memory, so it dies at ~10GB RAM. PySpark distributes data across nodes, handles terabytes easily, and is lazy by default.[^1_1][^1_2]


| Feature | Pandas | PySpark |
| :-- | :-- | :-- |
| Execution | Single machine | Distributed cluster |
| Data size | GBs | TBs/PBs |
| Evaluation | Eager (immediate) | Lazy (until action) |
| API | Native Python | Spark JVM via Python |

💡 **Interview Tip:** Always mention scale — *"Pandas is great for 1M rows, PySpark is built for 1 billion rows."* ZS Associates loves this practical framing. ✅

***

**Q2. Explain Spark Architecture — Driver vs Executor?**

**A:** Spark has a master-worker architecture:[^1_3]

- **Driver** — The brain 🧠. Runs your `main()`, creates the SparkContext, builds the DAG, and sends tasks to executors
- **Cluster Manager** — Allocates resources (YARN, Mesos, or Databricks Runtime)
- **Executors** — Worker nodes that actually run the tasks, store data in memory/disk, and report back to driver

💡 **Interview Tip:** Draw this out on a whiteboard if asked. Say *"Driver is like a project manager, executors are the developers who do actual work."* 🏗️

***

**Q3. What is a DAG in Spark and why does it matter?**

**A:** DAG stands for Directed Acyclic Graph — Spark's internal execution blueprint. Every transformation you write gets recorded as nodes (operations) and edges (data flow) in a DAG. When an action triggers execution, Spark's **Catalyst Optimizer** analyzes the full DAG and finds the most efficient execution path — combining steps, pushing filters early, skipping unnecessary reads.[^1_4][^1_3]

💡 **Interview Tip:** Say *"DAG enables query optimization before a single byte is processed"* — senior-sounding answer! 🔥

***

**Q4. What is Lazy Evaluation? Why does Spark use it?**

**A:** Lazy evaluation means Spark **does NOT execute transformations immediately** — it just records them in the DAG. Only when you call an **Action** (like `.show()`, `.count()`, `.write()`) does Spark actually compute.[^1_5][^1_6][^1_7]

**Why?** Because now Spark can:

- Look at the **entire plan** before executing
- Apply **predicate pushdown** (filter early, read less)
- Merge multiple transformations into fewer shuffle stages
- Eliminate redundant computations[^1_3]

💡 **Interview Tip:** Use this analogy — *"Like a chef who reads the entire recipe before cooking, not step-by-step."* 👨‍🍳

***

**Q5. Difference between Transformation and Action?**

**A:**

- **Transformation** — Returns a new DataFrame/RDD, lazy, builds the DAG. Examples: `filter()`, `select()`, `groupBy()`, `join()`[^1_5]
- **Action** — Triggers actual execution, returns a result or writes data. Examples: `show()`, `count()`, `collect()`, `write()`

💡 **Interview Tip:** Never say `collect()` in production without a caveat — it pulls ALL data to driver. Say *"I avoid collect() on large datasets because it can OOM the driver."* 🚨

***

**Q6. Narrow vs Wide Transformations — Examples?**

**A:**


| Type | Description | Examples |
| :-- | :-- | :-- |
| **Narrow** | Each input partition contributes to ONE output partition. No shuffle. Fast ⚡ | `filter()`, `map()`, `select()`, `union()` |
| **Wide** | Input partitions contribute to MULTIPLE output partitions. Causes **shuffle**. Expensive 💸 | `groupBy()`, `join()`, `distinct()`, `repartition()` |

💡 **Interview Tip:** Say *"Wide transformations cause shuffles — network I/O across executors — which is the \#1 cause of slow Spark jobs."* This shows production awareness. 🎯

***

**Q7. Difference between RDD, DataFrame, and Dataset?**

**A:**[^1_2]


|  | RDD | DataFrame | Dataset |
| :-- | :-- | :-- | :-- |
| Abstraction | Low-level | High-level | High-level |
| Type Safety | No | No | Yes (Scala/Java only) |
| Optimizer | None | Catalyst ✅ | Catalyst ✅ |
| When to use | Custom logic, legacy | Standard ETL (default) | Type-safe Java/Scala code |
| Python support | ✅ | ✅ | ❌ |

💡 **Interview Tip:** *"In production I always use DataFrames in Python — best performance, Catalyst optimizer, and easy to debug."* Don't recommend RDD unless asked specifically. ✅

***

**Q8. How does Spark handle Fault Tolerance internally?**

**A:** Spark uses **RDD Lineage** — every RDD/DataFrame knows the transformation chain (DAG) that created it. If an executor crashes mid-job, Spark doesn't restart from scratch. It **recomputes only the lost partition** by replaying the lineage chain from the last checkpoint or stable source. This is why caching + checkpointing matter for long lineage chains.[^1_2]

💡 **Interview Tip:** Mention `df.cache()` or `df.checkpoint()` for iterative algorithms — shows you know when lineage gets too long. 🔄

***

**Q9. What is Schema Inference in PySpark, and when does it go wrong?**

**A:** When you read a file without specifying a schema, Spark **samples the data** (default: first 100 rows for JSON, entire file for CSV) and guesses the data types. It goes wrong when:[^1_1]

- An integer column has `null` in first 100 rows → inferred as `string`
- Mixed types in JSON → unpredictable schema
- Large CSV inference = **full file scan = slow and expensive** 💸

💡 **Interview Tip:** *"In production I always define schemas explicitly using StructType — faster, predictable, no surprises in prod."* This is a senior answer from a fresher. 🌟

***

**Q10. File Formats — When to pick Parquet?**

**A:**


| Format | Compression | Schema | Best For |
| :-- | :-- | :-- | :-- |
| **Parquet** | ✅ Columnar | ✅ Embedded | Analytics, Data Lake default 🏆 |
| ORC | ✅ Columnar | ✅ Embedded | Hive-heavy Hadoop stacks |
| Avro | Row-based | ✅ Separate | Kafka, streaming, row-level writes |
| JSON | None | ❌ | APIs, debugging, human-readable |

Pick **Parquet** when doing analytical queries on large data — columnar storage means Spark reads only the columns you `SELECT`, not the whole row.[^1_3]

💡 **Interview Tip:** Say *"Parquet + Snappy compression is my default choice for any data lake on S3 or ADLS."* Instant credibility. 🔥

***

## 🛠️ Basic DataFrame Operations


***

**Q11. How do you read a CSV file with a defined schema?**

**A:**

```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

schema = StructType([
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("salary", IntegerType(), True)
])

df = spark.read.csv("s3://bucket/data.csv", header=True, schema=schema)
```

💡 **Interview Tip:** Always mention `header=True` and `schema=schema` — shows you don't leave it to chance. ✅

***

**Q12. Difference between `select()` and `withColumn()`?**

**A:**

- `select()` — **Chooses which columns to include**, returns only selected columns. Can also create new ones inline
- `withColumn()` — **Adds or replaces ONE column**, keeps all existing columns

```python
df.select("name", "age")                       # keeps only name, age
df.withColumn("age_plus_1", df.age + 1)        # keeps ALL cols + new col
```

💡 **Interview Tip:** For transforming many columns, `select()` is more efficient than chaining multiple `withColumn()` calls. Shows optimization thinking! ⚡

***

**Q13. How do you filter rows with multiple conditions?**

**A:**

```python
from pyspark.sql.functions import col

# Using & (AND), | (OR), ~ (NOT) — always use parentheses!
df.filter((col("age") > 25) & (col("salary") > 50000))
df.filter((col("dept") == "IT") | (col("dept") == "HR"))
df.where(col("status") != "inactive")  # .where() is alias of .filter()
```

💡 **Interview Tip:** Common mistake = using `and`/`or` instead of `&`/`|`. Mention the parentheses rule. Interviewers test this. ⚠️

***

**Q14. What does `printSchema()` tell you, and when is it useful?**

**A:** `printSchema()` prints the **column names, data types, and nullable flags** in a tree structure. Useful when:

- Debugging schema inference issues
- Verifying a join didn't duplicate columns
- Confirming nested JSON was parsed correctly

```python
df.printSchema()
# root
#  |-- name: string (nullable = true)
#  |-- age: integer (nullable = true)
```

💡 **Interview Tip:** Pair it with `df.dtypes` (returns list) for programmatic schema checks in pipelines. 🔍

***

**Q15. How do you drop duplicates?**

**A:**

```python
df.distinct()                              # removes fully duplicate rows
df.dropDuplicates(["email"])               # dedupe based on specific columns
df.dropDuplicates(["name", "dept"])        # combo key deduplication
```

💡 **Interview Tip:** `distinct()` = full row dedup. `dropDuplicates(subset)` = targeted. Mention that both trigger a shuffle (wide transformation). 🔄

***

**Q16. Difference between `drop()` and `dropna()`?**

**A:**

- `drop("col_name")` — **Removes a column** from the DataFrame
- `dropna()` — **Removes rows** where specified columns have null values

```python
df.drop("unnecessary_column")              # drops the column
df.dropna()                                # drop rows with ANY null
df.dropna(subset=["email", "phone"])       # drop rows where these cols are null
df.dropna(thresh=3)                        # keep rows with at least 3 non-null values
```

💡 **Interview Tip:** Easy to confuse under pressure. Repeat internally: *"drop = column gone, dropna = null rows gone."* 🎯

***

**Q17. How do you rename a column in PySpark?**

**A:**

```python
df.withColumnRenamed("old_name", "new_name")

# Rename multiple at once using select + alias (more efficient)
df.select(
    col("first_name").alias("fname"),
    col("last_name").alias("lname")
)
```

💡 **Interview Tip:** For renaming multiple columns, `select + alias` is cleaner than chaining `withColumnRenamed` multiple times. ⚡

***

**Q18. What is `lit()` used for?**

**A:** `lit()` (literal) creates a **constant column** with a fixed value — used when you need to add a hardcoded column to a DataFrame.[^1_1]

```python
from pyspark.sql.functions import lit

df.withColumn("source", lit("ZS_pipeline"))
df.withColumn("processed_flag", lit(True))
df.withColumn("version", lit(2))
```

💡 **Interview Tip:** Also useful in conditional logic — `when(condition, lit("yes")).otherwise(lit("no"))`. Very commonly tested! ✅

***

**Q19. How do you handle null values in PySpark?**

**A:**

```python
df.dropna(subset=["col"])                          # drop null rows
df.fillna({"age": 0, "name": "unknown"})           # fill nulls with defaults
df.fillna(0)                                       # fill all numeric nulls with 0

# Conditional null check
from pyspark.sql.functions import col, when, coalesce
df.withColumn("age_clean", coalesce(col("age"), lit(0)))
df.filter(col("email").isNull())                   # find nulls
df.filter(col("email").isNotNull())                # find non-nulls
```

💡 **Interview Tip:** Mention `coalesce()` for null fallback chaining — it's elegant and production-grade. Interviewers love it. 🌟

***

**Q20. How to convert PySpark DataFrame to Pandas? What's the risk?**

**A:**

```python
pandas_df = spark_df.toPandas()  # pulls ALL data to driver node
```

**The risks 🚨:**

- Pulls entire dataset into **driver RAM** — can crash if data is large
- Breaks distributed execution — you lose all Spark parallelism
- Network overhead moving data from executors to driver

**When it's okay:** Only on small, already-aggregated results (e.g., after a `groupBy().count()` that returns 100 rows).

💡 **Interview Tip:** Say *"I only call toPandas() after heavy aggregation — never on raw large DataFrames. In Databricks I prefer native PySpark or Pandas-on-Spark (`spark.createDataFrame(pandas_df)`)."* 🔥

***

## 🎯 ZS Associates Round 1 — Meta Tips

- 🗣️ **Always justify your answer** — not just what, but WHY (lazy eval, shuffle cost, memory risk)
- ⚡ **Mention performance implications** — shows production thinking even as a fresher
- 🏗️ **Use Databricks examples** — ZS uses cloud platforms, it's a bonus
- 🚫 **Avoid `collect()` and `toPandas()` on large data** — interviewers often trap freshers here
- 📋 **Schema always explicit in prod** — never rely on inference in real pipelines
- 🔄 **Shuffle = expensive** — mention it whenever groupBy/join/distinct comes up[^1_4][^1_3]
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_8][^1_9]</span>

<div align="center">⁂</div>


---



# 🟡 Phase 2 — Intermediate PySpark

## 🔗 Joins \& Aggregations


***

**Q1. Different join types in PySpark? When to use `left_anti`?**

**A:**


| Join Type | Returns | When to use |
| :-- | :-- | :-- |
| `inner` | Matching rows only | Standard join, most common |
| `left` | All left + matching right | Preserve all left records |
| `right` | All right + matching left | Rare, flip your tables instead |
| `full` / `outer` | All rows from both | Data reconciliation |
| `left_semi` | Left rows WHERE match exists | Existence check, no right cols |
| `left_anti` | Left rows WHERE NO match exists | Find orphan/missing records |
| `cross` | Cartesian product | 🚨 Almost never — explodes data |

```python
# left_anti: find customers who never placed an order
customers.join(orders, on="customer_id", how="left_anti").show()
```

💡 **Interview Tip:** `left_anti` is a ZS favourite — *"Find records in Table A not present in Table B."* It's the SQL `NOT IN` / `NOT EXISTS` equivalent. Mention it beats subqueries on Spark. ✅

***

**Q2. What is a Broadcast Join? Limitations?**

**A:** A broadcast join copies the **smaller table entirely to every executor's memory**, eliminating the shuffle of the larger table. Spark does this auto-matically if the table is under `spark.sql.autoBroadcastJoinThreshold` (default: **10MB**).[^2_1]

```python
from pyspark.sql.functions import broadcast

# Force broadcast on the smaller lookup table
result = large_transactions.join(broadcast(small_dim_table), on="product_id")
```

**Limitations 🚨:**

- Small table must fit in **executor memory** — default threshold 10MB (can tune to ~200MB safely)
- Can cause **OOM errors** if over-broadcast
- Does **not work well with large lookup tables** — network cost spikes

💡 **Interview Tip:** *"I always check `.explain()` to confirm BroadcastHashJoin is being used. If Spark chose SortMergeJoin on a 5MB table, I force broadcast manually."* Senior-level answer. 🔥

***

**Q3. `groupBy()` vs `rollup()` vs `cube()`?**

**A:**

```python
df.groupBy("region", "dept").agg(sum("sales"))
# Exact groups only: (region, dept)

df.rollup("region", "dept").agg(sum("sales"))
# Hierarchical subtotals: (region, dept), (region), (grand total)

df.cube("region", "dept").agg(sum("sales"))
# All combinations: (region, dept), (region), (dept), (grand total)
```

| Function | Combinations | Use Case |
| :-- | :-- | :-- |
| `groupBy` | Exact keys only | Standard aggregation |
| `rollup` | Hierarchical (left→right) | Drill-down reports |
| `cube` | All permutations | Full OLAP pivot tables |

💡 **Interview Tip:** `rollup` and `cube` generate `null` for the aggregate rows — use `coalesce(col, lit("ALL"))` to label them. Shows you've actually used it. 🎯

***

**Q4. Difference between `agg()` and `groupBy().agg()`?**

**A:** They're practically the same call — `agg()` is just shorthand when you've already called `groupBy()`. BUT `df.agg()` directly (without `groupBy`) computes **global aggregates across the entire DataFrame**:

```python
from pyspark.sql.functions import sum, avg, max, countDistinct

df.agg(sum("salary"), avg("salary"))          # global stats, no grouping

df.groupBy("dept").agg(                        # per-group stats
    sum("salary").alias("total_sal"),
    avg("salary").alias("avg_sal"),
    countDistinct("emp_id").alias("headcount")
)
```

💡 **Interview Tip:** Mention that chaining multiple `.agg()` calls inside ONE `groupBy()` is far more efficient than separate `groupBy()` calls — single shuffle pass. ⚡

***

**Q5. Pivot in PySpark — Performance concern?**

**A:**

```python
# Sales by region per quarter
df.groupBy("region") \
  .pivot("quarter", ["Q1", "Q2", "Q3", "Q4"]) \  # specify values!
  .agg(sum("sales"))
```

**Performance concern ⚠️:** Without specifying pivot values explicitly, Spark runs an **extra full scan** to discover all distinct values first — two passes on large data. Always pass the values list.[^2_2]

💡 **Interview Tip:** *"I always pre-define pivot values to avoid the extra scan. On 100M+ rows without it, I've seen 2x slower pivot jobs."* Real-world answer = offer points. 🌟

***

**Q6. Second-highest salary per department — Code it! 💻**

```python
from pyspark.sql.functions import col, dense_rank
from pyspark.sql.window import Window

windowSpec = Window.partitionBy("department").orderBy(col("salary").desc())

result = df.withColumn("rnk", dense_rank().over(windowSpec)) \
           .filter(col("rnk") == 2) \
           .select("department", "employee_name", "salary")

result.show()
```

💡 **Interview Tip:** Use `dense_rank()` not `rank()` — if two people tie for 1st, `rank()` skips rank 2, `dense_rank()` doesn't. Interviewers test exactly this edge case. ⚠️

***

**Q7. Find rows with duplicate email IDs — no subqueries?**

```python
from pyspark.sql.functions import count, col
from pyspark.sql.window import Window

# Method 1: Window function (most elegant)
w = Window.partitionBy("email")
df.withColumn("cnt", count("email").over(w)) \
  .filter(col("cnt") > 1) \
  .drop("cnt") \
  .show()

# Method 2: Join back on aggregated counts
dup_emails = df.groupBy("email").count().filter(col("count") > 1)
df.join(dup_emails, on="email", how="inner").show()
```

💡 **Interview Tip:** Method 1 (Window) keeps all duplicate rows in result — great for auditing. Method 2 (join) is more readable. Know both, explain the trade-off. 🎯

***

## 💾 Partitioning \& Storage


***

**Q8. `repartition()` vs `coalesce()`?**

**A:**


|  | `repartition(n)` | `coalesce(n)` |
| :-- | :-- | :-- |
| Shuffle | ✅ Full shuffle | ❌ No shuffle (mostly) |
| Direction | Increase or decrease | **Decrease only** |
| Balance | Perfectly balanced partitions | Can create uneven partitions |
| Speed | Slower (network I/O) | Faster |
| Use when | Need more partitions / rebalance skew | Reducing before write to disk |

```python
df.repartition(200)              # before heavy joins/groupBy — full shuffle
df.coalesce(10)                  # before df.write — avoid 1000 tiny files
df.repartition(50, col("dept"))  # partition by column for data locality
```

💡 **Interview Tip:** *"I use coalesce before writing to S3 to control file count. I use repartition by key column before joins to co-locate data and reduce shuffle."* 🔥

***

**Q9. What is Partition Pruning?**

**A:** Partition pruning means Spark **reads only the relevant partition folders** from disk/S3 instead of scanning the entire dataset. If your data is stored partitioned by `date`, a filter on `date = '2024-01-01'` skips every other folder entirely.[^2_3]

```python
# Data written as: s3://bucket/events/date=2024-01-01/
# s3://bucket/events/date=2024-01-02/

df = spark.read.parquet("s3://bucket/events/")
df.filter(col("date") == "2024-01-01")  # only reads date=2024-01-01 folder
```

💡 **Interview Tip:** Always filter on the **partition column** early. If you filter on a non-partition column, Spark reads everything. Use `.explain()` to confirm `PartitionFilters` appears in the plan. 🔍

***

**Q10. Write partitioned data to S3/ADLS?**

```python
df.write \
  .mode("overwrite") \
  .partitionBy("year", "month") \                  # creates folder hierarchy
  .parquet("s3://bucket/processed/transactions/")

# Or with dynamic partition overwrite (safe for incremental loads):
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
df.write.mode("overwrite").partitionBy("date").parquet("s3://bucket/data/")
```

💡 **Interview Tip:** `dynamic` partition overwrite only replaces partitions present in the current DataFrame — **doesn't wipe old partitions**. This is critical for incremental pipelines and ZS loves asking about it. ⚡

***

**Q11. Storage Levels in `persist()` — `MEMORY_AND_DISK` vs `MEMORY_ONLY`?**

```python
from pyspark import StorageLevel

df.persist(StorageLevel.MEMORY_ONLY)          # fast, OOM risk on large data
df.persist(StorageLevel.MEMORY_AND_DISK)      # spills to disk if RAM full
df.persist(StorageLevel.MEMORY_ONLY_SER)      # serialized = less RAM, more CPU
df.persist(StorageLevel.DISK_ONLY)            # slowest, but zero RAM use
```

| Level | Speed | Memory | Use When |
| :-- | :-- | :-- | :-- |
| `MEMORY_ONLY` | ⚡ Fastest | High | Small-medium, fits in RAM |
| `MEMORY_AND_DISK` | 🔄 Good | Medium | Default safe choice |
| `MEMORY_ONLY_SER` | Medium | Low | Memory constrained |
| `DISK_ONLY` | 🐢 Slow | Minimal | Very large, rarely re-used |

💡 **Interview Tip:** *"My default is MEMORY_AND_DISK — avoids OOM while still giving cache benefits. MEMORY_ONLY only when I'm sure the dataset fits."* 🎯

***

**Q12. Difference between `cache()` and `persist()`?**

**A:** `cache()` is literally just `persist(StorageLevel.MEMORY_AND_DISK)` — a convenience shortcut with no configuration options. `persist()` lets you **explicitly control the storage level**.[^2_4]

```python
df.cache()                                     # = MEMORY_AND_DISK, no control

df.persist(StorageLevel.MEMORY_ONLY_SER)       # full control over level

df.unpersist()                                 # always unpersist when done!
```

💡 **Interview Tip:** Always call `.unpersist()` when you're done with a cached DF — otherwise it occupies executor memory for the session's lifetime. Forgetting this = memory leak in long-running jobs. 🚨

***

## 🪟 Window Functions


***

**Q13. `rank()` vs `dense_rank()` vs `row_number()`?**

**A:** All three assign a sequential number per partition, but handle **ties differently**:

```python
from pyspark.sql.functions import rank, dense_rank, row_number
from pyspark.sql.window import Window

w = Window.partitionBy("dept").orderBy(col("salary").desc())

df.withColumn("rank",       rank().over(w)) \
  .withColumn("dense_rank", dense_rank().over(w)) \
  .withColumn("row_num",    row_number().over(w))
```

| Salary | `rank()` | `dense_rank()` | `row_number()` |
| :-- | :-- | :-- | :-- |
| 90000 | 1 | 1 | 1 |
| 90000 | 1 | 1 | 2 |
| 80000 | **3** (gap!) | **2** (no gap) | 3 |
| 70000 | 4 | 3 | 4 |

💡 **Interview Tip:** This is the \#1 most asked window function question at ZS/Barclays. The key: `rank()` skips numbers after ties, `dense_rank()` doesn't, `row_number()` is always unique regardless. Memorize the table. 🔥

***

**Q14. Moving Average using Window Functions?**

```python
from pyspark.sql.functions import avg
from pyspark.sql.window import Window

# 7-day moving average of sales per store
w = Window.partitionBy("store_id") \
          .orderBy("date") \
          .rowsBetween(-6, 0)         # current row + 6 preceding rows

df.withColumn("moving_avg_7d", avg("sales").over(w))
```

**Frame options:**

- `.rowsBetween(-6, 0)` → Physical rows (7-row rolling window)
- `.rangeBetween(-6, 0)` → Value-based range (based on ORDER BY column value)

💡 **Interview Tip:** Always clarify `rowsBetween` vs `rangeBetween` — most people don't know the difference. `rowsBetween` is usually what you want for fixed rolling windows. 🎯

***

**Q15. Running Total per User?**

```python
from pyspark.sql.functions import sum
from pyspark.sql.window import Window

w = Window.partitionBy("user_id") \
          .orderBy("transaction_date") \
          .rowsBetween(Window.unboundedPreceding, Window.currentRow)

df.withColumn("running_total", sum("amount").over(w))
```

💡 **Interview Tip:** `Window.unboundedPreceding` = "from the very first row of the partition." Combine with `.currentRow` for cumulative sum. Use `unboundedFollowing` for reverse cumulative. ✅

***

**Q16. Difference between `lag()` and `lead()`?**

**A:**

- `lag(col, n)` — Looks **n rows BEHIND** current row (previous value)
- `lead(col, n)` — Looks **n rows AHEAD** of current row (next value)

```python
from pyspark.sql.functions import lag, lead

w = Window.partitionBy("user_id").orderBy("date")

df.withColumn("prev_purchase", lag("amount", 1).over(w))   # last purchase
  .withColumn("next_purchase", lead("amount", 1).over(w))  # upcoming purchase
  .withColumn("day_over_day_change", col("amount") - lag("amount", 1).over(w))
```

💡 **Interview Tip:** Classic use case = **month-over-month change**, churn prediction (time since last event), or detecting consecutive events. ZS loves this for their analytics pipelines. 📈

***

## ⚙️ Performance Basics


***

**Q17. What is Catalyst Optimizer? What does it do internally?**

**A:** Catalyst is Spark SQL's **query optimization engine** — it takes your DataFrame/SQL code through 4 phases before executing:[^2_5][^2_2]

1. **Analysis** → Resolves column names, table references, creates AST (Abstract Syntax Tree)
2. **Logical Optimization** → Applies rules: predicate pushdown, constant folding, column pruning
3. **Physical Planning** → Picks execution strategies (BroadcastHashJoin vs SortMergeJoin) using cost estimation
4. **Code Generation** → Generates optimized **JVM bytecode** via Tungsten[^2_1]

💡 **Interview Tip:** Say *"Catalyst is why writing `df.filter().select()` is equivalent to `df.select().filter()` in performance — Catalyst reorders it anyway."* Mindblowing to interviewers. 🤯

***

**Q18. Tungsten Execution Engine — How it differs from Catalyst?**

**A:** Catalyst optimizes the **query plan** (what to execute). Tungsten optimizes the **actual execution** (how to execute it at the hardware level):[^2_6][^2_1]


|  | Catalyst | Tungsten |
| :-- | :-- | :-- |
| Layer | Logical/Physical planning | Physical execution |
| Focus | Query optimization | CPU + Memory efficiency |
| Key technique | Predicate pushdown, join reorder | Whole-stage codegen, off-heap memory |
| Output | Optimized physical plan | JVM bytecode + vectorized ops |

**Tungsten's 3 key tricks**:[^2_1]

- **Whole-stage code generation** — merges multiple ops into one compiled function
- **Vectorized execution** — processes rows in batches (SIMD CPU instructions)
- **Off-heap memory** — bypasses JVM GC entirely, manages memory directly

💡 **Interview Tip:** *"Catalyst = smart planner, Tungsten = fast runner."* One-liner that nails it. 🏃

***

**Q19. What is AQE (Adaptive Query Execution) in Spark 3.x?**

**A:** AQE is Spark 3.0's **runtime optimizer** — unlike Catalyst which plans ahead, AQE **replans mid-execution** using actual runtime statistics. Enabled by default since Spark 3.2.[^2_7][^2_8][^2_3]

**3 core AQE superpowers:**

- **Dynamic partition coalescing** — shrinks 200 shuffle partitions to 10 if data is small[^2_7]
- **SortMergeJoin → BroadcastJoin conversion** — if one side turns out small at runtime, switches join strategy[^2_8]
- **Skew join optimization** — splits fat partitions into smaller chunks, handles data skew automatically[^2_7]

```python
spark.conf.set("spark.sql.adaptive.enabled", "true")          # default ON in 3.2+
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true") # handle skew
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
```

💡 **Interview Tip:** *"AQE solves the \#1 cause of slow Spark jobs — data skew — automatically. Before Spark 3, you had to manually salting."* This is a ⭐ answer.[^2_9]

***

**Q20. What is `explain()` and how do you read it?**

```python
df.join(other, on="id").filter(col("age") > 30).explain()        # basic
df.join(other, on="id").filter(col("age") > 30).explain(True)    # full plan
df.explain(mode="formatted")  # Spark 3+ — cleaner readable format
```

**How to read the output — bottom to top 🔽:**

```
== Physical Plan ==
*(2) Filter (age > 30)                    ← 3rd: filter applied
+- *(2) BroadcastHashJoin [id]            ← 2nd: join type (broadcast = good!)
   :- *(1) Scan parquet [...]             ← 1st: reads large table
   +- BroadcastExchange                   ← small table broadcast
```

**Key things to spot:**

- `BroadcastHashJoin` ✅ vs `SortMergeJoin` ⚠️ (shuffle-heavy)
- `PartitionFilters` ✅ → pruning working
- `PushedFilters` ✅ → predicate pushdown working
- `Exchange` 🚨 → shuffle happening — minimize these

💡 **Interview Tip:** *"I use `explain(mode='formatted')` in Databricks to find unexpected SortMergeJoins or missing partition pruning — it's my first debugging step for slow jobs."*  Immediately signals production experience. 🔥[^2_3]

***

## 🎯 ZS Associates / Barclays Pune — Phase 2 Meta Tips

- 🪟 **Window functions are the \#1 tested topic** — know `rank/dense_rank/row_number` cold
- 🔀 **Shuffle = cost** — mention it whenever joins, groupBy, or repartition comes up
- 🧠 **`explain()` = your debugging superpower** — mention it proactively
- ⚡ **AQE is your flex** — most freshers don't know Spark 3.x features[^2_3]
- 💾 **`coalesce` before write, `repartition` before join** — production pattern ZS loves
- 🚨 **`cache()` without `unpersist()`** = memory leak — always mention this pair[^2_4]
<span style="display:none">[^2_10][^2_11][^2_12][^2_13][^2_14][^2_15][^2_16][^2_17]</span>

<div align="center">⁂</div>


<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>



## 🔷 CORE CONCEPTS

### What is PySpark?

PySpark is the **Python API for Apache Spark** — it lets you write distributed data processing apps in Python instead of Scala/Java. It bridges Python and Spark via the **Py4J** library under the hood. You get Spark's power (speed, scale, fault tolerance) with Python's simplicity.[^1_1][^1_5]

### Spark vs Hadoop

| Feature | Hadoop | Spark |
| :-- | :-- | :-- |
| Processing | Disk-based (MapReduce) | **In-memory** |
| Speed | Baseline | **10–100x faster** for iterative jobs |
| Real-time | ❌ Batch only | ✅ Batch + Streaming |
| ML | External libs only | **MLlib built-in** |
| Ease of Use | Verbose Java MapReduce | High-level Python/SQL APIs |
| Cost | Cheaper | More expensive (RAM-heavy) |

[^1_6][^1_11]

### Driver vs Executor 🧠

Spark uses a **Master-Slave architecture**:[^1_12]

- **Driver** — The brain 🧠. Runs your `main()`, builds the DAG, schedules tasks, and coordinates everything. One per application.
- **Executor** — The workers 💪. Live on worker nodes, actually run tasks and store data in memory/disk. Multiple per application.
- Executors register themselves with the Driver at startup.[^1_12]


### What is a Cluster Manager?

The **Cluster Manager** allocates resources (CPU, RAM) to Spark applications and launches executors. Spark supports:[^1_7][^1_9]

- **YARN** — Hadoop's resource manager (most common in enterprise)
- **Kubernetes** — Container-native, modern cloud deployments
- **Standalone** — Spark's own built-in manager (simple setups)

***

## 🔷 RDD vs DATAFRAME

### What is an RDD?

**Resilient Distributed Dataset** — Spark's **lowest-level** data abstraction. It's an immutable, distributed collection of objects partitioned across the cluster. "Resilient" = it can recover from node failures using **lineage** (knows how it was built).[^1_9][^1_15]

### RDD vs DataFrame

|  | RDD | DataFrame |
| :-- | :-- | :-- |
| API Level | Low-level | High-level |
| Schema | ❌ No schema | ✅ Has schema |
| Optimization | Manual | **Catalyst Optimizer** auto-optimizes |
| Type Safety | ✅ Compile-time | Runtime |
| Ease of Use | Verbose | SQL-like, concise |
| Use Case | Complex unstructured ops | 99% of real-world ETL |

[^1_9]

### Do DataFrames Internally Use RDDs?

**Yes, historically** — DataFrames were built on top of RDDs. In modern Spark (2.x+), DataFrames use an optimized internal format (**Tungsten**) and the **Catalyst query planner**, but the execution engine still ultimately processes data as RDDs at the lowest level. Think of DataFrame as RDD + schema + optimizer. ⚡

***

## 🔷 TRANSFORMATIONS vs ACTIONS

### What is a Transformation?

Operations that **define** what to do — they return a new DataFrame/RDD but **don't execute** anything immediately (lazy!).[^1_15]

Examples: `filter()`, `select()`, `groupBy()`, `join()`, `withColumn()`, `map()`, `flatMap()`

### What is an Action?

Operations that **trigger execution** and return a result to the driver or write to storage.[^1_9]

Examples: `show()`, `collect()`, `count()`, `save()`, `write()`, `take(n)`, `first()`

### What is Lazy Evaluation?

Spark **does NOT execute transformations when you define them**. It waits until an **action** is called, then builds an optimized execution plan (DAG) and runs everything in one optimized pass. Think of transformations as a recipe — Spark reads the whole recipe before cooking, so it can optimize steps.[^1_15]

***

## 🔷 DATAFRAME BASICS

### What is a Schema?

A schema is the **structure definition** of a DataFrame — column names, data types, and nullable flags. Example: `StructType([StructField("name", StringType(), True)])`. Enforces data contracts in pipelines. 🔐

### What is Schema Inference?

When you read a file with `inferSchema=True`, Spark **scans the data** (usually a sample) and guesses data types automatically. ⚠️ **Cost warning** — it triggers an extra read pass. In production, **always define schema explicitly** — faster and safer.

### `select()` vs `withColumn()`

|  | `select()` | `withColumn()` |
| :-- | :-- | :-- |
| Output | Only selected columns | All existing columns + new/modified |
| Use for | Projecting a subset | Adding/updating a single column |
| Performance | Slightly more efficient for many cols | Can be slow if chained heavily |

```python
df.select("name", "age")                          # keep only these cols
df.withColumn("age_plus_1", col("age") + 1)       # add new col, keep rest
```


***

## 🔷 FUNCTIONS

### `col()` and `lit()`

- **`col("column_name")`** — References an **existing column** in the DataFrame. `col("age")`
- **`lit(value)`** — Creates a **literal constant column**. `lit(42)` or `lit("India")`

```python
df.withColumn("country", lit("India"))     # adds constant column
df.filter(col("age") > 18)                 # references existing column
```


### `when()`

Used for **conditional logic** — Spark's equivalent of `IF/CASE WHEN` in SQL. Returns a Column expression.

```python
from pyspark.sql.functions import when, col

df.withColumn("status", 
    when(col("age") < 18, "minor")
    .when(col("age") < 60, "adult")
    .otherwise("senior")
)
```


***

## 🔷 FILE HANDLING

### CSV vs Parquet

|  | CSV | Parquet |
| :-- | :-- | :-- |
| Format | Row-based, text | **Columnar**, binary |
| Schema | ❌ No built-in schema | ✅ Schema embedded |
| Compression | Poor | **Excellent** (Snappy/GZIP) |
| Read speed (select few cols) | Reads ALL columns | Reads **only needed columns** |
| Human readable | ✅ Yes | ❌ No |
| Use case | Raw ingestion, exports | **Storage, analytics, production** |

### Why is Parquet Preferred in Spark?

Three big reasons 🎯:

- **Columnar reads** — if you query 3/50 columns, Parquet reads only those 3. CSV reads all 50.
- **Compression** — Parquet files are typically 5–10x smaller than equivalent CSVs.
- **Schema embedded** — no schema inference needed, faster reads.

***

## 🔷 PERFORMANCE (BASIC)

### Benefits of Lazy Evaluation

- Spark can **reorder, merge, and skip** unnecessary steps before executing[^1_15]
- Avoids redundant intermediate data materialization
- Enables **predicate pushdown** (filter early, read less data)
- Only computes what's actually needed when an action is called


### What is a DAG?

**Directed Acyclic Graph** — Spark builds a visual map of all transformations as a graph of stages. "Directed" = flows one way. "Acyclic" = no loops. The **DAG Scheduler** converts this into physical execution stages and tasks. You can view it in the **Spark UI**.[^1_9]

### Why is Spark Fast? ⚡

- **In-memory computation** — avoids disk I/O between steps (unlike Hadoop MapReduce)[^1_6]
- **Lazy evaluation + Catalyst Optimizer** — rewrites your query for maximum efficiency
- **Tungsten engine** — low-level memory management, binary processing
- **DAG-based execution** — minimizes data shuffles across the network

***

## 🔷 BONUS — Common Interview Traps

### `show()` vs `collect()`

|  | `show()` | `collect()` |
| :-- | :-- | :-- |
| Returns | Prints to console | Returns List to driver |
| Safe for large data? | ✅ (default 20 rows) | ⚠️ **OOM risk** — pulls ALL data to driver |
| Use in production | Debugging only | Never on large datasets |

> 🚨 **Interview trap**: "When would you NOT use collect()?" — Answer: On any large dataset. Use `take(n)` or `write()` instead.

### `repartition()` vs `coalesce()`

|  | `repartition(n)` | `coalesce(n)` |
| :-- | :-- | :-- |
| Direction | Increase **or** decrease | **Only decrease** |
| Shuffle | ✅ Full shuffle | ❌ Minimal/no shuffle |
| Balance | Even partition sizes | May have uneven partitions |
| Use case | Before heavy joins | Before writing to fewer files |

### Is Data Mutable in PySpark?

**No.** RDDs and DataFrames are **immutable** — every transformation returns a **new** DataFrame. The original is never modified. This is core to Spark's fault tolerance model (lineage-based recovery).

### Are Transformations Executed Immediately?

**No** — this is **lazy evaluation**. Transformations like `filter()`, `select()`, `join()` are only *recorded* in the DAG. Execution happens **only when an action** (`show()`, `count()`, `write()`) is triggered. This is one of Spark's biggest performance advantages.[^1_15]
<span style="display:none">[^1_10][^1_13][^1_14][^1_2][^1_3][^1_4][^1_8]</span>

<div align="center">⁂</div>



---


## 🔷 PART 2: TRANSFORMATIONS \& OPTIMIZATION

### `groupBy()` vs `reduceByKey()`

|  | `groupBy()` | `reduceByKey()` |
| :-- | :-- | :-- |
| API | **DataFrame API** | RDD API only |
| How it works | Groups all rows, then aggregates | Combines values **locally** before shuffle |
| Shuffle data | More data shuffled | **Less data shuffled** (pre-aggregates) |
| Use case | 99% of your DE work | Legacy RDD pipelines |

In modern Spark (DataFrame API), `groupBy().agg()` is what you always use. `reduceByKey()` is RDD-era — mostly trivia now unless you work on old codebases.[^2_1]

```python
# DataFrame way — use this always
df.groupBy("department").agg(sum("salary").alias("total_sal"))
```


### `repartition()` vs `coalesce()`

|  | `repartition(n)` | `coalesce(n)` |
| :-- | :-- | :-- |
| Direction | ↑ increase or ↓ decrease | ↓ **decrease only** |
| Shuffle | ✅ Full shuffle | ❌ Minimal shuffle |
| Partition balance | Even | May be uneven |
| When to use | Before heavy joins/groupBy | Before `.write()` to reduce output files |
| Cost | Expensive | Cheap |

> ⚠️ **Rule of thumb**: Before writing output → `coalesce()`. Before a join on a large dataset → `repartition()`.

***

### What is Data Skew? 😱

**Data skew** = some partitions have way more data than others, causing a few tasks to take forever while others finish immediately. Classic symptom: **one task stuck at 99% forever** in Spark UI.

**How to identify it:**

```python
df.groupBy("key_column").count().orderBy("count", ascending=False).show(10)
```

**3 ways to handle it**:[^2_2][^2_3]

**1. Broadcast Join** — if one table is small (< ~10MB default), broadcast it:

```python
from pyspark.sql.functions import broadcast
df_big.join(broadcast(df_small), "key")
```

**2. Salting** — artificially distribute skewed keys across partitions:[^2_1]

```python
from pyspark.sql.functions import col, concat, lit, floor, rand, explode, array

SALT_BUCKETS = 10

# Add random salt to the large skewed table
salted_big = df_big.withColumn("salt", floor(rand() * SALT_BUCKETS).cast("int")) \
                   .withColumn("salted_key", concat(col("id"), lit("_"), col("salt")))

# Explode small table to match all salt values
salted_small = df_small.withColumn("salt", explode(array([lit(i) for i in range(SALT_BUCKETS)]))) \
                       .withColumn("salted_key", concat(col("id"), lit("_"), col("salt")))

result = salted_big.join(salted_small, "salted_key").drop("salt", "salted_key")
```

**3. AQE (Adaptive Query Execution)** — Spark 3+ auto-fixes skew at runtime:[^2_4]

```python
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
# Databricks: AQE is ON by default ✅
```


***

### Caching — When to `.cache()` 🗄️

**`.cache()`** stores a DataFrame in memory (or memory+disk) so it's NOT recomputed on every action.

**✅ Use cache when:**

- You call `count()`, then `show()`, then `write()` on the **same DataFrame**
- Iterative ML/aggregation pipelines reusing the same base data
- After an expensive join or aggregation that feeds multiple downstream steps

**❌ Don't cache when:**

- DataFrame is used only once
- Dataset is huge and RAM is limited (causes GC pressure)
- In a write-once pipeline

```python
df_transformed = df.filter(col("status") == "active").join(df_lookup, "id")
df_transformed.cache()          # materialize once

df_transformed.count()          # triggers cache
df_transformed.show()           # reads from cache, no recompute
df_transformed.write.parquet("output/")  # reads from cache again
```

> ⚠️ **Always unpersist** when done: `df_transformed.unpersist()`

***

## 🔷 PART 3: SPARK SQL \& UDFs

### Temp View + SQL Query

```python
df.createOrReplaceTempView("employees")

result = spark.sql("""
    SELECT department, COUNT(*) as headcount, AVG(salary) as avg_salary
    FROM employees
    WHERE status = 'active'
    GROUP BY department
    ORDER BY avg_salary DESC
""")
result.show()
```

> `createOrReplaceTempView` → session-scoped. `createGlobalTempView` → accessible across sessions via `global_temp.table_name`.

***

### What is a UDF? ⚠️

A **User Defined Function** lets you run custom Python logic column-by-column. It's a **black box to Spark** — Catalyst optimizer can't optimize it.

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Define UDF
@udf(returnType=StringType())
def classify_salary(salary):
    if salary > 100000:
        return "high"
    elif salary > 50000:
        return "mid"
    return "low"

df.withColumn("salary_band", classify_salary(col("salary")))
```

**When to AVOID UDFs** 🚨:

- When native Spark functions (`when()`, `regexp_replace()`, `concat()`, etc.) can do the same job
- On very large datasets — UDFs serialize data to Python, row by row = **massive overhead**
- Instead use **Pandas UDFs (vectorized)** for better performance:

```python
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf(StringType())
def classify_salary_v2(salary: pd.Series) -> pd.Series:
    return salary.apply(lambda x: "high" if x > 100000 else "mid" if x > 50000 else "low")
```


### Spark SQL vs DataFrame API

|  | Spark SQL | DataFrame API |
| :-- | :-- | :-- |
| Syntax | Standard SQL | Chained Python methods |
| Readability | ✅ Great for analysts | ✅ Great for engineers |
| Type safety | Lower | Higher |
| Optimization | Both use **Catalyst** equally | Same execution plan |
| Debugging | Harder (string queries) | Easier (method chain) |
| Reusability | Less reusable | **More composable** |

> 🔑 **Key insight**: Both compile to the **same physical plan** — there is zero performance difference. Use SQL for reporting/ad-hoc, DataFrame API for production pipelines.

***

## 🔷 PART 4: SCENARIO-BASED

### Large Table + Small Lookup Table = Broadcast Join 📡

**Problem**: Default join shuffles BOTH tables. For a 10GB left table + 500MB lookup, this is wasteful.

**Solution**: Broadcast the small table so every executor gets a full copy — zero shuffle on the big table:[^2_1]

```python
from pyspark.sql.functions import broadcast

# Option 1: Explicit hint (most reliable)
result = df_large.join(broadcast(df_lookup), "product_id", "left")

# Option 2: Config-based auto-broadcast (default threshold: 10MB)
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 50 * 1024 * 1024)  # 50MB

# Option 3: SQL hint
spark.sql("SELECT /*+ BROADCAST(lookup) */ * FROM events JOIN lookup ON events.id = lookup.id")
```

> 💡 On Databricks, check the query plan with `df.explain("formatted")` to verify `BroadcastHashJoin` is used.

***

## 🔷 COMMONLY ASKED: Advanced Concepts

### `rank()` vs `dense_rank()` 🏆

The difference is **gap behavior on ties**:[^2_5][^2_6][^2_7]


| Scores | `rank()` | `dense_rank()` |
| :-- | :-- | :-- |
| 100 | 1 | 1 |
| 90 | 2 | 2 |
| 90 | 2 | 2 |
| 80 | **4** (gap!) | **3** (no gap) |

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, dense_rank

w = Window.partitionBy("department").orderBy(col("salary").desc())

df.withColumn("rank", rank().over(w)) \
  .withColumn("dense_rank", dense_rank().over(w))
```

> 📌 **Interview answer**: Use `dense_rank()` when you want consecutive ranking (leaderboards). Use `rank()` when gaps should reflect actual position (Olympics-style).

***

### What Happens During a Shuffle? 🔀

A **shuffle** is the most expensive operation in Spark — it involves moving data **across executors over the network**. It's triggered by wide transformations: `groupBy`, `join`, `distinct`, `repartition`.

Steps:

1. **Map phase** — each task writes partitioned shuffle files to local disk
2. **Network transfer** — downstream executors fetch their partition data over the network
3. **Reduce phase** — data is sorted and aggregated in the new partitions

> ⚠️ Shuffle writes to disk and transfers over network — both slow. This is WHY Spark engineers obsess over minimizing shuffles.

***

### Why Are Joins Expensive? 💸

Joins are expensive because they almost always **trigger a shuffle** — Spark must ensure all rows with the same join key end up in the same partition across the cluster. This means:[^2_8]

- Full data movement across the network
- Disk I/O for shuffle files
- Risk of data skew if keys are unevenly distributed

Exception: **Broadcast Hash Join** avoids shuffle entirely — always prefer it when one side is small.

***

### How Does Spark Handle Fault Tolerance? 🛡️

Spark uses **lineage-based recovery**, NOT data replication:[^2_8]

1. **Immutable data + lineage** — every transformation is recorded. If a partition is lost, Spark re-computes it from its parent using the DAG.
2. **DAG Scheduler retries** — failed tasks are retried on a different executor automatically.
3. **Checkpointing** — for long-running streaming jobs, Spark materializes state to S3/HDFS to truncate lineage and speed up recovery.[^2_8]

> 💡 This is why transformations must be **deterministic** — Spark may re-run them on failure. If you use `rand()` without a fixed seed in a lineage path, you could get different data on recompute.
<span style="display:none">[^2_10][^2_11][^2_12][^2_13][^2_14][^2_15][^2_16][^2_9]</span>

<div align="center">⁂</div>


---

# PART 2: THEORY QUESTIONS (10)

11. What is the difference between distinct() and dropDuplicates()?
12. What is lineage in Spark? Why is it important?
13. What are narrow and wide transformations?
14. What is the difference between map() and flatMap()?
15. What is the difference between persist() and cache()?
16. What happens if executor fails in Spark?
17. What is the difference between DataFrame and Dataset?
18. What is serialization in Spark? Why is it important?
19. What is the role of SparkContext?
20. What is a partition in Spark? Why is it important?,, give interview tips for each question too

**TL;DR** 🎯 — Questions 11–20 are the "second wave" questions interviewers ask after you clear the basics. These separate juniors from mid-level engineers. Each answer below has an interview tip that gives you the edge 💪

***

## Q11. `distinct()` vs `dropDuplicates()`

Both remove duplicate rows — but `dropDuplicates()` is strictly more powerful.


|  | `distinct()` | `dropDuplicates()` |
| :-- | :-- | :-- |
| Scope | ALL columns | Subset of columns |
| Flexibility | ❌ Fixed | ✅ Choose specific cols |
| Internally | Calls `dropDuplicates()` | Native implementation |

```python
df.distinct()                                       # dedupe on all columns
df.dropDuplicates(["email"])                        # dedupe on email only
df.dropDuplicates(["email", "phone"])               # combo key dedup
```

> 🎤 **Interview tip**: Say _"distinct() is actually syntactic sugar for dropDuplicates() on all columns. In production ETL I always prefer dropDuplicates() with a business key (like customer_id or email) rather than full-row comparison — it's more intentional and performant."_

***

## Q12. Lineage in Spark 🔗

**Lineage** is Spark's complete record of every transformation applied to create a DataFrame — essentially a "recipe" stored as a DAG. It's the backbone of Spark's fault tolerance.

When a partition is lost (executor crash), Spark doesn't recover from a backup — it **re-runs the lineage** from the original source to recreate that exact partition.[^3_1]

```python
df = spark.read.parquet("s3://bucket/raw/")
df2 = df.filter(col("status") == "active")
df3 = df2.withColumn("upper_name", upper(col("name")))

df3.explain()  # shows the full lineage / execution plan
```

> 🎤 **Interview tip**: _"Lineage is why Spark doesn't need HDFS replication for fault tolerance. The trade-off is that very long lineages (100+ transformations) slow down recovery. That's when you use `.checkpoint()` to truncate the lineage and write state to S3."_

***

## Q13. Narrow vs Wide Transformations ⚡

This is one of the most important Spark internals to understand.


|  | Narrow | Wide |
| :-- | :-- | :-- |
| Data movement | No shuffle — each partition maps to **one** output partition | **Shuffle required** — data crosses executors |
| Examples | `filter()`, `select()`, `map()`, `union()` | `groupBy()`, `join()`, `distinct()`, `repartition()` |
| Performance | Fast ✅ | Expensive ⚠️ |
| Fault recovery | Simple — recompute from parent | Expensive — may need to re-run upstream stages |

> 🎤 **Interview tip**: _"This is WHY we always push filters (`filter()`, `where()`) before joins — they're narrow transformations that reduce data before the expensive wide shuffle. Catalyst optimizer does this automatically via predicate pushdown, but it's good practice to be explicit."_

***

## Q14. `map()` vs `flatMap()`

Both are RDD-level transformations. The difference is **how many outputs each input produces**.


|  | `map()` | `flatMap()` |
| :-- | :-- | :-- |
| Input → Output | 1 element → **1 element** | 1 element → **0 or more elements** |
| Result shape | Same length as input | Can be longer (flattened) |
| Use case | Transform each row | Split/explode rows |

```python
rdd = sc.parallelize(["hello world", "spark rocks"])

# map: 2 inputs → 2 outputs (list of lists)
rdd.map(lambda x: x.split()).collect()
# [['hello', 'world'], ['spark', 'rocks']]

# flatMap: 2 inputs → 4 outputs (flat list)
rdd.flatMap(lambda x: x.split()).collect()
# ['hello', 'world', 'spark', 'rocks']
```

> 🎤 **Interview tip**: _"In DataFrame world, `flatMap()` equivalent is `explode()` — far more common in real ETL. If asked to split comma-separated tags into individual rows, I'd use `split()` + `explode()` on a DataFrame, not RDD flatMap."_

***

## Q15. `persist()` vs `cache()`

`cache()` is simply `persist()` with a **default storage level** — they're the same thing under the hood.[^3_2]


|  | `cache()` | `persist(storageLevel)` |
| :-- | :-- | :-- |
| Control | ❌ Fixed: MEMORY_AND_DISK | ✅ You choose the level |
| Default level | `MEMORY_AND_DISK` | Whatever you specify |
| Use case | Quick dev/simple pipelines | Production — fine-tune memory vs disk |

```python
from pyspark import StorageLevel

df.cache()                                          # MEMORY_AND_DISK (default)

df.persist(StorageLevel.MEMORY_ONLY)               # fastest, OOM risk on large data
df.persist(StorageLevel.MEMORY_AND_DISK)           # spills to disk if memory full
df.persist(StorageLevel.DISK_ONLY)                 # large data, slow reads
df.persist(StorageLevel.MEMORY_ONLY_2)             # replicated across 2 nodes (fault tolerant)
```

> 🎤 **Interview tip**: _"In Databricks I always use `persist(MEMORY_AND_DISK)` explicitly — it's more intentional than `cache()`. For streaming aggregations or iterative ML, I use `MEMORY_ONLY_2` for the replication safety. Always pair with `unpersist()` to release memory."_

***

## Q16. What Happens When an Executor Fails? 🛡️

Spark handles this gracefully through its lineage + retry mechanism:[^3_1]

1. **Driver detects** the executor is unresponsive (heartbeat timeout)
2. **Tasks on that executor are marked as failed**
3. **DAG Scheduler reschedules** those tasks on surviving executors
4. Spark **re-computes lost partitions** using lineage from parent data
5. If the same task fails 4 times (default), the **whole job fails**
```python
# Configure retry behavior
spark.conf.set("spark.task.maxFailures", "4")        # default
spark.conf.set("spark.executor.heartbeatInterval", "10s")
spark.conf.set("spark.network.timeout", "120s")
```

> 🎤 **Interview tip**: _"The nuance here is — if the lost data was CACHED, that cache entry is also lost and gets recomputed. If your lineage is very long, checkpoint first before caching expensive intermediate results. On Databricks, Delta Lake handles this more gracefully with ACID guarantees."_

***

## Q17. DataFrame vs Dataset 📦

**Key rule**: Dataset API is **only available in Scala/Java**. In PySpark, you always work with DataFrames.[^3_3]


|  | DataFrame | Dataset |
| :-- | :-- | :-- |
| Type safety | ❌ Runtime errors | ✅ Compile-time type checking |
| Language | Python, Scala, Java, R | **Scala and Java only** |
| Internally | `Dataset[Row]` | `Dataset[T]` (custom type) |
| Optimization | Catalyst + Tungsten ✅ | Catalyst + Tungsten ✅ |
| Use case | 99% of Spark usage | Strongly-typed Scala pipelines |

```python
# In PySpark — DataFrame IS Dataset[Row] conceptually
df = spark.read.parquet("s3://bucket/data/")
# You're always using DataFrame in Python — Dataset[T] doesn't exist in PySpark
```

> 🎤 **Interview tip**: _"This is a common trick question for Python engineers. The honest answer: Dataset[T] doesn't exist in PySpark — it's Scala/Java only. DataFrame in Spark is internally `Dataset[Row]`. As a Python DE, I focus on DataFrames and Spark SQL. Strong-typing at the pipeline level I handle via schema enforcement with StructType."_

***

## Q18. Serialization in Spark 🔄

**Serialization** = converting objects into a byte stream so they can be sent over the network (executor-to-executor, driver-to-executor) or written to disk. It's critical for shuffle performance.[^3_4]

Spark supports two serializers:[^3_2]


|  | Java Serialization | Kryo Serialization |
| :-- | :-- | :-- |
| Default? | ✅ Yes | ❌ Must configure |
| Speed | Slow | **~10x faster** |
| Size | Large | **Compact** |
| Compatibility | Works with all `Serializable` | Must **register** custom classes |

```python
# Enable Kryo (recommended for shuffle-heavy jobs)
spark = SparkSession.builder \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryo.registrationRequired", "false") \
    .getOrCreate()
```

> 🎤 **Interview tip**: _"Serialization matters most during shuffles and when broadcasting large variables. I always enable Kryo on shuffle-heavy PySpark jobs on Databricks — easy config win. Also relevant for UDFs — Python UDFs serialize data row-by-row between JVM and Python, which is WHY Pandas UDFs (PyArrow, vectorized) are much faster."_

***

## Q19. Role of SparkContext 🕹️

`SparkContext` (pre-Spark 2.0) was the original entry point to Spark — it represented the connection to the Spark cluster.[^3_5]

**In modern Spark (2.0+)**, `SparkSession` replaced it as the unified entry point — it wraps SparkContext + SQLContext + HiveContext:[^3_6]

```python
# OLD way (Spark 1.x)
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("app")
sc = SparkContext(conf=conf)

# MODERN way (Spark 2.0+) — SparkContext accessible via session
spark = SparkSession.builder.appName("app").getOrCreate()
sc = spark.sparkContext           # still accessible when needed (e.g., for RDD ops)

# SparkContext still needed for:
sc.broadcast(lookup_dict)         # broadcast variables
sc.accumulator(0)                 # accumulators
sc.textFile("hdfs://...")         # low-level RDD file reads
```

> 🎤 **Interview tip**: _"In production PySpark (2.0+), I always use SparkSession. SparkContext is still relevant when working with broadcast variables or accumulators — those APIs live on `spark.sparkContext`. On Databricks, `spark` and `sc` are pre-initialized — you never create them manually."_

***

## Q20. What is a Partition? Why Does it Matter? 🍕

A **partition** is a logical chunk of data that lives on one executor and is processed by one task. It's the **unit of parallelism** in Spark.

```python
df = spark.read.parquet("s3://bucket/data/")
print(df.rdd.getNumPartitions())     # check current partition count

# Default shuffle partitions = 200 (often wrong for your data size)
spark.conf.set("spark.sql.shuffle.partitions", "100")   # tune this!

# Rules of thumb:
# - Target partition size: 128MB–200MB each
# - Number of partitions ≈ 2-3x number of CPU cores in cluster
# - Too few → idle CPUs, memory pressure
# - Too many → task scheduling overhead, small files problem
```

**Why it matters:**

- Too **few** partitions → some executors sit idle, OOM on large data
- Too **many** partitions → task scheduling overhead, tiny shuffle files
- **Skewed** partitions → one task takes 10x longer than others (data skew!)

> 🎤 **Interview tip**: _"This is the most practical tuning lever in Spark. I always check partition count and size with `df.rdd.getNumPartitions()` and `df.explain()`. On Databricks with AQE enabled, Spark auto-coalesces shuffle partitions — but for reads from S3, I still manually repartition after loading if I know the file sizes are uneven."_
<span style="display:none">[^3_10][^3_11][^3_12][^3_13][^3_14][^3_15][^3_16][^3_7][^3_8][^3_9]</span>

<div align="center">⁂</div>

---



## 🔷 SCENARIO QUESTIONS


***

### Q1. 🐢 Large Data Join — 500GB + 5MB Table

**Q: Join is very slow. What do you do?**

**A:** The 5MB table is tiny — this is a classic **Broadcast Join** case 📡

- Default join → Spark shuffles BOTH tables → 500GB moves across network → slow ❌
- Fix → **broadcast the small table** to every executor → zero shuffle on the big table → fast ✅
- Auto-broadcast threshold is 10MB by default — 5MB qualifies automatically
- Verify the fix → run `df.explain()` → you should see `BroadcastHashJoin` not `SortMergeJoin`
- On Databricks → AQE can auto-detect and switch join strategy at runtime

> 🎤 **Interview tip**: _"I'd first confirm the join type in Spark UI — if it's SortMergeJoin with a tiny table, it's misconfigured. I'd force broadcast explicitly and bump the threshold config. Also verify the 5MB is post-filter size, not pre-filter."_

***

### Q2. 📊 Data Skew — 80% Data on One Key

**Q: dept_id = 10 has 80% of data. Job is slow. Fix it?**

**A:** One partition = one task. That one task is processing 80% of all data alone 😱

**Fix options (in order of preference):**

1. 🥇 **AQE (Spark 3+)** → enable `spark.sql.adaptive.skewJoin.enabled=true` → Spark auto-splits skewed partitions at runtime → zero code change
2. 🥈 **Broadcast Join** → if the other side is small enough, skew disappears entirely (no shuffle = no skew)
3. 🥉 **Salting** → add random suffix to skewed keys → distribute across N buckets → join works in parallel → drop salt after

> 🎤 **Interview tip**: _"I always lead with AQE — it's free and requires zero code change on Databricks. Only reach for salting if AQE isn't enough or you're on Spark 2.x. Interviewer loves when you mention checking skew first via `.groupBy(key).count()` before jumping to solutions."_

***

### Q3. 📁 Small File Problem — 10,000 Files in S3

**Q: What's the issue? How to fix it?**

**A:** Big data engines are optimized for **128MB–1GB files**. With 10,000 small files:[^4_1]

**The problem 😤:**

- Each file = 1 task = scheduling overhead × 10,000
- Driver gets overwhelmed managing 10,000 task metadata entries
- S3 API call per file → massive I/O overhead[^4_2]
- Parallelism is actually broken — small tasks waste executor slots

**Fix it:**

- 🔧 **On read** → use `coalesce()` after reading to merge partitions in memory
- 🔧 **On write** → always `coalesce()` or `repartition()` before writing → target 128–256MB per file[^4_3]
- 🔧 **Upstream fix** → use **Delta Lake / Iceberg OPTIMIZE** command to compact small files periodically
- 🔧 **Databricks** → `OPTIMIZE` + `ZORDER` handles this automatically

> 🎤 **Interview tip**: _"The real answer is fixing it at the source — bad write patterns create small files. In production I'd set up a Delta OPTIMIZE job running on a schedule. For existing mess, coalesce after read is the quickest fix."_

***

### Q4. 💥 Executors Failing Randomly

**Q: What are the reasons? How to debug?**

**A:**

**Common reasons:**

- 💾 **OOM (Out of Memory)** — most common cause. Too much data per partition, too little memory per executor
- 🔄 **Shuffle spill** — shuffle data doesn't fit in memory, spills to disk → GC pressure → executor dies
- 🌐 **Network timeout** — slow S3 reads or shuffle fetch timeouts on large datasets
- 🏔️ **Data skew** — one partition overwhelms a single executor
- ⚙️ **Bad config** — too many cores per executor, executor memory too low

**How to debug 🔍:**

- **Spark UI → Executors tab** → look for GC time > 10% of task time → memory issue
- **Spark UI → Stages tab** → look for tasks with huge shuffle spill or input size outliers → skew
- **Driver logs** → search for `java.lang.OutOfMemoryError` or `FetchFailed`
- **Increase executor memory** → `spark.executor.memory` + `spark.executor.memoryOverhead`
- Reduce partition size → more partitions = less data per executor

> 🎤 **Interview tip**: _"First thing I check is GC time in Spark UI. If GC > 10%, it's a memory issue. Then I check input data per task — if one task has 10x more data than others, it's skew. 90% of executor failures I've seen in production are either OOM or data skew."_

***

### Q5. 💡 Memory Optimization

**Q: Job is running out of memory. What do you do?**

**Steps in order:**

1. 🔍 **Diagnose first** → Spark UI → check shuffle spill, GC time, input size per task
2. 📦 **Reduce partition size** → increase partition count so each task handles less data
3. 🗂️ **Push filters early** → filter before join/aggregation → less data in memory
4. 📡 **Use Broadcast join** → eliminates shuffle memory pressure for small tables
5. 💾 **Tune executor memory** → increase `spark.executor.memory`, add `spark.executor.memoryOverhead` for off-heap
6. 🧹 **Unpersist unused caches** → cached DataFrames eat memory even after you're done
7. 🔄 **Use Kryo serialization** → smaller serialized objects = less memory during shuffle
8. 📁 **Switch to Parquet** → columnar format reads fewer columns → less memory footprint

> 🎤 **Interview tip**: _"I always tune `spark.executor.memoryOverhead` first when I see container-killed errors — that's usually PySpark's Python worker overhead, not JVM heap. Interviewer loves when you distinguish JVM heap OOM vs container OOM."_

***

## 🔷 INTERNALS QUESTIONS


***

### Q6. ⚙️ Full Spark Execution Flow

**Code → DAG → Stages → Tasks → Execution**[^4_4][^4_5]

1. **You write code** → transformations like `filter()`, `join()`, `groupBy()` → nothing executes yet (lazy)
2. **Action is called** → `count()`, `write()`, `show()` → triggers execution
3. **Driver builds Logical Plan** → Catalyst Optimizer rewrites and optimizes it
4. **DAG is created** → a graph of all transformations and their dependencies
5. **DAG → Stages** → every shuffle boundary creates a new stage → narrow transformations stay in same stage
6. **Stages → Tasks** → 1 partition = 1 task → e.g., 200 partitions = 200 parallel tasks
7. **TaskScheduler assigns tasks** → sends to available executors via Cluster Manager
8. **Executors run tasks** → process their partition, write shuffle output if needed
9. **Results returned** → to Driver (for `collect()`) or written to S3 (for `write()`)

> 🎤 **Interview tip**: _"The key insight is that stage boundaries = shuffle boundaries. Interviewers love when you say: 'Every wide transformation creates a new stage because data must be re-partitioned across executors — that's the shuffle.'"_

***

### Q7. 🔗 What Happens Internally During a Join?

Spark picks a join strategy based on table sizes:[^4_6][^4_7]


| Strategy | When Used | What Happens |
| :-- | :-- | :-- |
| **Broadcast Hash Join** | One side < threshold (~10MB) | Small table broadcast to all executors, no shuffle |
| **Sort Merge Join** | Both tables large, keys sortable | Both sides shuffled by join key → sorted → merged. Most common for large-large joins |
| **Shuffle Hash Join** | One side moderately small | One side shuffled + hashed, other side probes the hash |

- SortMergeJoin = Spark's **default for large tables** — shuffles BOTH sides → expensive but robust[^4_7]
- You can verify strategy using `df.explain()` → look for `BroadcastHashJoin` or `SortMergeJoin`

> 🎤 **Interview tip**: _"When asked about join internals, always mention the join strategy selection order: Broadcast → Shuffle Hash → Sort Merge. Then say 'I always check explain() to confirm which strategy Spark chose — I've caught cases where AQE degraded to SortMerge when a broadcast should have been used.'"_

***

### Q8. 🔀 Wide Transformation vs Shuffle Dependency

- **Wide transformation** = the *operation* that requires data from multiple partitions (e.g., `groupBy`, `join`) — it's a **concept** describing WHAT the transformation does
- **Shuffle dependency** = the *mechanism* Spark creates internally when it needs to redistribute data — it's HOW Spark implements wide transformations
- Every wide transformation **creates a shuffle dependency** in the DAG, which becomes a **stage boundary**
- Narrow transformations create **narrow dependencies** — each output partition depends on exactly one input partition

> 🎤 **Interview tip**: _"Wide transformation = the 'what'. Shuffle dependency = the 'how'. They always come together. The practical implication: every wide transformation = new stage in Spark UI = potential performance bottleneck."_

***

### Q9. ⚡ What is Task Parallelism? How Do You Control It?

- **Task parallelism** = number of tasks running in parallel = determined by **number of partitions** + **executor cores**
- 1 partition = 1 task = 1 core used
- If you have 100 partitions and 50 cores → 50 tasks run in parallel → 2 waves

**Control knobs:**

- `spark.sql.shuffle.partitions` → controls parallelism AFTER a shuffle (default: 200 — often wrong!)
- `repartition(n)` → explicitly set parallelism before a heavy operation
- `spark.default.parallelism` → controls RDD parallelism
- AQE auto-adjusts shuffle partitions based on actual data size

> 🎤 **Interview tip**: _"Biggest mistake I see in junior code is leaving `spark.sql.shuffle.partitions=200` on a 2TB job (too few) or on a 100MB job (way too many). Rule of thumb: target 128MB per partition. On Databricks, AQE handles this automatically."_

***

### Q10. 🖥️ Role of Spark UI in Debugging

Spark UI is your **X-ray machine** for job performance. Key tabs:

- **Jobs tab** → see which action triggered which job, total duration
- **Stages tab** → see shuffle read/write size, spill to disk, task count per stage — 🔍 **most useful tab**
- **Tasks tab (within a stage)** → find skewed tasks (one task 10x slower than median = skew!)
- **Executors tab** → GC time, memory used, shuffle bytes — GC > 10% = memory issue ⚠️
- **Storage tab** → see what's cached and how much memory it's consuming
- **SQL tab** → visual DAG of your query plan, see if BroadcastHashJoin or SortMergeJoin is used

> 🎤 **Interview tip**: _"I tell interviewers: Spark UI is my first stop, not logs. The Tasks tab inside a stage will immediately show you if one task has 100x more data than others — that's your skew. The SQL tab shows the physical plan visually — you can see if Catalyst made a bad decision."_

***

## 🔷 SQL / LOGIC QUESTIONS


***

### Q11. 🏅 Employees Who NEVER Received Highest Salary in Their Dept

**Logic:** For each employee, check if their max salary ever equaled the department's max salary. If not → they never got the top.

**Approach:**

- Find `MAX(salary)` per department
- Find `MAX(salary)` per employee per department
- Employees where their personal max < department max → they never got the highest

```sql
-- Concept in SQL
WITH dept_max AS (
  SELECT dept_id, MAX(salary) AS dept_top
  FROM employees
  GROUP BY dept_id
),
emp_max AS (
  SELECT emp_id, dept_id, MAX(salary) AS emp_top
  FROM employees
  GROUP BY emp_id, dept_id
)
SELECT e.emp_id
FROM emp_max e
JOIN dept_max d ON e.dept_id = d.dept_id
WHERE e.emp_top < d.dept_top
```

> 🎤 **Interview tip**: _"Always clarify — 'highest salary ever in that dept historically, or just current?' This question tests whether you think about the problem domain, not just syntax."_

***

### Q12. 📈 Salary Trend: Increasing / Decreasing / Stable

**Logic:** For each employee, look at salary across dates. Compare consecutive records using LAG window function.

**Approach:**

- Use `LAG(salary)` over `PARTITION BY emp_id ORDER BY date`
- If all current > previous → Increasing
- If all current < previous → Decreasing
- If all equal → Stable
- Else → Fluctuating

```sql
-- Concept
WITH lagged AS (
  SELECT emp_id, salary, date,
         LAG(salary) OVER (PARTITION BY emp_id ORDER BY date) AS prev_salary
  FROM salaries
),
diffs AS (
  SELECT emp_id,
         SUM(CASE WHEN salary > prev_salary THEN 1 ELSE 0 END) AS ups,
         SUM(CASE WHEN salary < prev_salary THEN 1 ELSE 0 END) AS downs
  FROM lagged WHERE prev_salary IS NOT NULL
  GROUP BY emp_id
)
SELECT emp_id,
  CASE WHEN downs = 0 AND ups > 0 THEN 'Increasing'
       WHEN ups = 0 AND downs > 0 THEN 'Decreasing'
       WHEN ups = 0 AND downs = 0 THEN 'Stable'
       ELSE 'Fluctuating' END AS trend
FROM diffs
```

> 🎤 **Interview tip**: _"The trap is assuming only 3 states. Always add 'Fluctuating' as a 4th state — shows you think in real data terms, not ideal scenarios."_

***

### Q13. 🔍 Detect Data Skew Programmatically

**Logic:** Group by the join/partition key → count rows per key → find keys where count is significantly above average.

**Approach:**

- `groupBy(key).count()`
- Calculate mean and std deviation
- Flag keys where `count > mean + 2*stddev` (or simply top 5% by count)
- Output: key, row_count, % of total data

> 🎤 **Interview tip**: _"Don't just find the max — find keys that are statistical outliers. A key with 50K rows in a dataset with median 100 rows is skewed. I'd also check what % of total rows the top 10 keys hold — if top 5 keys = 70% of data, you have severe skew."_

***

### Q14. 🗂️ Custom Partitioning by Salary Range

**Logic:** Add a derived column `salary_bucket` (e.g., low/mid/high or 0–50K, 50K–100K, 100K+) → `partitionBy` that column on write.

**Approach:**

- Use `when()` to create salary band column
- Write with `.partitionBy("salary_band")`
- Result: S3 folders like `/salary_band=low/`, `/salary_band=high/`
- Downstream queries filtering on salary range → **partition pruning** → read only relevant folders

> 🎤 **Interview tip**: _"Partition columns should have LOW cardinality (3–5 buckets) not high (like exact salary). High cardinality partitioning = small files problem all over again. Always think about how this partition will be queried downstream."_

***

### Q15. ⚡ Optimized Read → Filter → Aggregate → Write

**Best practices (no fluff):**

**Read:**

- Use **Parquet or Delta**, never CSV in production
- Define schema explicitly → no inferSchema overhead
- Read only needed columns → projection pushdown

**Filter:**

- Push filters as early as possible → before joins, before aggregations
- Filter on **partition columns** if available → partition pruning → read less data

**Aggregate:**

- Tune `spark.sql.shuffle.partitions` to match data size (not default 200)
- Enable AQE → `spark.sql.adaptive.enabled=true`
- Cache intermediate result if used multiple times

**Write:**

- Use `.partitionBy("date", "region")` on business keys — low cardinality only
- Target 128–256MB files → `coalesce()` or `repartition()` before write
- Use **Delta format** → ACID, time travel, OPTIMIZE support

> 🎤 **Interview tip**: _"The golden rule I follow: filter early, shuffle late, write smart. If an interviewer asks 'anything else?' — mention AQE as a free win, and Delta over Parquet for production reliability."_
<span style="display:none">[^4_10][^4_11][^4_12][^4_13][^4_14][^4_15][^4_8][^4_9]</span>

<div align="center">⁂</div>

