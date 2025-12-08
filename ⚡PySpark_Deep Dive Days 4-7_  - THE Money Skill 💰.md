<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# ⚡ Days 4-7: PySpark Deep Dive - THE Money Skill 💰

Let me break down this critical learning path with detailed explanations and practical code! 🚀

## 📅 Day 4: Spark Architecture \& Core Concepts

### 🏗️ Spark Architecture

Spark uses a **master-worker architecture** where a Driver program coordinates work across multiple Executor nodes . The Driver runs your main program, creates SparkContext, and distributes tasks to Executors that run on worker nodes . Each Executor has multiple task slots and manages its own cache/memory .

**Key Components:**

- **Driver**: Converts your program into tasks and schedules them
- **Cluster Manager**: Allocates resources (YARN, Mesos, Kubernetes, Standalone)
- **Executors**: Run tasks and store data for your application
- **Tasks**: Smallest unit of work sent to one executor

```python
# 🎯 Creating SparkSession (entry point for PySpark)
from pyspark.sql import SparkSession

# Initialize Spark with configuration
spark = SparkSession.builder \
    .appName("PySpark_Deep_Dive") \
    .master("local[*]") \  # Use all available cores locally
    .config("spark.executor.memory", "4g") \  # Set executor memory
    .config("spark.driver.memory", "2g") \     # Set driver memory
    .getOrCreate()

# Check Spark version
print(f"Spark Version: {spark.version}")
```


### 🆚 RDDs vs DataFrames

**RDDs (Resilient Distributed Datasets)** are the fundamental data structure - immutable, distributed collections that can be processed in parallel . **DataFrames** are higher-level abstractions built on RDDs with named columns and schema, offering better optimization through Catalyst optimizer .


| Feature | RDD | DataFrame |
| :-- | :-- | :-- |
| Structure | Unstructured collection of objects | Structured with named columns \& schema |
| Optimization | No optimization, manual tuning needed | Catalyst optimizer + Tungsten execution |
| Performance | Slower (no optimizations) | 2-5x faster than RDDs |
| API | Low-level transformations | High-level SQL-like operations |
| Type Safety | Compile-time type safety | Runtime schema validation |

```python
# 📊 RDD Example (Old Way - rarely used now)
rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5])
squared_rdd = rdd.map(lambda x: x ** 2)  # Transformation
result = squared_rdd.collect()  # Action
print(f"RDD Result: {result}")

# ✨ DataFrame Example (Modern Way - USE THIS!)
df = spark.createDataFrame([(1,), (2,), (3,), (4,), (5,)], ["number"])
squared_df = df.select((df.number ** 2).alias("squared"))
squared_df.show()
```


### 🦥 Lazy Evaluation

Spark uses **lazy evaluation** - transformations are not executed immediately but recorded as a lineage . Execution only happens when an **action** is called . This allows Spark to optimize the entire execution plan before running anything .

### ⚙️ Transformations vs Actions

**Transformations** create new DataFrames from existing ones (lazy) . **Actions** trigger computation and return results to the driver or write to storage .

```python
# 🔄 TRANSFORMATIONS (Lazy - nothing executes yet)
df = spark.read.csv("data.csv", header=True)  # Lazy
filtered_df = df.filter(df.age > 25)          # Lazy - just records plan
grouped_df = filtered_df.groupBy("city")      # Lazy - still no execution

# ⚡ ACTIONS (Eager - triggers execution of all above)
count = filtered_df.count()        # Action - executes now!
filtered_df.show(10)               # Action - displays data
filtered_df.collect()              # Action - brings all data to driver (careful!)
filtered_df.write.parquet("out/")  # Action - writes to disk

# 🔍 View execution plan (before action)
filtered_df.explain(True)  # Shows logical and physical plans
```


***

## 📅 Day 5: PySpark DataFrames - Core Operations

### 🎯 Select, Filter, GroupBy, Agg

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# 📥 Create sample DataFrame
data = [
    ("Alice", "Sales", 5000, "NY"),
    ("Bob", "IT", 6000, "CA"),
    ("Charlie", "Sales", 5500, "NY"),
    ("David", "IT", 7000, "CA"),
    ("Eve", "HR", 4500, "TX")
]
df = spark.createDataFrame(data, ["name", "dept", "salary", "city"])

# 🔍 SELECT - Choose specific columns
selected = df.select("name", "salary")  # Select columns
selected_expr = df.select(
    df.name,                                    # Column reference
    (df.salary * 1.1).alias("salary_with_bonus"),  # Expression with alias
    F.upper(df.dept).alias("dept_upper")       # Using functions
)
selected_expr.show()

# 🔎 FILTER/WHERE - Filter rows based on conditions
filtered = df.filter(df.salary > 5000)  # Method 1
filtered = df.where(df.salary > 5000)   # Method 2 (same as filter)
filtered = df.filter((df.salary > 5000) & (df.dept == "IT"))  # Multiple conditions
filtered.show()

# 📊 GROUPBY + AGG - Aggregations
grouped = df.groupBy("dept").agg(
    F.count("*").alias("employee_count"),          # Count employees
    F.avg("salary").alias("avg_salary"),           # Average salary
    F.max("salary").alias("max_salary"),           # Maximum salary
    F.min("salary").alias("min_salary"),           # Minimum salary
    F.sum("salary").alias("total_salary")          # Total salary
)
grouped.show()

# 🎯 Multiple GroupBy columns
multi_group = df.groupBy("dept", "city").agg(
    F.count("*").alias("count"),
    F.avg("salary").alias("avg_sal")
)
multi_group.show()
```


### 🔗 Joins (Broadcast \& Shuffle)

**Joins** combine DataFrames based on keys . **Broadcast joins** replicate small DataFrames to all executors (fast, no shuffle) . **Shuffle joins** redistribute data across network (expensive for large datasets) .

```python
# 📊 Create two DataFrames for joining
employees = spark.createDataFrame([
    (1, "Alice", 101),
    (2, "Bob", 102),
    (3, "Charlie", 101),
    (4, "David", 103)
], ["emp_id", "name", "dept_id"])

departments = spark.createDataFrame([
    (101, "Sales"),
    (102, "IT"),
    (103, "HR")
], ["dept_id", "dept_name"])

# 🔗 INNER JOIN (default)
inner_join = employees.join(departments, "dept_id", "inner")
inner_join.show()

# 🔗 LEFT JOIN (keep all from left, nulls for non-matches)
left_join = employees.join(departments, "dept_id", "left")

# 🔗 RIGHT JOIN (keep all from right)
right_join = employees.join(departments, "dept_id", "right")

# 🔗 FULL OUTER JOIN (keep all from both)
full_join = employees.join(departments, "dept_id", "outer")

# 📡 BROADCAST JOIN (for small DataFrames < 10MB)
# Forces Spark to broadcast small DataFrame to all executors
from pyspark.sql.functions import broadcast
broadcast_join = employees.join(
    broadcast(departments),  # Explicitly broadcast small table
    "dept_id"
)
broadcast_join.show()

# 🔍 Check if broadcast happened
broadcast_join.explain()  # Look for "BroadcastHashJoin" in plan
```


### 🪟 Window Functions

**Window functions** perform calculations across rows related to the current row without grouping . They enable ranking, running totals, moving averages, and lag/lead operations .

```python
from pyspark.sql.window import Window
from pyspark.sql import functions as F

# 📊 Sample sales data
sales_data = [
    ("Alice", "Sales", "2024-01", 1000),
    ("Alice", "Sales", "2024-02", 1500),
    ("Alice", "Sales", "2024-03", 1200),
    ("Bob", "IT", "2024-01", 2000),
    ("Bob", "IT", "2024-02", 2200),
    ("Charlie", "Sales", "2024-01", 900)
]
sales_df = spark.createDataFrame(sales_data, ["name", "dept", "month", "amount"])

# 🪟 Define window specifications
# Partition by dept, order by amount
window_spec = Window.partitionBy("dept").orderBy("amount")

# Window with no partition (entire dataset)
window_all = Window.orderBy("amount")

# Window with row range (moving average)
window_range = Window.partitionBy("name").orderBy("month").rowsBetween(-1, 1)

# 📈 RANKING Functions
ranked_df = sales_df.withColumn(
    "row_number", F.row_number().over(window_spec)  # Sequential: 1,2,3,4...
).withColumn(
    "rank", F.rank().over(window_spec)              # Gaps for ties: 1,2,2,4...
).withColumn(
    "dense_rank", F.dense_rank().over(window_spec)  # No gaps: 1,2,2,3...
)
ranked_df.show()

# 📊 AGGREGATE Window Functions
agg_window_df = sales_df.withColumn(
    "running_total", F.sum("amount").over(window_spec)  # Cumulative sum
).withColumn(
    "avg_in_dept", F.avg("amount").over(Window.partitionBy("dept"))  # Avg per dept
).withColumn(
    "max_in_dept", F.max("amount").over(Window.partitionBy("dept"))  # Max per dept
)
agg_window_df.show()

# ⏮️ LAG/LEAD Functions (previous/next row values)
lag_lead_df = sales_df.withColumn(
    "prev_month_amount", F.lag("amount", 1).over(Window.partitionBy("name").orderBy("month"))
).withColumn(
    "next_month_amount", F.lead("amount", 1).over(Window.partitionBy("name").orderBy("month"))
).withColumn(
    "month_over_month_change", 
    F.col("amount") - F.lag("amount", 1).over(Window.partitionBy("name").orderBy("month"))
)
lag_lead_df.show()
```


***

## 📅 Day 6: Data Ingestion \& Schema Management

### 📂 Reading Different File Formats

```python
# 📄 CSV - Most common but slowest
csv_df = spark.read.csv(
    "data/input.csv",
    header=True,           # First row is header
    inferSchema=True,      # Auto-detect data types (slow!)
    sep=",",               # Delimiter
    quote='"',             # Quote character
    escape="\\",           # Escape character
    nullValue="NA",        # What represents null
    dateFormat="yyyy-MM-dd"  # Date format
)

# 📋 JSON - Structured, human-readable
json_df = spark.read.json(
    "data/input.json",
    multiLine=True,        # For pretty-printed JSON
    mode="PERMISSIVE"      # How to handle corrupt records
)

# 📦 PARQUET - Columnar, compressed, FAST! (BEST for Big Data)
parquet_df = spark.read.parquet("data/input.parquet")

# 🗂️ Read multiple files/patterns
multi_df = spark.read.parquet("data/year=2024/month=*/day=*/*.parquet")

# 📊 Read with options chaining
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("mode", "DROPMALFORMED") \
    .csv("data/*.csv")
```


### 🏗️ Schema Inference vs Explicit Schemas

**Schema inference** automatically detects data types by scanning data (slow for large files) . **Explicit schemas** define structure upfront (faster, more reliable, recommended for production) .

```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

# ❌ Schema Inference (SLOW - scans data twice)
inferred_df = spark.read.csv("data.csv", header=True, inferSchema=True)
inferred_df.printSchema()

# ✅ Explicit Schema (FAST - recommended for production!)
explicit_schema = StructType([
    StructField("employee_id", IntegerType(), nullable=False),  # Not null
    StructField("name", StringType(), nullable=False),
    StructField("department", StringType(), nullable=True),     # Can be null
    StructField("salary", DoubleType(), nullable=True),
    StructField("hire_date", DateType(), nullable=True)
])

explicit_df = spark.read.csv(
    "data.csv",
    header=True,
    schema=explicit_schema  # Apply explicit schema
)
explicit_df.printSchema()

# 🔍 Benefits of Explicit Schema:
# 1. Faster loading (no data scanning)
# 2. Data validation (type enforcement)
# 3. Predictable behavior
# 4. Better error handling
```


### 🛠️ Handling Null Values

```python
from pyspark.sql import functions as F

# 🎯 Sample data with nulls
data_with_nulls = [
    (1, "Alice", 5000, "NY"),
    (2, "Bob", None, "CA"),      # Null salary
    (3, None, 5500, "TX"),        # Null name
    (4, "David", 6000, None),     # Null city
    (5, None, None, None)         # All nulls
]
df_nulls = spark.createDataFrame(data_with_nulls, ["id", "name", "salary", "city"])

# 🔍 DETECT Nulls
df_nulls.filter(F.col("salary").isNull()).show()      # Find null salaries
df_nulls.filter(F.col("name").isNotNull()).show()     # Find non-null names

# 🗑️ DROP Nulls
dropped_any = df_nulls.dropna()             # Drop rows with ANY null
dropped_all = df_nulls.dropna(how="all")    # Drop rows with ALL nulls
dropped_cols = df_nulls.dropna(subset=["name", "salary"])  # Drop if these cols null

# 🔧 FILL Nulls
filled_all = df_nulls.fillna(0)             # Fill all nulls with 0
filled_specific = df_nulls.fillna({
    "name": "Unknown",                      # Fill null names
    "salary": 4000,                         # Fill null salaries
    "city": "Not Specified"                 # Fill null cities
})
filled_specific.show()

# 🔄 REPLACE with column expressions
replaced = df_nulls.withColumn(
    "salary",
    F.when(F.col("salary").isNull(), F.lit(4500))  # If null, use 4500
     .otherwise(F.col("salary"))                    # Otherwise keep original
)

# 📊 COALESCE - Return first non-null value
df_nulls.withColumn(
    "salary_clean",
    F.coalesce(F.col("salary"), F.lit(4000))  # Use salary or 4000 if null
).show()
```


***

## 📅 Day 7: Advanced Techniques

### 🔧 UDFs (User Defined Functions)

**UDFs** allow custom Python functions to process DataFrame columns . However, they're slower than built-in functions because data must be serialized to Python . Use built-in functions whenever possible .

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType

# 🐍 Define Python function
def categorize_salary(salary):
    """Categorize salary into bands"""
    if salary is None:
        return "Unknown"
    elif salary < 5000:
        return "Low"
    elif salary < 7000:
        return "Medium"
    else:
        return "High"

# 📦 Register as UDF
categorize_salary_udf = udf(categorize_salary, StringType())

# ✨ Apply UDF to DataFrame
df_with_category = df.withColumn(
    "salary_category",
    categorize_salary_udf(F.col("salary"))
)
df_with_category.show()

# 🎯 UDF with multiple inputs
def calculate_bonus(salary, dept):
    """Calculate bonus based on dept"""
    if salary is None:
        return 0
    multiplier = 0.2 if dept == "Sales" else 0.1
    return int(salary * multiplier)

calculate_bonus_udf = udf(calculate_bonus, IntegerType())

df_with_bonus = df.withColumn(
    "bonus",
    calculate_bonus_udf(F.col("salary"), F.col("dept"))
)
df_with_bonus.show()

# ⚡ BETTER APPROACH: Use built-in functions when possible (faster!)
df_builtin = df.withColumn(
    "salary_category",
    F.when(F.col("salary") < 5000, "Low")
     .when(F.col("salary") < 7000, "Medium")
     .otherwise("High")
)
```


### ✅ Data Quality Checks

```python
from pyspark.sql import functions as F

# 📊 Sample data for quality checks
quality_df = spark.createDataFrame([
    (1, "Alice", 25, "alice@email.com"),
    (2, "Bob", -5, "bob@invalid"),      # Negative age, invalid email
    (3, "Charlie", 150, None),          # Unrealistic age, null email
    (4, "David", 30, "david@email.com"),
    (1, "Alice", 25, "alice@email.com") # Duplicate
], ["id", "name", "age", "email"])

# 🔍 CHECK 1: Null values count
null_counts = quality_df.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c)
    for c in quality_df.columns
])
print("=== Null Counts ===")
null_counts.show()

# 🔍 CHECK 2: Duplicate detection
print(f"Total rows: {quality_df.count()}")
print(f"Distinct rows: {quality_df.distinct().count()}")
print(f"Duplicates: {quality_df.count() - quality_df.distinct().count()}")

# Find duplicate rows
duplicates = quality_df.groupBy(quality_df.columns).count().filter("count > 1")
duplicates.show()

# 🔍 CHECK 3: Value range validation
invalid_ages = quality_df.filter(
    (F.col("age") < 0) | (F.col("age") > 120)
)
print("=== Invalid Ages ===")
invalid_ages.show()

# 🔍 CHECK 4: Pattern validation (email)
invalid_emails = quality_df.filter(
    ~F.col("email").rlike("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$")
)
print("=== Invalid Emails ===")
invalid_emails.show()

# 🔧 CLEANSE: Remove duplicates and invalid records
clean_df = quality_df \
    .dropDuplicates() \
    .filter((F.col("age") >= 0) & (F.col("age") <= 120)) \
    .filter(F.col("email").isNotNull()) \
    .filter(F.col("email").rlike("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"))

print("=== Clean Data ===")
clean_df.show()
```


### 📁 Partitioning \& Bucketing Strategies

**Partitioning** splits data into subdirectories based on column values (great for filtering) . **Bucketing** distributes data into fixed number of files based on hash (great for joins) .

```python
# 📊 Sample sales data
sales_large = spark.createDataFrame([
    ("2024-01-15", "Electronics", "NY", 1000),
    ("2024-01-20", "Clothing", "CA", 500),
    ("2024-02-10", "Electronics", "TX", 1500),
    ("2024-02-15", "Clothing", "NY", 700),
    ("2024-03-05", "Electronics", "CA", 2000)
], ["date", "category", "state", "amount"])

# 📂 PARTITIONING - Creates directory structure
# Use for: Time-based queries, category filtering
sales_large.write \
    .mode("overwrite") \
    .partitionBy("category", "state") \
    .parquet("output/sales_partitioned")

# Result directory structure:
# output/sales_partitioned/
#   category=Electronics/state=NY/part-00000.parquet
#   category=Electronics/state=CA/part-00001.parquet
#   category=Clothing/state=NY/part-00002.parquet
#   ...

# 📖 Read partitioned data (prune partitions!)
# Only reads relevant partitions = MUCH FASTER!
filtered = spark.read.parquet("output/sales_partitioned") \
    .filter((F.col("category") == "Electronics") & (F.col("state") == "CA"))

# 🪣 BUCKETING - Fixed number of files based on hash
# Use for: Optimizing joins on bucketed column
sales_large.write \
    .mode("overwrite") \
    .bucketBy(4, "category") \
    .sortBy("amount") \
    .saveAsTable("sales_bucketed")  # Must use saveAsTable for bucketing

# 🎯 BEST PRACTICES:
# 1. Partition on low-cardinality columns (date, category, region)
# 2. Avoid over-partitioning (don't create millions of small files)
# 3. Target partition size: 128MB - 1GB
# 4. Use bucketing for frequently joined columns
# 5. Number of buckets should be power of 2 (2, 4, 8, 16...)

# ⚡ Repartition vs Coalesce
# Repartition: Full shuffle, can increase/decrease partitions
repartitioned = sales_large.repartition(10)  # 10 partitions

# Coalesce: No shuffle, can only decrease partitions (faster!)
coalesced = sales_large.coalesce(2)  # Reduce to 2 partitions
```


***

## 🎯 Mini Project: Complete ETL Pipeline

```python
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

# 🚀 Initialize Spark
spark = SparkSession.builder \
    .appName("ETL_Pipeline_Project") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# 📋 STEP 1: Define Explicit Schema
schema = StructType([
    StructField("customer_id", IntegerType(), nullable=False),
    StructField("customer_name", StringType(), nullable=True),
    StructField("transaction_date", StringType(), nullable=True),  # Will convert to date
    StructField("product_category", StringType(), nullable=True),
    StructField("amount", DoubleType(), nullable=True),
    StructField("quantity", IntegerType(), nullable=True),
    StructField("region", StringType(), nullable=True)
])

# 📥 STEP 2: Read Messy CSV Data
raw_df = spark.read \
    .option("header", "true") \
    .option("mode", "PERMISSIVE") \
    .option("nullValue", "NA") \
    .schema(schema) \
    .csv("input/messy_sales_data.csv")

print("=== RAW DATA ===")
raw_df.show(10, truncate=False)
print(f"Raw record count: {raw_df.count()}")

# 🧹 STEP 3: Data Cleansing

# 3.1 - Remove rows with null customer_id (critical field)
cleaned_df = raw_df.filter(F.col("customer_id").isNotNull())

# 3.2 - Handle null values in other columns
cleaned_df = cleaned_df.fillna({
    "customer_name": "Unknown",
    "product_category": "Uncategorized",
    "region": "Unknown",
    "quantity": 0,
    "amount": 0.0
})

# 3.3 - Convert date string to proper DateType
cleaned_df = cleaned_df.withColumn(
    "transaction_date",
    F.to_date(F.col("transaction_date"), "yyyy-MM-dd")
)

# 3.4 - Remove duplicates (based on all columns)
cleaned_df = cleaned_df.dropDuplicates()

# 3.5 - Data validation: Remove invalid records
cleaned_df = cleaned_df.filter(
    (F.col("amount") >= 0) &          # Non-negative amounts
    (F.col("quantity") >= 0) &        # Non-negative quantities
    (F.col("transaction_date").isNotNull())  # Valid dates
)

# 3.6 - Type conversions & derived columns
cleaned_df = cleaned_df.withColumn(
    "unit_price",
    F.when(F.col("quantity") > 0, F.col("amount") / F.col("quantity"))
     .otherwise(0.0)
).withColumn(
    "year", F.year(F.col("transaction_date"))
).withColumn(
    "month", F.month(F.col("transaction_date"))
).withColumn(
    "quarter", F.quarter(F.col("transaction_date"))
)

print("=== CLEANED DATA ===")
cleaned_df.show(10, truncate=False)
print(f"Cleaned record count: {cleaned_df.count()}")

# 📊 STEP 4: Perform Aggregations

# 4.1 - Aggregation by region and category
agg_by_region_category = cleaned_df.groupBy("region", "product_category") \
    .agg(
        F.count("*").alias("transaction_count"),
        F.sum("amount").alias("total_revenue"),
        F.avg("amount").alias("avg_transaction_amount"),
        F.sum("quantity").alias("total_quantity_sold"),
        F.countDistinct("customer_id").alias("unique_customers")
    ) \
    .orderBy(F.desc("total_revenue"))

print("=== AGGREGATION: Region & Category ===")
agg_by_region_category.show()

# 4.2 - Monthly trends
monthly_trends = cleaned_df.groupBy("year", "month") \
    .agg(
        F.sum("amount").alias("monthly_revenue"),
        F.count("*").alias("monthly_transactions"),
        F.avg("amount").alias("avg_transaction")
    ) \
    .orderBy("year", "month")

print("=== MONTHLY TRENDS ===")
monthly_trends.show()

# 4.3 - Top customers
top_customers = cleaned_df.groupBy("customer_id", "customer_name") \
    .agg(
        F.sum("amount").alias("total_spent"),
        F.count("*").alias("transaction_count")
    ) \
    .orderBy(F.desc("total_spent")) \
    .limit(10)

print("=== TOP 10 CUSTOMERS ===")
top_customers.show()

# 💾 STEP 5: Write to Parquet with Partitioning

# 5.1 - Write main cleaned data (partitioned by year and month)
cleaned_df.write \
    .mode("overwrite") \
    .partitionBy("year", "month") \
    .parquet("output/cleaned_transactions")

print("✅ Cleaned data written to: output/cleaned_transactions")

# 5.2 - Write aggregated data (no partitioning needed for small data)
agg_by_region_category.write \
    .mode("overwrite") \
    .parquet("output/aggregated_by_region_category")

monthly_trends.write \
    .mode("overwrite") \
    .parquet("output/monthly_trends")

top_customers.write \
    .mode("overwrite") \
    .parquet("output/top_customers")

print("✅ Aggregated data written successfully!")

# 📈 STEP 6: Data Quality Report
print("\n" + "="*50)
print("📊 DATA QUALITY REPORT")
print("="*50)

# Count nulls in critical fields
null_report = cleaned_df.select([
    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"null_{c}")
    for c in cleaned_df.columns
])
print("\n=== Null Value Counts ===")
null_report.show()

# Statistical summary
print("\n=== Statistical Summary ===")
cleaned_df.select("amount", "quantity", "unit_price").summary().show()

# Record count by region
print("\n=== Records by Region ===")
cleaned_df.groupBy("region").count().orderBy(F.desc("count")).show()

# 🎉 Pipeline Complete!
print("\n" + "="*50)
print("✅ ETL PIPELINE COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"📥 Input: messy CSV data")
print(f"🧹 Cleaned: {cleaned_df.count()} valid records")
print(f"📊 Aggregations: 3 summary tables created")
print(f"💾 Output: Parquet files with partitioning")
print("="*50)

# Stop Spark session
spark.stop()
```


### 📝 Sample Input CSV (messy_sales_data.csv)

```csv
customer_id,customer_name,transaction_date,product_category,amount,quantity,region
1,Alice Johnson,2024-01-15,Electronics,1200.50,2,NY
2,Bob Smith,2024-01-16,Clothing,350.00,5,CA
1,Alice Johnson,2024-01-15,Electronics,1200.50,2,NY
3,NA,2024-01-17,NA,500.00,1,TX
4,David Lee,invalid_date,Electronics,800.00,2,CA
5,Eve Martinez,2024-02-01,Furniture,2500.00,1,NY
NA,Unknown,2024-02-05,Clothing,150.00,3,TX
6,Frank Wilson,2024-02-10,Electronics,-100.00,1,CA
7,Grace Chen,2024-03-01,Clothing,450.00,0,NY
8,Henry Brown,2024-03-15,Furniture,3200.00,2,TX
```


***

## 🎯 Key Takeaways

✅ **DataFrame API > RDDs** - Always use DataFrames for better performance
⚡ **Lazy Evaluation** - Transformations build execution plan, actions trigger it
🔗 **Broadcast small tables** - Avoid shuffles in joins
📋 **Explicit schemas** - Faster and more reliable than inference
🛠️ **Built-in functions > UDFs** - Native functions are much faster
📁 **Partition wisely** - Target 128MB-1GB per partition
✨ **Data quality first** - Always validate and cleanse before processing!

💰 **This is THE money skill** - Master these concepts and you'll be unstoppable in the data engineering world! 🚀

