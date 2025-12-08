<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 🎯 SQL Mastery: Days 1-3 Complete Learning Guide

## 📅 Day 1: Advanced SQL - Window Functions, CTEs \& Subqueries

### 🪟 Window Functions Overview

Window functions perform calculations across rows related to the current row without collapsing them into a single output like aggregate functions. They enable advanced analytics while preserving row-level detail.

### 🔢 ROW_NUMBER()

ROW_NUMBER() assigns a unique sequential integer to each row within a partition, starting from 1. This function is perfect for creating unique identifiers or removing duplicates.

```sql
-- Example: Assign employee rank by salary within each department
SELECT 
    employee_id,
    department,
    salary,
    -- ROW_NUMBER assigns unique numbers even for ties
    ROW_NUMBER() OVER (
        PARTITION BY department  -- Restart numbering for each department
        ORDER BY salary DESC     -- Highest salary gets number 1
    ) AS salary_rank
FROM employees;

-- Example: Remove duplicate records (keep only first occurrence)
WITH ranked_records AS (
    SELECT 
        customer_id,
        order_date,
        amount,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id  -- Group by customer
            ORDER BY order_date       -- Earliest order first
        ) AS rn
    FROM orders
)
SELECT customer_id, order_date, amount
FROM ranked_records
WHERE rn = 1;  -- Keep only the first order per customer
```


### 🏆 RANK() vs DENSE_RANK()

RANK() assigns ranks with gaps after ties, while DENSE_RANK() assigns consecutive ranks without gaps.

```sql
-- Example: Compare RANK and DENSE_RANK behavior
SELECT 
    student_name,
    score,
    -- RANK: If two students tie for 2nd, next rank is 4
    RANK() OVER (ORDER BY score DESC) AS rank_with_gaps,
    -- DENSE_RANK: If two students tie for 2nd, next rank is 3
    DENSE_RANK() OVER (ORDER BY score DESC) AS rank_no_gaps,
    -- ROW_NUMBER: Always unique (1,2,3,4,5...)
    ROW_NUMBER() OVER (ORDER BY score DESC) AS row_num
FROM students;

/* Sample Output:
student_name | score | rank_with_gaps | rank_no_gaps | row_num
Alice        | 95    | 1              | 1            | 1
Bob          | 90    | 2              | 2            | 2
Charlie      | 90    | 2              | 2            | 3
David        | 85    | 4              | 3            | 4
*/
```


### ⏮️⏭️ LAG() and LEAD()

LAG() accesses data from previous rows, while LEAD() accesses data from subsequent rows. These are essential for time-series analysis and comparisons.

```sql
-- Example: Calculate month-over-month revenue change
WITH monthly_revenue AS (
    SELECT 
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS revenue
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
)
SELECT 
    month,
    revenue,
    -- LAG gets previous month's revenue (default NULL if no previous row)
    LAG(revenue, 1) OVER (ORDER BY month) AS prev_month_revenue,
    -- Calculate percentage change
    ROUND(
        ((revenue - LAG(revenue, 1) OVER (ORDER BY month)) / 
         LAG(revenue, 1) OVER (ORDER BY month)) * 100, 2
    ) AS pct_change,
    -- LEAD gets next month's revenue
    LEAD(revenue, 1) OVER (ORDER BY month) AS next_month_revenue
FROM monthly_revenue;

-- Example: Compare current price with previous and next price
SELECT 
    product_id,
    price_date,
    price,
    LAG(price, 1, 0) OVER (
        PARTITION BY product_id 
        ORDER BY price_date
    ) AS previous_price,  -- 0 is default if no previous row
    LEAD(price, 1) OVER (
        PARTITION BY product_id 
        ORDER BY price_date
    ) AS next_price
FROM product_prices;
```


### 🔄 Common Table Expressions (CTEs)

CTEs create temporary named result sets that make complex queries more readable and maintainable.

```sql
-- Example: Multi-level CTE for sales analysis
WITH 
-- CTE 1: Calculate monthly sales per customer
monthly_sales AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS monthly_total,
        COUNT(order_id) AS order_count
    FROM orders
    WHERE order_date >= '2024-01-01'
    GROUP BY customer_id, DATE_TRUNC('month', order_date)
),
-- CTE 2: Calculate average monthly spending per customer
customer_avg AS (
    SELECT 
        customer_id,
        AVG(monthly_total) AS avg_monthly_spend,
        SUM(monthly_total) AS total_spend,
        COUNT(DISTINCT month) AS active_months
    FROM monthly_sales
    GROUP BY customer_id
),
-- CTE 3: Classify customers into segments
customer_segments AS (
    SELECT 
        customer_id,
        avg_monthly_spend,
        total_spend,
        active_months,
        CASE 
            WHEN avg_monthly_spend > 1000 THEN 'Premium'
            WHEN avg_monthly_spend > 500 THEN 'Gold'
            WHEN avg_monthly_spend > 100 THEN 'Silver'
            ELSE 'Bronze'
        END AS segment
    FROM customer_avg
)
-- Final query: Get customer details with segments
SELECT 
    c.customer_name,
    cs.segment,
    cs.avg_monthly_spend,
    cs.total_spend,
    cs.active_months,
    -- Calculate loyalty score
    ROUND(cs.active_months * cs.avg_monthly_spend / 100, 2) AS loyalty_score
FROM customer_segments cs
JOIN customers c ON cs.customer_id = c.customer_id
ORDER BY loyalty_score DESC;
```


### 🎯 Subqueries

Subqueries are nested queries used in SELECT, FROM, WHERE, or HAVING clauses.

```sql
-- Example: Correlated subquery - Find employees earning above department average
SELECT 
    e1.employee_id,
    e1.name,
    e1.department,
    e1.salary,
    -- Correlated subquery runs for each row
    (SELECT AVG(salary) 
     FROM employees e2 
     WHERE e2.department = e1.department) AS dept_avg_salary
FROM employees e1
WHERE salary > (
    SELECT AVG(salary) 
    FROM employees e2 
    WHERE e2.department = e1.department
);

-- Example: Subquery in FROM clause (derived table)
SELECT 
    dept_stats.department,
    dept_stats.employee_count,
    dept_stats.total_salary,
    dept_stats.avg_salary
FROM (
    -- This subquery creates a temporary result set
    SELECT 
        department,
        COUNT(*) AS employee_count,
        SUM(salary) AS total_salary,
        AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department
) AS dept_stats
WHERE dept_stats.avg_salary > 50000;

-- Example: Subquery with EXISTS - Find customers with orders in last 30 days
SELECT 
    c.customer_id,
    c.customer_name
FROM customers c
WHERE EXISTS (
    -- EXISTS returns true/false, doesn't return actual data
    SELECT 1 
    FROM orders o 
    WHERE o.customer_id = c.customer_id 
      AND o.order_date >= CURRENT_DATE - INTERVAL '30 days'
);
```


***

## ⚡ Day 2: Spark SQL - Syntax, Partitioning \& Optimization

💥 **Alright listen, champ.**
You want this for **Databricks**? I'll rewrite your code **exactly how it should be used inside a Databricks notebook**, plus **Spark SQL vs ANSI SQL differences**.
And I'll roast anything weak. 😈🔥

---

# ⚔️ **Spark SQL vs ANSI SQL — Real Differences (No BS)**

| Feature          | ANSI SQL                       | Spark SQL                                | Verdict                            |
| ---------------- | ------------------------------ | ---------------------------------------- | ---------------------------------- |
| Execution        | Runs on single machine         | **Distributed execution across cluster** | Spark destroys ANSI                |
| Joins            | Normal joins                   | **Broadcast joins, shuffle joins**       | Spark adds performance steroids 💉 |
| Window functions | Supported                      | **Same but distributed optimized**       | Faster on big data                 |
| Data formats     | Mostly relational DB           | **Parquet, ORC, Delta, JSON, CSV**       | Real Big Data                      |
| Partitioning     | Usually table partitions in DB | **File-level + shuffle partitioning**    | Essential for scaling              |
| Optimization     | Limited                        | **Catalyst + Tungsten optimizer**        | Smart AF 🤖                        |
| Hints            | Rare                           | **broadcast(), sort merge, shuffle**     | Manual performance control         |
| NULL handling    | ANSI strict                    | Spark looser & configurable              | Needs attention                    |

---

# 🧠 **Databricks Notebook Version of Your Code**

### (Already optimized & with comments, ready for Day 2 practice)

```python
# Databricks Notebook Cell 1: Create Spark Session (Databricks auto creates but good for config)
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SparkSQL_Day2") \
    .config("spark.sql.adaptive.enabled", "true") \  # Enables Adaptive Query Execution (AQE optimization)
    .getOrCreate()
```

```python
# Cell 2: Import required functions
from pyspark.sql.functions import col, sum, avg, window, lag, broadcast
from pyspark.sql.window import Window
```

```python
# Cell 3: Read dataset (use dbfs or upload file manually on Databricks)
orders_df = spark.read.format("delta").load("/mnt/data/orders_delta")

# Register as temporary view so Spark SQL can query it
orders_df.createOrReplaceTempView("orders")
```

```python
# Cell 4: Spark SQL Query Example
result = spark.sql("""
    SELECT 
        customer_id,
        date_trunc('month', order_date) AS month,
        SUM(amount) AS monthly_revenue,

        -- Window function
        LAG(SUM(amount)) OVER (
            PARTITION BY customer_id
            ORDER BY date_trunc('month', order_date)
        ) AS prev_month_revenue

    FROM orders
    WHERE YEAR(order_date) = 2024
    GROUP BY customer_id, date_trunc('month', order_date)
""")

result.show()
```

---

# 🛠️ **Broadcast Join Example (Databricks Special Training)**

```python
small_df = spark.read.csv("/mnt/data/customer_details.csv", header=True)
big_df = spark.read.parquet("/mnt/data/orders_big.parquet")

joined = big_df.join(broadcast(small_df), "customer_id")  # broadcast = avoid shuffle
joined.explain(True)  # show execution plan
```

---


# 🧩 Partitioning Strategies

🧩 **Partitioning Strategies in Spark — Deep but Clear**

**What is Partitioning?**
- Partitioning = How Spark divides data across different nodes/executors
- Enables parallel processing across cluster

**If your partitioning strategy is trash, Spark will:**
- Shuffle data like crazy 🌀
- Slow down joins, group-by, window functions 🐌
- Use huge memory and crash ❌

**If you partition smart:**
- Spark reads only required partitions 🏎️
- Joins become faster ⚡
- Less shuffle → less cost ☘️

Proper partitioning is critical for Spark SQL performance, as it determines data distribution across cluster nodes.

```python
# Example: Repartitioning for join optimization
from pyspark.sql.functions import year, month

# Read large datasets
orders = spark.read.parquet("s3://data/orders/")
customers = spark.read.parquet("s3://data/customers/")

# Strategy 1: Repartition by join key before join
# This ensures matching keys are on same partition
orders_partitioned = orders.repartition(200, "customer_id")
customers_partitioned = customers.repartition(200, "customer_id")

# Now join is more efficient (no shuffle needed)
result = orders_partitioned.join(
    customers_partitioned,
    on="customer_id",
    how="inner"
)

# Strategy 2: Partition by date for time-series queries
# Write data partitioned by year and month
orders.write \
    .partitionBy("year", "month") \
    .parquet("s3://output/orders_partitioned/")

# When reading, only scan relevant partitions
recent_orders = spark.read.parquet("s3://output/orders_partitioned/") \
    .filter("year = 2024 AND month >= 10")  # Only reads 3 months of data

# Strategy 3: Bucketing for repeated joins
# Pre-shuffle data once, reuse for multiple queries
orders.write \
    .bucketBy(50, "customer_id") \  # 50 buckets based on customer_id
    .sortBy("order_date") \          # Sort within buckets
    .saveAsTable("orders_bucketed")

customers.write \
    .bucketBy(50, "customer_id") \  # Same number of buckets & key
    .saveAsTable("customers_bucketed")

# Bucketed joins avoid shuffle completely
bucketed_join = spark.sql("""
    SELECT o.*, c.customer_name
    FROM orders_bucketed o
    JOIN customers_bucketed c ON o.customer_id = c.customer_id
""")
```


# 📡 Broadcast Joins

📡 **Broadcast Joins in Spark — REAL Meaning**

**What is Broadcast Join?**
- Broadcast join = Spark copies a small dataset to every executor node, so joins happen locally without shuffling the big dataset.

👉 **Instead of moving billions of rows across network (shuffle), Spark moves the tiny table everywhere.**

⚡ **WHY?**
- Because network shuffle is the biggest bottleneck in distributed computing.
- Broadcast join kills shuffle → massive performance boost.

🎯 **When to Use Broadcast Join**

Use broadcast join when:
- One table is very small (dimension / lookup table)
- Other table is super large (fact table)
- Join key is common

| Table         | Size Example  |
|---------------|---------------|
| Small products| 5–50 MB       |
| Large orders  | 50 GB – 10 TB |

Broadcast joins are optimal when one table is small enough to fit in memory on each executor.

```python
# Example: Broadcast join optimization
from pyspark.sql.functions import broadcast

# Large table: 100GB of orders
large_orders = spark.read.parquet("s3://data/orders/")

# Small table: 10MB dimension table
small_products = spark.read.parquet("s3://data/products/")

# Method 1: Automatic broadcast (configure threshold)
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 10485760)  # 10MB

# Method 2: Explicit broadcast hint (recommended for clarity)
result = large_orders.join(
    broadcast(small_products),  # Force broadcast of small table
    on="product_id",
    how="inner"
)

# Spark SQL syntax for broadcast hint
result_sql = spark.sql("""
    SELECT /*+ BROADCAST(p) */ 
        o.order_id,
        o.amount,
        p.product_name,
        p.category
    FROM orders o
    JOIN products p ON o.product_id = p.product_id
""")

# Example: Multiple broadcast joins
countries = spark.read.parquet("s3://data/countries/")  # Small
regions = spark.read.parquet("s3://data/regions/")      # Small

result = large_orders \
    .join(broadcast(small_products), "product_id") \
    .join(broadcast(countries), "country_id") \
    .join(broadcast(regions), "region_id")
```


### ⚙️ Spark SQL Optimizations

Key optimization techniques for Spark SQL queries.

```python
# Optimization 1: Predicate pushdown - Filter early
# BAD: Filter after reading all data
df_bad = spark.read.parquet("s3://data/orders/") \
    .filter(col("order_date") >= "2024-01-01")

# GOOD: Push filter to data source (reads less data)
df_good = spark.read \
    .option("pushdown", "true") \
    .parquet("s3://data/orders/") \
    .where("order_date >= '2024-01-01'")  # Filter pushed to Parquet reader

# Optimization 2: Column pruning - Select only needed columns
# BAD: Read all columns then select
df_bad = spark.read.parquet("s3://data/orders/").select("order_id", "amount")

# GOOD: Specify columns early (Parquet reads only these columns)
df_good = spark.read.parquet("s3://data/orders/") \
    .select("order_id", "amount")

# Optimization 3: Adjust shuffle partitions based on data size
# Default 200 is too many for small datasets, too few for large
spark.conf.set("spark.sql.shuffle.partitions", "1000")  # For TB-scale data

# Optimization 4: Cache frequently accessed data
dim_products = spark.read.parquet("s3://data/products/").cache()
# Now multiple queries on dim_products won't re-read from storage

# Optimization 5: Adaptive Query Execution (AQE)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

# AQE automatically:
# - Combines small partitions after shuffle
# - Handles data skew in joins
# - Converts sort-merge to broadcast join if one side becomes small

# Example: Complex optimized query
optimized_query = spark.sql("""
    SELECT 
        c.customer_segment,
        p.category,
        SUM(o.amount) as total_revenue,
        COUNT(DISTINCT o.order_id) as order_count,
        AVG(o.amount) as avg_order_value
    FROM orders o
    JOIN /*+ BROADCAST(p) */ products p ON o.product_id = p.product_id
    JOIN /*+ BROADCAST(c) */ customers c ON o.customer_id = c.customer_id
    WHERE o.order_date >= '2024-01-01'
      AND p.category IN ('Electronics', 'Clothing')
    GROUP BY c.customer_segment, p.category
    HAVING SUM(o.amount) > 10000
""")
```


***

## 🚀 Day 3: Query Optimization, Indexing \& Execution Plans

### 📊 Understanding Execution Plans

Execution plans show how the database engine processes your query, revealing performance bottlenecks.

```sql
-- Example: Analyze execution plan in PostgreSQL
EXPLAIN ANALYZE
SELECT 
    c.customer_name,
    SUM(o.amount) as total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date >= '2024-01-01'
GROUP BY c.customer_name
HAVING SUM(o.amount) > 1000;

/* Execution Plan Output (example):
HashAggregate  (cost=1234.56..1456.78 rows=100)
  Group Key: c.customer_name
  Filter: (sum(o.amount) > 1000)
  -> Hash Join  (cost=234.56..789.12 rows=5000)
        Hash Cond: (o.customer_id = c.customer_id)
        -> Seq Scan on orders o  (cost=0.00..456.78 rows=10000)  ⚠️ SLOW
              Filter: (order_date >= '2024-01-01')
        -> Hash  (cost=123.45..123.45 rows=1000)
              -> Seq Scan on customers c  (cost=0.00..123.45 rows=1000)
*/

-- Spark SQL execution plan
-- spark.sql("YOUR_QUERY").explain(mode="extended")
```


### 🔍 Indexing Strategies

Indexes dramatically improve query performance by avoiding full table scans.

```sql
-- Strategy 1: Single-column index for frequent WHERE filters
CREATE INDEX idx_orders_date ON orders(order_date);

-- Now this query uses index instead of scanning all rows
SELECT * FROM orders WHERE order_date = '2024-01-15';

-- Strategy 2: Composite index for multi-column filters
CREATE INDEX idx_orders_customer_date 
ON orders(customer_id, order_date);

-- Optimizes queries filtering on both columns
SELECT * FROM orders 
WHERE customer_id = 123 AND order_date >= '2024-01-01';

-- Strategy 3: Covering index (includes all needed columns)
CREATE INDEX idx_orders_covering 
ON orders(customer_id, order_date) 
INCLUDE (amount, product_id);  -- Include additional columns

-- Query can be satisfied entirely from index (no table access)
SELECT customer_id, order_date, amount 
FROM orders 
WHERE customer_id = 123;

-- Strategy 4: Partial index for subset of data
CREATE INDEX idx_orders_recent 
ON orders(customer_id, amount)
WHERE order_date >= '2024-01-01';  -- Only index recent orders

-- Smaller index, faster for queries on recent data
SELECT SUM(amount) FROM orders 
WHERE customer_id = 123 AND order_date >= '2024-01-01';

-- Strategy 5: Index on JOIN columns
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_customers_id ON customers(customer_id);

-- Optimizes join performance
SELECT c.customer_name, o.amount
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id;

-- ⚠️ Avoid over-indexing
-- Too many indexes slow down INSERT/UPDATE/DELETE
-- Monitor index usage:
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read
FROM pg_stat_user_indexes
WHERE idx_scan = 0  -- Unused indexes
ORDER BY schemaname, tablename;

-- Drop unused indexes
DROP INDEX idx_unused_index;
```


### 🎯 Query Optimization Techniques

```sql
-- Technique 1: Rewrite subqueries as JOINs
-- BAD: Correlated subquery (runs for each row)
SELECT 
    o.order_id,
    o.amount,
    (SELECT customer_name FROM customers c WHERE c.customer_id = o.customer_id)
FROM orders o;

-- GOOD: JOIN (runs once)
SELECT 
    o.order_id,
    o.amount,
    c.customer_name
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id;

-- Technique 2: Use EXISTS instead of IN for large datasets
-- BAD: IN with subquery (materializes entire list)
SELECT * FROM customers
WHERE customer_id IN (
    SELECT customer_id FROM orders WHERE amount > 1000
);

-- GOOD: EXISTS (stops at first match)
SELECT * FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o 
    WHERE o.customer_id = c.customer_id AND amount > 1000
);

-- Technique 3: Avoid SELECT * (column pruning)
-- BAD: Reads all columns
SELECT * FROM orders WHERE order_date = '2024-01-01';

-- GOOD: Read only needed columns
SELECT order_id, customer_id, amount 
FROM orders WHERE order_date = '2024-01-01';

-- Technique 4: Use UNION ALL instead of UNION when duplicates OK
-- BAD: UNION removes duplicates (expensive sort/distinct)
SELECT customer_id FROM orders_2023
UNION
SELECT customer_id FROM orders_2024;

-- GOOD: UNION ALL keeps duplicates (no sort needed)
SELECT customer_id FROM orders_2023
UNION ALL
SELECT customer_id FROM orders_2024;

-- Technique 5: Limit result sets early
-- BAD: Sort all rows then limit
SELECT * FROM orders ORDER BY order_date DESC LIMIT 10;

-- GOOD: Use index on order_date for top-N query
CREATE INDEX idx_orders_date_desc ON orders(order_date DESC);
SELECT * FROM orders ORDER BY order_date DESC LIMIT 10;
```


***

## 🎯 Mini Project: 5 Complex Analytical Queries

### Query 1: Sales Analysis with Running Totals 💰

```sql
-- ANSI SQL Version
WITH daily_sales AS (
    SELECT 
        order_date,
        product_category,
        SUM(amount) as daily_revenue,
        COUNT(DISTINCT order_id) as order_count
    FROM orders
    WHERE order_date >= '2024-01-01'
    GROUP BY order_date, product_category
)
SELECT 
    order_date,
    product_category,
    daily_revenue,
    order_count,
    -- Running total of revenue
    SUM(daily_revenue) OVER (
        PARTITION BY product_category 
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_revenue,
    -- 7-day moving average
    AVG(daily_revenue) OVER (
        PARTITION BY product_category 
        ORDER BY order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS ma_7day,
    -- Rank days by revenue within each category
    RANK() OVER (
        PARTITION BY product_category 
        ORDER BY daily_revenue DESC
    ) AS revenue_rank
FROM daily_sales
ORDER BY product_category, order_date;
```

```python
# Spark SQL Version
from pyspark.sql import Window
from pyspark.sql.functions import sum, count, avg, rank, col

# Define window specifications
category_date_window = Window.partitionBy("product_category").orderBy("order_date")
category_date_range_window = Window.partitionBy("product_category") \
    .orderBy("order_date") \
    .rowsBetween(-6, 0)  # 7-day window
category_revenue_window = Window.partitionBy("product_category") \
    .orderBy(col("daily_revenue").desc())

# Calculate daily sales
daily_sales = spark.sql("""
    SELECT 
        order_date,
        product_category,
        SUM(amount) as daily_revenue,
        COUNT(DISTINCT order_id) as order_count
    FROM orders
    WHERE order_date >= '2024-01-01'
    GROUP BY order_date, product_category
""")

# Apply window functions
result = daily_sales.select(
    "order_date",
    "product_category",
    "daily_revenue",
    "order_count",
    sum("daily_revenue").over(category_date_window).alias("cumulative_revenue"),
    avg("daily_revenue").over(category_date_range_window).alias("ma_7day"),
    rank().over(category_revenue_window).alias("revenue_rank")
).orderBy("product_category", "order_date")

result.show()
```


### Query 2: User Retention Cohort Analysis 👥

```sql
-- ANSI SQL Version
WITH first_purchase AS (
    -- Get each customer's first purchase date
    SELECT 
        customer_id,
        MIN(order_date) as cohort_date
    FROM orders
    GROUP BY customer_id
),
customer_orders AS (
    -- Join orders with cohort dates
    SELECT 
        o.customer_id,
        o.order_date,
        fp.cohort_date,
        -- Calculate months since first purchase
        DATE_PART('year', o.order_date) * 12 + DATE_PART('month', o.order_date) -
        (DATE_PART('year', fp.cohort_date) * 12 + DATE_PART('month', fp.cohort_date)) 
        AS months_since_first
    FROM orders o
    JOIN first_purchase fp ON o.customer_id = fp.customer_id
)
SELECT 
    DATE_TRUNC('month', cohort_date) AS cohort_month,
    months_since_first,
    COUNT(DISTINCT customer_id) as customers,
    -- Calculate retention rate
    ROUND(
        100.0 * COUNT(DISTINCT customer_id) / 
        FIRST_VALUE(COUNT(DISTINCT customer_id)) OVER (
            PARTITION BY DATE_TRUNC('month', cohort_date)
            ORDER BY months_since_first
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ),
        2
    ) AS retention_rate
FROM customer_orders
WHERE cohort_date >= '2024-01-01'
GROUP BY DATE_TRUNC('month', cohort_date), months_since_first
ORDER BY cohort_month, months_since_first;
```

```python
# Spark SQL Version
from pyspark.sql.functions import min, max, col, countDistinct, round, months_between, first

# First purchase dates
first_purchase = spark.sql("""
    SELECT 
        customer_id,
        MIN(order_date) as cohort_date
    FROM orders
    GROUP BY customer_id
""")

# Retention analysis
retention = spark.sql("""
    WITH first_purchase AS (
        SELECT customer_id, MIN(order_date) as cohort_date
        FROM orders GROUP BY customer_id
    ),
    customer_orders AS (
        SELECT 
            o.customer_id,
            TRUNC(fp.cohort_date, 'MM') as cohort_month,
            months_between(o.order_date, fp.cohort_date) as months_since_first
        FROM orders o
        JOIN first_purchase fp ON o.customer_id = fp.customer_id
        WHERE fp.cohort_date >= '2024-01-01'
    )
    SELECT 
        cohort_month,
        CAST(months_since_first AS INT) as month_number,
        COUNT(DISTINCT customer_id) as active_customers
    FROM customer_orders
    GROUP BY cohort_month, CAST(months_since_first AS INT)
    ORDER BY cohort_month, month_number
""")

retention.show(50)
```


### Query 3: Time-Series Aggregations with Gaps 📈

```sql
-- ANSI SQL Version: Fill missing dates in time series
WITH date_range AS (
    -- Generate all dates in range
    SELECT generate_series(
        DATE '2024-01-01',
        DATE '2024-12-31',
        INTERVAL '1 day'
    )::DATE AS date
),
actual_sales AS (
    SELECT 
        order_date,
        SUM(amount) as revenue
    FROM orders
    WHERE order_date BETWEEN '2024-01-01' AND '2024-12-31'
    GROUP BY order_date
)
SELECT 
    dr.date,
    COALESCE(as_table.revenue, 0) as revenue,  -- Fill gaps with 0
    -- Previous non-null value (forward fill)
    LAST_VALUE(as_table.revenue IGNORE NULLS) OVER (
        ORDER BY dr.date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as revenue_ffill,
    -- 30-day rolling sum
    SUM(COALESCE(as_table.revenue, 0)) OVER (
        ORDER BY dr.date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as rolling_30day_sum,
    -- Year-over-year comparison
    LAG(as_table.revenue, 365) OVER (ORDER BY dr.date) as revenue_yoy
FROM date_range dr
LEFT JOIN actual_sales as_table ON dr.date = as_table.order_date
ORDER BY dr.date;
```

```python
# Spark SQL Version
from pyspark.sql.functions import expr, coalesce, sum, lag, lit
from pyspark.sql import Window

# Generate date range
date_range = spark.sql("""
    SELECT explode(sequence(
        to_date('2024-01-01'), 
        to_date('2024-12-31'), 
        interval 1 day
    )) as date
""")

# Actual sales
actual_sales = spark.sql("""
    SELECT 
        order_date,
        SUM(amount) as revenue
    FROM orders
    WHERE order_date BETWEEN '2024-01-01' AND '2024-12-31'
    GROUP BY order_date
""")

# Join and fill gaps
window_spec = Window.orderBy("date").rowsBetween(-29, 0)
yoy_window = Window.orderBy("date")

result = date_range.join(actual_sales, date_range.date == actual_sales.order_date, "left") \
    .select(
        "date",
        coalesce(col("revenue"), lit(0)).alias("revenue"),
        sum(coalesce(col("revenue"), lit(0))).over(window_spec).alias("rolling_30day_sum"),
        lag("revenue", 365).over(yoy_window).alias("revenue_yoy")
    ) \
    .orderBy("date")

result.show()
```

This comprehensive guide covers all SQL mastery concepts for Days 1-3 with detailed explanations, emoji organization, and thoroughly commented code examples in both ANSI SQL and Spark SQL. Practice these queries on platforms like LeetCode or HackerRank to solidify your understanding! 🎓✨

<div align="center">⁂</div>

