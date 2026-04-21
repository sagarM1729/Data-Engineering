<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>



> 💡 **GLOBAL INTERVIEW TIPS BEFORE WE START**
> - 🎯 Always **think aloud** — interviewers care about *how* you think, not just the answer
> - ⏱️ Say time \& space complexity after every coding answer
> - 🔄 Start with brute force → then optimize
> - ❓ Ask 1-2 clarifying questions before coding
> - 🧪 Always trace through your code with a sample input
> - 🚫 Don't say "I don't know" — say "Let me think through this..."

***

## 🧩 Section 1: Coding Questions


***

### ❓ Q1 — Reverse a String Without Built-ins

```python
def reverse_string(s):
    result = ""
    for char in s:
        result = char + result  # prepend each char
    return result

print(reverse_string("hello"))  # "olleh"
```

✅ **Answer:** Loop through characters and prepend each one. Can also use a pointer swap approach on a list.
⏱️ **Time:** O(n) | **Space:** O(n)

***

### ❓ Q2 — Second Largest Number in a List

```python
def second_largest(nums):
    first = second = float('-inf')
    for n in nums:
        if n > first:
            second = first
            first = n
        elif n > second and n != first:
            second = n
    return second

print(second_largest([3, 1, 4, 1, 5, 9, 2]))  # 5
```

✅ **Answer:** Single-pass, track top two values. Handles duplicates.
⏱️ **Time:** O(n) | **Space:** O(1)

***

### ❓ Q3 — Remove Duplicates Without `set()`

```python
def remove_duplicates(lst):
    seen = {}
    result = []
    for item in lst:
        if item not in seen:
            seen[item] = True
            result.append(item)
    return result

print(remove_duplicates([1, 2, 2, 3, 1, 4]))  # [1, 2, 3, 4]
```

✅ **Answer:** Use a dictionary as a hash map to track seen elements. Preserves order.

***

### ❓ Q4 — Count Frequency of Each Character

```python
def char_frequency(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq

print(char_frequency("abracadabra"))
# {'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1}
```

✅ **Answer:** Use a dict with `.get(key, 0) + 1` pattern. Clean and interview-safe.

***

### ❓ Q5 — Check if String is a Palindrome

```python
def is_palindrome(s):
    s = s.lower().replace(" ", "")
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True

print(is_palindrome("Race Car"))  # True
```

✅ **Answer:** Two-pointer approach. Mention edge case handling (case, spaces).

***

### ❓ Q6 — Find Missing Number in List (1 to n)

```python
def find_missing(nums):
    n = len(nums) + 1
    expected_sum = n * (n + 1) // 2  # Gauss formula
    return expected_sum - sum(nums)

print(find_missing([1, 2, 4, 5, 6]))  # 3
```

✅ **Answer:** Use the Gauss formula `n*(n+1)/2`. O(n) time, O(1) space. 🔥 Interviewers love this.

***

### ❓ Q7 — Common Elements Between Two Lists

```python
def common_elements(a, b):
    set_b = set(b)  # O(1) lookup
    return [x for x in a if x in set_b]

print(common_elements([1, 2, 3, 4], [2, 4, 6]))  # [2, 4]
```

✅ **Answer:** Convert one to a set for O(1) lookup, then iterate the other.

***

### ❓ Q8 — Flatten a Nested List

```python
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))  # recursion for arbitrary depth
        else:
            result.append(item)
    return result

print(flatten([1, [2, [3, 4], 5], 6]))  # [1, 2, 3, 4, 5, 6]
```

✅ **Answer:** Recursion handles arbitrary nesting. Mention `isinstance` check as key pattern.

***

### ❓ Q9 — First Non-Repeating Character

```python
def first_non_repeating(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    for char in s:  # preserve order
        if freq[char] == 1:
            return char
    return None

print(first_non_repeating("aabbcde"))  # 'c'
```

✅ **Answer:** Two-pass: first count, then find first with count=1.

***

### ❓ Q10 — Group Words That Are Anagrams

```python
from collections import defaultdict

def group_anagrams(words):
    groups = defaultdict(list)
    for word in words:
        key = tuple(sorted(word))  # sorted chars as key
        groups[key].append(word)
    return list(groups.values())

print(group_anagrams(["eat","tea","tan","ate","nat","bat"]))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

✅ **Answer:** Sorted characters of each word = unique anagram key. Classic HashMap grouping.

***

## 🐍 Section 2: Python Concepts


***

### ❓ Q11 — Sort a Dictionary by Values

```python
data = {"a": 3, "b": 1, "c": 2}

# ascending
sorted_dict = dict(sorted(data.items(), key=lambda x: x[^1_1]))
print(sorted_dict)  # {'b': 1, 'c': 2, 'a': 3}
```

✅ **Answer:** `sorted()` with `key=lambda x: x[^1_1]`. Add `reverse=True` for descending.

***

### ❓ Q12 — Merge Two Dictionaries

```python
d1 = {"a": 1, "b": 2}
d2 = {"b": 99, "c": 3}

# Python 3.9+
merged = d1 | d2          # d2 wins on conflict
# OR
merged = {**d1, **d2}     # same behavior, older compatible

print(merged)  # {'a': 1, 'b': 99, 'c': 3}
```

✅ **Answer:** `|` operator (3.9+) or `{**d1, **d2}`. Latter dict wins on key conflict.

***

### ❓ Q13 — List vs Tuple vs Set vs Dictionary

| 🔧 Type | Order | Mutable | Duplicates | Key-Value |
| :-- | :-- | :-- | :-- | :-- |
| `list` | ✅ | ✅ | ✅ | ❌ |
| `tuple` | ✅ | ❌ | ✅ | ❌ |
| `set` | ❌ | ✅ | ❌ | ❌ |
| `dict` | ✅ (3.7+) | ✅ | ❌ keys | ✅ |

✅ **Interview tip:** Always give a real use case — "tuples for coordinates, sets for dedup."

***

### ❓ Q14 — Mutable vs Immutable

✅ **Answer:**

- **Immutable:** `int, float, str, tuple, frozenset` — value can't change after creation. Safe as dict keys.
- **Mutable:** `list, dict, set` — can be modified in-place.

```python
x = "hello"
x[^1_0] = "H"  # ❌ TypeError

lst = [1, 2, 3]
lst[^1_0] = 99  # ✅ Works
```


***

### ❓ Q15 — What Are Decorators?

✅ **Answer:** A decorator wraps a function to add behavior without modifying its code. Used for logging, auth, timing.

```python
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-start:.4f}s")
        return result
    return wrapper

@timer
def slow_task():
    import time; time.sleep(1)

slow_task()  # slow_task took 1.0003s
```

🎯 **Tip:** Mention `functools.wraps` to preserve docstrings in production decorators.

***

### ❓ Q16 — What Are Generators?

✅ **Answer:** Generators yield values lazily — one at a time — instead of loading everything into memory. Use `yield` instead of `return`.

```python
def count_up(n):
    for i in range(n):
        yield i  # pauses here, resumes on next()

gen = count_up(1000000)  # uses almost no memory
print(next(gen))  # 0
print(next(gen))  # 1
```

🎯 **Use when:** Large datasets, streaming data pipelines (very relevant for your DE work! 🔥).

***

### ❓ Q17 — Shallow Copy vs Deep Copy

```python
import copy

original = [[1, 2], [3, 4]]

shallow = copy.copy(original)
deep = copy.deepcopy(original)

original[^1_0][^1_0] = 99
print(shallow[^1_0][^1_0])  # 99 — shares inner objects
print(deep[^1_0][^1_0])     # 1  — fully independent
```

✅ **Answer:** Shallow copy duplicates the outer container but shares inner references. Deep copy is fully independent.

***

### ❓ Q18 — `*args` vs `**kwargs`

```python
def demo(*args, **kwargs):
    print(args)    # tuple: (1, 2, 3)
    print(kwargs)  # dict: {'name': 'Sagar', 'role': 'DE'}

demo(1, 2, 3, name="Sagar", role="DE")
```

✅ **Answer:** `*args` = variable positional args (tuple). `**kwargs` = variable keyword args (dict). Both can be used together.

***

### ❓ Q19 — Lambda Functions

```python
square = lambda x: x ** 2
print(square(5))  # 25

# Common use: sorting
data = [{"name": "Zara", "age": 25}, {"name": "Amit", "age": 22}]
data.sort(key=lambda x: x["age"])
```

✅ **Answer:** Anonymous one-liner function. Best for short, throwaway logic in `sorted()`, `map()`, `filter()`.

***

### ❓ Q20 — Recursion — When to Use It

```python
def factorial(n):
    if n <= 1:
        return 1          # base case — always required!
    return n * factorial(n - 1)

print(factorial(5))  # 120
```

✅ **Answer:** Use recursion when the problem has **natural self-similarity** (trees, graphs, nested structures). Avoid for deep recursion (Python default stack limit = 1000). Prefer iteration for performance.

***

### ❓ Q21 — Multithreading vs Multiprocessing

|  | Multithreading | Multiprocessing |
| :-- | :-- | :-- |
| 🔒 GIL | Blocked | Bypasses it |
| Best for | I/O-bound tasks | CPU-bound tasks |
| Memory | Shared | Separate |
| Example | Web requests, file I/O | Image processing, ML |

✅ **Answer:** For CPU-heavy work, use `multiprocessing`. For I/O (network, disk), threading works fine.

***

### ❓ Q22 — What is GIL?

✅ **Answer:** The **Global Interpreter Lock (GIL)** is a mutex in CPython that allows only one thread to execute Python bytecode at a time — even on multi-core CPUs.[^1_1]

🔥 **2026 Update:** Python 3.13 introduced an experimental **no-GIL build** (`--disable-gil`), which enables true multi-core parallelism for the first time in CPython history.  You need to compile from source to use it currently.[^1_2]

🎯 **Interview gold:** "GIL affects CPU-bound threads. I/O-bound threads are fine. Multiprocessing bypasses it entirely."

***

### ❓ Q23 — Iterators vs Iterables

```python
# Iterable = has __iter__()
my_list = [1, 2, 3]  # iterable

# Iterator = has __iter__() AND __next__()
my_iter = iter(my_list)
print(next(my_iter))  # 1
print(next(my_iter))  # 2
```

✅ **Answer:** Every iterator is iterable, but not every iterable is an iterator. `for` loops call `iter()` behind the scenes.

***

### ❓ Q24 — List Comprehension

```python
# Without comprehension
squares = []
for x in range(10):
    if x % 2 == 0:
        squares.append(x**2)

# With comprehension — same thing, 1 line
squares = [x**2 for x in range(10) if x % 2 == 0]
```

✅ **Answer:** Syntactic sugar for creating lists. More readable and slightly faster than loops. Can also do dict `{}` and set `{}` comprehensions.

***

### ❓ Q25 — `map()` vs `filter()`

```python
nums = [1, 2, 3, 4, 5]

doubled = list(map(lambda x: x * 2, nums))     # [2, 4, 6, 8, 10]
evens   = list(filter(lambda x: x % 2 == 0, nums))  # [2, 4]
```

✅ **Answer:** `map()` transforms every element. `filter()` selects elements matching a condition. Both return lazy iterators — wrap in `list()` to evaluate.

***

### ❓ Q26 — `append()` vs `extend()`

```python
a = [1, 2, 3]
a.append([4, 5])   # [1, 2, 3, [4, 5]]  ← adds as ONE element

b = [1, 2, 3]
b.extend([4, 5])   # [1, 2, 3, 4, 5]   ← unpacks and adds each
```

✅ **Answer:** `append()` adds the object as-is. `extend()` iterates and adds each element individually.

***

### ❓ Q27 — Memory Management in Python

✅ **Answer:** Python uses **reference counting** as the primary mechanism — when an object's count hits 0, it's freed. A **cyclic garbage collector** handles circular references. Memory is managed via `pymalloc`, a private heap. You never manually allocate/free memory like C.

🎯 **Tip:** Mention `gc` module for manual GC control and `sys.getsizeof()` for object size.

***

### ❓ Q28 — Passing a List to a Function

```python
def add_item(lst):
    lst.append(99)  # modifies the ORIGINAL list

my_list = [1, 2, 3]
add_item(my_list)
print(my_list)  # [1, 2, 3, 99] — changed!
```

✅ **Answer:** Python passes by **object reference**. Mutable objects (lists, dicts) CAN be modified inside the function. Pass a copy (`lst[:]`) to avoid side effects.

***

### ❓ Q29 — Why is Python Slower Than Compiled Languages?

✅ **Answer:**

- Python is **interpreted** — code translates at runtime, not ahead-of-time
- **Dynamic typing** — type checks happen every operation
- **GIL** limits true multi-threading
- **Object overhead** — everything is a full Python object

🎯 **Tip:** "For performance-critical code, we use NumPy (C backend), Cython, or offload to Spark/compiled services."

***

### ❓ Q30 — Set vs List — When to Use Set?

✅ **Use a `set` when:**

- You need **O(1) membership testing** (`in` operator)
- You need **deduplication**
- You're doing **mathematical set operations** (union, intersection)

```python
# O(n) with list
if x in [1, 2, 3, ..., 1000000]:  # slow

# O(1) with set
if x in {1, 2, 3, ..., 1000000}:  # fast ⚡
```


***

## 🌐 Section 3: Flask Questions


***

### ❓ Q31 — What is Flask?

✅ **Answer:** Flask is a **lightweight WSGI micro web framework** for Python. It gives you just the essentials (routing, request/response, templating) with no forced project structure. Best for small to medium web apps, dashboards, and quick REST APIs.[^1_3]

🎯 **When to use:** Prototypes, internal tools, apps with HTML rendering (Jinja2 templates), or when you need Flask's massive extension ecosystem (Flask-SQLAlchemy, Flask-Login, etc.).[^1_3]

***

### ❓ Q32 — Basic Flask Application

```python
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(debug=True)  # never use debug=True in prod
```

✅ **Run:** `python app.py` → opens on `http://localhost:5000`

***

### ❓ Q33 — What Are Routes in Flask?

```python
from flask import Flask, request

app = Flask(__name__)

@app.route("/user/<int:user_id>", methods=["GET"])
def get_user(user_id):
    return {"user_id": user_id}

@app.route("/submit", methods=["POST"])
def submit():
    data = request.json
    return {"received": data}, 201
```

✅ **Answer:** Routes map URL paths to Python functions using `@app.route()`. Support path params (`<type:name>`), query strings, and HTTP method filtering.

***

### ❓ Q34 — FastAPI vs Flask

| Feature | Flask 🧴 | FastAPI ⚡ |
| :-- | :-- | :-- |
| Architecture | Sync WSGI | Async ASGI |
| Performance | ~2–3k req/s | ~15–20k req/s [^1_3] |
| Async support | Workarounds | Native `async/await` [^1_4] |
| Data validation | Manual | Auto via Pydantic [^1_4] |
| API Docs | None built-in | Auto Swagger + ReDoc [^1_4] |
| Best for | Web apps, dashboards | High-perf APIs, ML serving |
| Maturity | 14+ years | Newer but growing fast [^1_3] |


***

### ❓ Q35 — GET Endpoint in FastAPI

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{user_id}")
def get_user(user_id: int, active: bool = True):
    return {"user_id": user_id, "active": active}
```

✅ **Answer:** Use `@app.get()` decorator. Path params are type-annotated in the function signature. Query params (like `active`) are optional with defaults. FastAPI auto-validates types.[^1_4]

***

### ❓ Q36 — Handle POST Request Body with Pydantic

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name: str
    age: int
    email: str

@app.post("/users/", status_code=201)
def create_user(user: User):
    return {"message": f"Created {user.name}", "data": user}
```

✅ **Answer:** Define a `BaseModel` class with Pydantic. FastAPI auto-parses the JSON body into the model and returns a 422 with details if validation fails.[^1_3]

***

### ❓ Q37 — GET vs POST vs PUT

| Method | Purpose | Body? | Idempotent? |
| :-- | :-- | :-- | :-- |
| `GET` | Fetch resource | ❌ | ✅ |
| `POST` | Create new resource | ✅ | ❌ |
| `PUT` | Replace full resource | ✅ | ✅ |
| `PATCH` | Partial update | ✅ | ✅ |

✅ **Idempotent** = calling it multiple times gives same result.

***

### ❓ Q38 — PUT vs POST in API Design

✅ **Answer:**

- Use **POST** when the server assigns the ID: `POST /users` → creates a new user
- Use **PUT** when the client knows the full resource: `PUT /users/42` → replaces user 42 entirely
- Use **PATCH** for partial updates: `PATCH /users/42` → only update email

🎯 **Interview tip:** "POST is not idempotent — calling it twice creates two users. PUT is idempotent — calling it twice gives the same result."

***

### ❓ Q39 — FastAPI Validation with Pydantic

```python
from pydantic import BaseModel, EmailStr, validator

class Product(BaseModel):
    name: str
    price: float
    quantity: int = 0

    @validator("price")
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Price must be positive")
        return v
```

✅ **Answer:** FastAPI uses Pydantic models for automatic request body validation. Invalid data returns a `422 Unprocessable Entity` with detailed error messages in JSON.  Custom validators via `@validator` decorator handle business logic.[^1_3]

***

### ❓ Q40 — Running FastAPI with Uvicorn

```bash
# Install
pip install fastapi uvicorn

# Run (development)
uvicorn main:app --reload --port 8000

# Run (production - multiple workers)
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
```

✅ **Answer:** Uvicorn is an **ASGI server** that runs FastAPI apps.  `--reload` auto-restarts on code changes (dev only). For production, use multiple `--workers` or deploy behind Gunicorn with Uvicorn workers. Access auto-docs at `http://localhost:8000/docs` 📖.[^1_3]

***

## 🏆 Final Interview Cheat Sheet

> 🎯 **Top 5 things that make you stand out:**
> 1. 🔢 Always mention **time \& space complexity**
> 2. 🔄 Mention **edge cases** (empty input, negatives, duplicates)
> 3. 🆕 Drop **Python 3.13 no-GIL** mention for GIL question — instant wow factor[^1_1]
> 4. ⚡ For FastAPI, say "15-20k RPS vs Flask's 2-3k RPS" — shows you know benchmarks[^1_3]
> 5. 🏗️ Relate answers to your **Data Engineering context** (pipelines, Spark, scale)
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_5][^1_6][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>
