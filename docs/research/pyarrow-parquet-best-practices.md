# PyArrow/Parquet Best Practices for Batch Writing

## Overview

Based on official PyArrow documentation, this guide covers schema definition, partitioned writes, and performance optimization for the Stellaris Advisor data pipeline use case: batch Parquet ingestion with partitioning by `game_id` and `year`, large float arrays, mixed data types, and repeated appends.

---

## 1. Schema Definition for List<Float64> Columns

### Basic Schema with Float Arrays

```python
import pyarrow as pa

# Define schema with float array columns (109-element feature vectors)
schema = pa.schema([
    ('game_id', pa.string()),
    ('year', pa.int32()),
    ('empire_id', pa.int32()),
    ('system_id', pa.int32()),
    ('timestamp', pa.int64()),
    # Float array column (109 features)
    ('empire_features', pa.list_(pa.float64())),
    ('system_features', pa.list_(pa.float64())),
    # Mixed data types
    ('strategy_name', pa.string()),
    ('config_json', pa.string()),
    ('reward', pa.float64()),
    ('action_id', pa.int16()),
])
```

### Creating Arrays with List Values

```python
import pyarrow as pa
import numpy as np

# Create list<float64> array from nested Python lists
feature_vectors = [
    [1.5, 2.3, 3.1, ...],  # 109 floats
    [2.1, 3.4, 4.2, ...],
    None,  # Nullable array
]
empire_features_array = pa.array(
    feature_vectors,
    type=pa.list_(pa.float64())
)

# From NumPy: reshape 2D array to list of arrays
features_2d = np.random.randn(1000, 109)
feature_list = [row.tolist() for row in features_2d]
empire_features_array = pa.array(feature_list, type=pa.list_(pa.float64()))

# Build table
table = pa.table({
    'game_id': pa.array(['game_001'] * 1000, type=pa.string()),
    'year': pa.array([2500] * 1000, type=pa.int32()),
    'empire_features': empire_features_array,
}, schema=schema)
```

### Key Points

- **List type**: `pa.list_(pa.float64())` defines variable-length arrays of floats
- **Nullable by default**: Use `pa.list_(pa.float64(), list_size=109)` for fixed-size arrays if all rows have exactly 109 elements (more efficient)
- **Schema flexibility**: Schemas can be unified across batches using `pa.unify_schemas([schema1, schema2, ...])`

---

## 2. Partitioned Write Patterns

### Pattern A: Using `write_to_dataset()` (Recommended for Appends)

```python
import pyarrow.parquet as pq

# Write table with partitioning by game_id and year
# Creates directory structure: dataset/game_id=game_001/year=2500/
pq.write_to_dataset(
    table,
    root_path='data/dataset',
    partition_cols=['game_id', 'year'],
)
```

**Behavior:**
- Automatically creates Hive-style directories: `game_id=<value>/year=<value>/`
- Columns in `partition_cols` are **removed** from the Parquet files (stored in directory structure only)
- **Does NOT support append**: If partition directory exists, overwrites with new data
- Creates `_metadata` and `_common_metadata` files for efficient schema discovery

### Pattern B: Using `ds.write_dataset()` with Explicit Partitioning

```python
import pyarrow.dataset as ds
import pyarrow as pa

table = pa.table({...})

# Define partitioning scheme
partitioning = ds.partitioning(
    pa.schema([
        ('game_id', pa.string()),
        ('year', pa.int32()),
    ]),
    flavor='hive'  # Hive-style: game_id=value/year=value
)

# Write with explicit partitioning
ds.write_dataset(
    table,
    root_path='data/dataset',
    format='parquet',
    partitioning=partitioning,
    basename_template='part-{i:03d}.parquet',  # For multiple files per partition
)
```

**Advantages:**
- Explicit partitioning control
- `basename_template` allows multiple files per partition (useful for large partitions)
- Can specify `existing_data_behavior='overwrite'` (default) or `'error'`

### Pattern C: Manual Partition-Level Writes (For True Append)

```python
import pyarrow.parquet as pq
from pathlib import Path
import uuid

root_path = Path('data/dataset')
game_id = 'game_001'
year = 2500

# Create partition directory
partition_dir = root_path / f'game_id={game_id}' / f'year={year}'
partition_dir.mkdir(parents=True, exist_ok=True)

# Write with unique filename (enables appending to same partition)
unique_id = str(uuid.uuid4())[:8]
file_path = partition_dir / f'part-{unique_id}.parquet'

# Remove partition columns before writing
table_to_write = table.drop(['game_id', 'year'])

pq.write_table(
    table_to_write,
    str(file_path),
    metadata_collector=[],  # For later metadata consolidation
)
```

**When to use:**
- Repeated appends to the same partition
- Each batch write gets a unique filename
- Schema must be identical across all writes to same partition

### Key Gotcha: Metadata Management

After multiple manual writes, consolidate metadata:

```python
import pyarrow.parquet as pq

root_path = Path('data/dataset')
metadata_collector = []

# After all writes, consolidate metadata
for partition_dir in root_path.glob('game_id=*/year=*/'):
    for parquet_file in partition_dir.glob('part-*.parquet'):
        table = pq.read_table(parquet_file)
        # Metadata already managed by PyArrow internally

# PyArrow automatically regenerates _metadata on read if missing
```

---

## 3. Column Ordering and Chunking

### Column Ordering Best Practices

```python
# Order columns: partitioning columns first (will be removed),
# then frequently accessed columns, then large arrays

schema = pa.schema([
    # Partition columns (appear first, removed from files)
    ('game_id', pa.string()),
    ('year', pa.int32()),
    # Frequently filtered/accessed
    ('empire_id', pa.int32()),
    ('timestamp', pa.int64()),
    # Large arrays
    ('empire_features', pa.list_(pa.float64())),
    ('system_features', pa.list_(pa.float64())),
    # Remaining columns
    ('strategy_name', pa.string()),
    ('reward', pa.float64()),
])
```

**Why?** Parquet readers scan columns in order; frequent access first minimizes I/O.

### Chunking Configuration

```python
import pyarrow.parquet as pq

# For large batches (1000+ rows) with large arrays:
pq.write_table(
    table,
    'output.parquet',
    # Enable content-defined chunking for better compression
    use_content_defined_chunking=True,
    compression='snappy',  # Fast compression
    compression_level=4,   # Balance speed vs ratio
)

# For very large arrays, customize chunk sizes
pq.write_table(
    table,
    'output.parquet',
    use_content_defined_chunking={
        'min_chunk_size': 256 * 1024,    # 256 KiB
        'max_chunk_size': 1024 * 1024,   # 1 MiB
    },
    data_page_size=1024 * 1024,  # 1 MiB per column page
)
```

**Gotcha:**
- Default `data_page_size=1MiB` may be too small for 109-float arrays × 1000 rows (≈1MB per column)
- Increase to `data_page_size=4 * 1024 * 1024` (4 MiB) for large arrays

---

## 4. Performance Tips for Large Batches

### Compression Strategy

```python
import pyarrow.parquet as pq

# Per-column compression for mixed types
pq.write_table(
    table,
    'output.parquet',
    # String columns: use brotli (high compression)
    # Float arrays: use snappy (fast)
    compression={
        'strategy_name': 'brotli',
        'config_json': 'brotli',
        'empire_features': 'snappy',
        'system_features': 'snappy',
    },
    compression_level={
        'brotli': 11,  # Max compression (slow, for strings)
        'snappy': 1,   # Min compression (fast, for floats)
    },
)
```

**Recommendations:**
- **Float arrays (109 elements):** `snappy` (3-4% compression, fastest)
- **String/JSON columns:** `brotli` (25-30% compression, slower)
- **Mixed batch:** `zstd` as middle ground (8-12% compression)

### Dictionary Encoding

```python
import pyarrow.parquet as pq

# Enable dictionary encoding for low-cardinality string columns
pq.write_table(
    table,
    'output.parquet',
    use_dictionary=['empire_id', 'strategy_name'],  # High repeat rate
    use_dictionary=False,  # Disable for float arrays
)
```

**When to use:**
- String columns with <1000 unique values
- Integer ID columns (empire_id, system_id)
- **Avoid** for float arrays (compression is minimal)

### Batch Size Tuning

```python
# For 1000s of rows per write:
# Row count: 1000
# Float array size: 109 × 8 bytes = 872 bytes per row
# Total per batch: ~1MB per feature column
# Recommended: 1000-5000 rows per batch

def write_batch(table: pa.Table, root_path: str, game_id: str, year: int):
    """Write batch with auto-partitioning."""
    partition_cols = ['game_id', 'year']

    # Add partition columns to table
    table = table.append_column(pa.field('game_id', pa.string()),
                                pa.array([game_id] * len(table)))
    table = table.append_column(pa.field('year', pa.int32()),
                                pa.array([year] * len(table)))

    pq.write_to_dataset(
        table,
        root_path=root_path,
        partition_cols=partition_cols,
        compression='snappy',
        use_content_defined_chunking=True,
    )
```

---

## 5. Handling Repeated Writes to Same Partition

### Pattern: Append with Unique Filenames

```python
import pyarrow.parquet as pq
from pathlib import Path
import uuid
from datetime import datetime

def append_to_partition(
    table: pa.Table,
    game_id: str,
    year: int,
    root_path: str = 'data/dataset'
) -> str:
    """Append table to partition, creating new file."""
    partition_dir = Path(root_path) / f'game_id={game_id}' / f'year={year}'
    partition_dir.mkdir(parents=True, exist_ok=True)

    # Unique filename: timestamp + random suffix
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    suffix = str(uuid.uuid4())[:8]
    file_path = partition_dir / f'part-{timestamp}-{suffix}.parquet'

    # Remove partition columns (they're in directory structure)
    table_to_write = table.drop(['game_id', 'year'])

    pq.write_table(
        table_to_write,
        str(file_path),
        compression='snappy',
        use_content_defined_chunking=True,
    )

    return str(file_path)

# Usage: Write multiple batches to same partition
for batch in batches:
    file_path = append_to_partition(
        batch,
        game_id='game_001',
        year=2500
    )
    print(f"Wrote: {file_path}")
```

### Reading from Multi-File Partitions

```python
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# Read all files in a partition as single table
partition_dir = Path('data/dataset/game_id=game_001/year=2500')
table = pq.read_table(str(partition_dir))

# Or use PyArrow Dataset API (recommended for large datasets)
dataset = ds.dataset(
    'data/dataset',
    format='parquet',
    partitioning=ds.partitioning(
        pa.schema([
            ('game_id', pa.string()),
            ('year', pa.int32()),
        ]),
        flavor='hive'
    )
)

# Filter by partition
filtered = dataset.to_table(
    filter=ds.field('game_id') == 'game_001'
)
```

### Schema Consistency Check

```python
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def validate_partition_schema(
    partition_dir: Path,
    expected_schema: pa.Schema
) -> bool:
    """Ensure all files in partition have consistent schema."""
    parquet_files = list(partition_dir.glob('part-*.parquet'))

    for parquet_file in parquet_files:
        file_schema = pq.read_schema(str(parquet_file))
        # Unify schemas (accounts for field ordering differences)
        unified = pa.unify_schemas([expected_schema, file_schema])
        if unified != expected_schema:
            print(f"Schema mismatch in {parquet_file}")
            return False

    return True

# Before appending
partition = Path('data/dataset/game_id=game_001/year=2500')
if partition.exists():
    assert validate_partition_schema(partition, schema)
```

---

## 6. Summary: Key Gotchas & Best Practices

| Issue | Gotcha | Solution |
|-------|--------|----------|
| **Partition column handling** | `write_to_dataset()` removes partition cols from files | Add them back before writing, or manage manually with unique filenames |
| **Append mode** | `write_to_dataset()` overwrites, not appends | Use manual writes with unique filenames per batch |
| **Large arrays** | Default `data_page_size=1MB` too small for 109×f64 arrays | Increase to `4MB` or use content-defined chunking |
| **Schema drift** | Repeated appends can introduce schema inconsistencies | Validate schema before each append with `pa.unify_schemas()` |
| **Metadata staleness** | `_metadata` file can become stale with many appends | Regenerate periodically with `pq.write_metadata()` or rely on PyArrow's auto-regeneration |
| **Float compression** | Dictionary encoding ineffective for float arrays | Use `snappy` or `zstd`, disable dictionary for floats |
| **Column order** | Parquet reads columns in order | Place frequently accessed columns first |
| **Mixed types** | Single compression codec suboptimal for all types | Use per-column compression settings |

---

## 7. Complete Example: Production-Ready Batch Writer

```python
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
import uuid
from typing import Optional

class ParquetBatchWriter:
    """Production-grade Parquet writer for Stellaris Advisor data."""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

        # Define schema once
        self.schema = pa.schema([
            ('game_id', pa.string()),
            ('year', pa.int32()),
            ('empire_id', pa.int32()),
            ('system_id', pa.int32()),
            ('timestamp', pa.int64()),
            ('empire_features', pa.list_(pa.float64())),
            ('system_features', pa.list_(pa.float64())),
            ('strategy_name', pa.string()),
            ('config_json', pa.string()),
            ('reward', pa.float64()),
            ('action_id', pa.int16()),
        ])

    def write_batch(
        self,
        table: pa.Table,
        game_id: str,
        year: int,
        batch_id: Optional[str] = None,
    ) -> str:
        """Write batch to partition with unique filename."""

        # Validate schema
        if table.schema != self.schema:
            table = table.cast(self.schema)

        # Create partition directory
        partition_dir = (
            self.root_path / f'game_id={game_id}' / f'year={year}'
        )
        partition_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        if batch_id is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            batch_id = f'{timestamp}-{str(uuid.uuid4())[:8]}'

        file_path = partition_dir / f'part-{batch_id}.parquet'

        # Remove partition columns before writing
        table_to_write = table.drop(['game_id', 'year'])

        # Write with optimized settings
        pq.write_table(
            table_to_write,
            str(file_path),
            compression={
                'strategy_name': 'brotli',
                'config_json': 'brotli',
                'empire_features': 'snappy',
                'system_features': 'snappy',
            },
            use_dictionary=['empire_id', 'system_id', 'strategy_name'],
            use_content_defined_chunking={
                'min_chunk_size': 256 * 1024,
                'max_chunk_size': 1024 * 1024,
            },
            data_page_size=4 * 1024 * 1024,  # 4 MiB for large arrays
        )

        return str(file_path)

    def read_partition(self, game_id: str, year: int) -> pa.Table:
        """Read all files in a partition as single table."""
        partition_dir = self.root_path / f'game_id={game_id}' / f'year={year}'

        if not partition_dir.exists():
            return pa.table({}, schema=self.schema.drop(['game_id', 'year']))

        return pq.read_table(str(partition_dir))

# Usage
writer = ParquetBatchWriter('data/stellaris_advisor')

# Write batch 1
batch_1 = pa.table({
    'game_id': ['game_001'] * 1000,
    'year': [2500] * 1000,
    'empire_id': list(range(1000)),
    'system_id': list(range(1000)),
    'timestamp': [0] * 1000,
    'empire_features': [[1.0] * 109] * 1000,
    'system_features': [[2.0] * 12] * 1000,
    'strategy_name': ['expand'] * 1000,
    'config_json': ['{}'] * 1000,
    'reward': [0.5] * 1000,
    'action_id': [1] * 1000,
})

file1 = writer.write_batch(batch_1, 'game_001', 2500)
print(f"Wrote: {file1}")

# Later: Write batch 2 to same partition (appends)
batch_2 = pa.table({...})  # Next 1000 rows
file2 = writer.write_batch(batch_2, 'game_001', 2500)
print(f"Wrote: {file2}")

# Read entire partition
partition_data = writer.read_partition('game_001', 2500)
print(f"Total rows: {len(partition_data)}")  # 2000
```

---

## References

- **PyArrow Schema Documentation:** https://arrow.apache.org/docs/python/data.html
- **Parquet Write Options:** https://arrow.apache.org/docs/python/parquet.html
- **Dataset API (Partitioning):** https://arrow.apache.org/docs/python/dataset.html
- **Compression & Encoding:** https://arrow.apache.org/docs/python/parquet.html#compression-and-encoding
