# Local dataset layout

Dataset shards are intentionally excluded from Git. Place the course-provided
WebDataset archives at:

```text
data/train/shard-000000.tar
data/train/shard-000001.tar
data/train/shard-000002.tar
data/train/shard-000003.tar
data/train/shard-000004.tar
data/test-clean/*.tar
data/test-other/*.tar
```

The strict protocol trains on train shards `000000`–`000003`, uses `000004`
only for validation/checkpoint and decoder selection, and accesses the test
splits only during final evaluation.
