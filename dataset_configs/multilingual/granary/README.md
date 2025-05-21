# README

This folder is designated for Granary speech data processing configuration files will be added soon. It is associated with a forthcoming paper, which will detail the work done within this project.

Note: This folder is a work in progress.

# Granary

## Yodas2

### Convert to tarred audio dataset
Suggested values for parameters like num_shards and buckets_num depend on the selected `source_lang` and whether `en_translation` is enabled. These values are provided below to help efficiently prepare a ready-to-train tarred audio dataset.

| `source_lang`    |   `bg`  |  `bg`  |   `cs`  |  `cs`  |   `da`  |  `da`  |   `de`  |  `de`  |   `el`  |  `el`  |   `en`  |   `es`  |  `es`  |   `et`  |  `et`  |   `fi`  |  `fi`  |   `fr`  |  `fr`  |   `hr`  |  `hr`  |   `hu`  |  `hu`  |   `it`  |  `it`  |   `lt`  |  `lt`  |   `lv`  |  `lv`  |   `nl`  |  `nl`  |   `pl`  |  `pl`  |   `pt`  |  `pt`  |   `ro`  |  `ro`  |   `ru`  |  `ru`  |   `sk`  |  `sk`  |   `sv`  |  `sv`  |   `uk`  |  `uk`  |
|------------------|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|
| `en_translation` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `True` | `False` | `True` |
| `num_shards`     |   16  |  16  |   32  |  32  |   16  |  16  |  4096 | 1024 |   16  |  16  |  8192 |  8192 | 1024 |   16  |  16  |   64  |  32  |  4096 | 1024 |   16  |  16  |   64  |  32  |  1024 | 1024 |   16  |  16  |   16  |  16  |  1024 |  512 |  256  |  256 |  4096 | 4096 |   16  |  16  |  8192 | 1024 |   16  |  16  |   64  |  32  |  128  |  128 |
| `buckets_num`    |   1   |   1  |   1   |   1  |   1   |   1  |   1   |   1  |   1   |   1  |   4   |   1   |   1  |   1   |   1  |   1   |   1  |   1   |   1  |   1   |   1  |   1   |   1  |   1   |   1  |   1   |   1  |   1   |   1  |   1   |   1  |   1   |   1  |   1   |   1  |   1   |   1  |   1   |   1  |   1   |   1  |   1   |   1  |   1   |   1  |
