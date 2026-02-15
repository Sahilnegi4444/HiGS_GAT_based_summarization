# Data Directory

## Quick Start — Download Pre-Cleaned Data

📥 **[Download Cleaned Dataset (Parquet)](https://drive.google.com/drive/folders/1l_5WC5gacZAnCjZCgcSC6ZvNc4Sa2Igc?usp=sharing)**

Place the downloaded file at:
```
data/processed/newssumm_cleaned.parquet
```

📥 **[Download HiGS Model Checkpoint](https://drive.google.com/drive/folders/1hqYPvjdl443WFcgfs9OA-73p0U5Nusbm?usp=sharing)**

Place the downloaded `.pt` file at:
```
results/higs/best_checkpoint.pt
```

## Raw Dataset (Optional)

To run the full cleaning pipeline from scratch, place the raw Excel file here:
```
data/NewsSumm_Dataset.xlsx
```

After running the cleaning and preprocessing pipelines, the following files will be generated:

```
data/
├── NewsSumm_Dataset.xlsx          # Original dataset (place manually)
└── processed/
    ├── newssumm_cleaned.parquet   # Cleaned dataset
    ├── train.parquet              # Training split (80%)
    ├── val.parquet                # Validation split (10%)
    ├── test.parquet               # Test split (10%)
    ├── eda_distributions.png     # EDA visualizations
    └── eda_compression.png       # Compression ratio plots
```

## Dataset Description

The **NewsSumm** dataset is a large-scale Indian English multi-document news summarization corpus containing ~100,000+ article-summary pairs from major Indian news sources.

**Columns:**
- `newspaper_name` — Source publication
- `published_date` — Publication date
- `headline` — Article headline
- `article_text` — Full article text
- `human_summary` — Human-written reference summary
- `news_category` — News category (politics, sports, business, etc.)
