# Data Directory

Place the NewsSumm dataset file here:

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
