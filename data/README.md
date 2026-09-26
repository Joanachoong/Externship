## Amazon Operational Externship Dataset

See **[data_dictionary.md](data_dictionary.md)** for column details.

**Raw Data (`data/raw/`)**
This directory contains the original, unprocessed datasets collected during the externship:
- `glass_reviews.csv` - Raw employee reviews from Glassdoor for Amazon Fulfillment Centers, including star ratings, employment status, feedback (pros, cons, advice to management), and reviewer metadata.
- `youtube_transcripts.csv` - Raw spoken transcripts and video title metadata extracted from YouTube videos covering worker experiences and conditions at Amazon warehouses.

**Processed Data (`data/processed/`)**
This directory contains cleaned, filtered, and feature-engineered datasets ready for analysis and NLP modeling:
- `youtube_cleaned_text - youtube_cleaned_text.csv` - Cleaned text data from YouTube transcripts with token normalization and noise removed.
- `glassdoor_cleaned.csv` - Cleaned and filtered Glassdoor employee reviews.
- `glassdoor_cleaned_text.csv` - Preprocessed text fields from Glassdoor reviews.
- `glassdoor_cleaned_v2.csv` - Refined version of cleaned Glassdoor review text.
- `glassdoor_sentiment.csv` - Sentiment analysis outputs and scored sentiment metrics for Glassdoor reviews.

## Disclaimer

The current dataset are based on publicly available sources on Glassdoor and Youtube (2022-2025), and do not represent the views or opinions of any individuals or organizations. Please use this dataset responsibly and ethically.