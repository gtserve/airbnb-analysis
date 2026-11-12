# airbnb-analysis

[![Open EDA Notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1F3G5bKosOoRIGyfoyR_zbynAnv1Aa9eD)
[![Open NLP Notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_O5HCaUTbrReWYpaG7kRV_uSbA4AmL_9)

An end-to-end data mining and natural language processing (NLP) project investigating post-pandemic shifts in Athens' short-term rental market. Using multi-month Airbnb listing and guest review datasets from **2019** and **2023**, this repository combines geospatial market analysis, content-based recommendation, transformer sentiment tracking, and distributional semantics.

## Dataset & Problem Context

Short-term rentals in Athens expanded rapidly between 2019 and 2023 alongside evolving regulatory thresholds (such as restrictions on hosts managing 3+ properties). Using monthly snapshots (February–April) of Athens Airbnb listings and guest reviews across both years, this project answers four core questions:

1. **Market & Spatial Evolution:** How did listing supply, neighborhood pricing, and commercial host concentration shift between 2019 and 2023?
2. **Property Discovery:** Can listing titles and descriptions power an accurate content-based recommendation engine?
3. **Guest Sentiment Over Time:** How did traveler sentiment change across Athens neighborhoods from 2019 to 2023, and which classical ML models best classify review sentiment?
4. **Semantic Word Drift:** How do neighborhood-based word embedding metrics capture contextual similarity between words in guest reviews?

## Repository Structure

```text
athens-airbnb-nlp-analysis/
├── README.md
├── notebooks/
│   ├── 01_athens_airbnb_eda_and_recommender.ipynb
│   └── 02_nlp_sentiment_and_semantic_similarity.ipynb
└── images/
    ├── chart_comparison.png
    └── folium_map.png
```

---

## Methodology & Key Components

### 1. Market Exploration & Content-Based Recommender

*Notebook: `01_athens_airbnb_eda_and_recommender.ipynb`*

* **Data Consolidation & Cleaning:** Merged multi-month CSV snapshots for 2019 and 2023, resolved missing values, removed extreme outliers (e.g., unrealistic room/bedroom counts), and consolidated granular property amenities into high-level categories (*Kitchen, Accessibility, Tech & Electricity, Facilities, Kid-Friendly, Security, Services*).
* **Geospatial & Pricing Analysis:** Mapped listing density across Athens using **Folium**, benchmarked average prices for 2-guest accommodations across neighborhoods, tiered neighborhoods into pricing brackets, and identified top commercial hosts operating 3+ properties.
* **Content-Based Listing Recommender:** Engineered a **TF-IDF** matrix (unigrams and bigrams) over concatenated listing names and descriptions, computing pairwise **Cosine Similarity** to retrieve the top-$N$ most similar properties for any given listing ID, supplemented with `BigramCollocationFinder` phrase extraction.

### 2. Longitudinal Sentiment & Distributional Semantics

*Notebook: `02_nlp_sentiment_and_semantic_similarity.ipynb`*

* **Transformer Sentiment Annotation:** Cleaned guest review text (lowercasing, removing punctuation, URLs, emojis, and stopwords) and labeled 2019 vs. 2023 reviews (*positive, neutral, negative*) using pre-trained **Hugging Face** Transformer models (RoBERTa/BERT) to track overall and neighborhood-level sentiment shifts.
* **Supervised Classifier Benchmarking:** Split annotated reviews (80/20 train/test) and evaluated **SVM, Random Forest, and KNN** classifiers across both **TF-IDF** and **Word Embedding** representations using **10-fold cross-validation** (Accuracy, Precision, Recall, F1-score).
* **Neighborhood-Based Semantic Similarity:** Built a distributional semantics module over review word embeddings to compute similarity between target words using three neighborhood-based graph metrics across varying neighborhood sizes ($N$):
  * **Maximum Similarity** of semantic neighborhoods ($M_n$)
  * **Pearson Correlation** of neighborhood similarity vectors ($R_n$)
  * **Sum of Squared** neighborhood similarities ($E_n^{\theta=2}$)

---

## Key Results

| Classifier | Feature Representation | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SVM** | TF-IDF / Word Embeddings | `[XX.X%]` | `[XX.X%]` | `[XX.X%]` | `[XX.X%]` |
| **Random Forest** | TF-IDF / Word Embeddings | `[XX.X%]` | `[XX.X%]` | `[XX.X%]` | `[XX.X%]` |
| **KNN** | TF-IDF / Word Embeddings | `[XX.X%]` | `[XX.X%]` | `[XX.X%]` | `[XX.X%]` |

* **Market & Pricing Shift (2019 vs. 2023):** *[Insert 1–2 sentences summarizing how average prices, listing volume, or dominant neighborhoods evolved between 2019 and 2023.]*
* **Sentiment & Semantics Takeaway:** *[Insert 1–2 sentences summarizing overall guest sentiment trends and how increasing neighborhood size $N$ impacted word similarity scores.]*

---

## Visual Snapshots

| Neighborhood Price & Sentiment Comparison | Athens Geospatial Listing Map |
| :---: | :---: |
| ![Comparison Chart](images/chart_comparison.png) | ![Folium Map](images/folium_map.png) |

---

## Tech Stack

* **Data Wrangling & Visualization:** Python, Pandas, NumPy, Matplotlib, Seaborn, Folium, WordCloud
* **NLP & Machine Learning:** Hugging Face Transformers, Scikit-Learn, NLTK, Gensim (Word2Vec / FastText / GloVe)
