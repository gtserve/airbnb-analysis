# airbnb-analysis

A Python data analysis and NLP project exploring Athens Airbnb listings and guest reviews from **2019 and 2023**. It combines market visualisation, content-based property recommendations and sentiment classification.

## Explore the project

| Notebook | What it covers |
| --- | --- |
| [Market analysis & listing recommender](notebooks/01_athens_airbnb_eda_and_recommender.ipynb) | Data cleaning, neighbourhood and room-type comparisons, price distributions, an interactive listing map, and text-based recommendations. |
| [Review sentiment analysis](notebooks/02_nlp_sentiment_and_semantic_similarity.ipynb) | RoBERTa sentiment annotation, sentiment distributions over time, and comparisons of SVM, Random Forest and KNN. |

Both notebooks include saved plots and example outputs.

## Approach

- **Explore the market:** combine monthly listing snapshots with pandas, compare property distributions and prices with Matplotlib and Seaborn, and map 300 sampled listings with Folium.
- **Recommend similar properties:** represent listing names and descriptions with unigram TF-IDF and rank 500 sampled listings by cosine similarity. Extract associated word pairs separately with NLTK bigram collocations.
- **Analyse guest sentiment:** label 1,000 sampled reviews per snapshot year as positive, neutral or negative using a pre-trained RoBERTa model from Hugging Face.
- **Compare text classifiers:** train SVM, Random Forest and KNN with TF-IDF and averaged 100-dimensional Word2Vec embeddings, using 80/20 train/test splits, cross-validation accuracy and held-out classification reports.

## Data & scope

The notebooks use Athens listing and review CSV snapshots from **February–April 2019** and **March, June and September 2023**. Review collections include historical reviews; snapshot years do not identify when every review was written.

Sentiment labels are model-generated, so classifier scores measure agreement with RoBERTa. Evaluation refinement and a clean rerun are pending; the saved scores should be treated as exploratory. Recommendation quality has not been formally evaluated. The semantic neighbourhood similarity section is a planned extension, not an implemented feature.

## Tools

**Python** · **pandas · NumPy · Matplotlib · Seaborn · Folium · WordCloud · scikit-learn · NLTK · Gensim / Word2Vec · Hugging Face Transformers**

## Running status

The notebooks currently target Google Colab and reference personal Google Drive paths. Source datasets are not included. Portable setup instructions and evaluation improvements are planned follow-up work.
