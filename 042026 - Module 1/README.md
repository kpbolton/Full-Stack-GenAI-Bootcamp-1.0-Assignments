# Text Feature Engineering on Amazon Product Reviews
This project originated as an assignment from Krish Naik's Full-Stack Generative AI & Agentic AI Bootcamp.  The first module focused on the core fundamentals of NLP, namely encoding and embedding techniques.  These were the fundamentals that results in the transformer architecture that underpins our state-of-the-art LLMs today.

This project first collects real-word product review data from Amazon and applies fundamental NLP techniques, converting them from raw text into vectors ready to begin machine learning analysis.  This work was implemented with Github Copilot.

Due to poor model performance of this dataset (poor data scraping strategy), a second dataset was obtained from ![Kaggle](https://www.kaggle.com/code/jalesummak/amazon-reviews-topic-modeling-with-nlp-nmf-lda) of Amazon reviews for a different product with a better balance of positive and negative reviews.

# Project Objectives
1. Scrape real custoemr reviews from Amazon.  This project selected the ![GAOMON S620 OSU Signature Graphics Tablet](https://www.amazon.co.uk/GAOMON-S620-Signature-Online-Learning-Compatible-Black/dp/B07R49KYCF/ref=sr_1_5?sr=8-5).
2. Build a structured dataset from raw HTML content.
3. Perform text preprocessing, cleaning and vocabulary creation.
4. Apply and compare different text encoding techniques.
- One-Hot Encoding.
- Bag of Words.
- TF-IDF (Term Frequency-Inverse Document Frequency)

Project workflow and analysis output is given in `REPORT.md`, specifically for the Kaggle dataset.

# Project Structure
042026 - Module 1/
├─ data/
│  ├─ 7817_1.csv
│  ├─ amazon_reviews.csv
├─ notebooks/
│  ├─ Assignment.ipynb
│  ├─ scraping.ipynb
├─ screenshots/
│  ├─ index.css
├─ README.md
├─ REPORT.md
├─ Text_Feature_Engineering_Assingment.pdf

# Datasets
- Sources:  
    - ![Amazon](https://www.amazon.co.uk/GAOMON-S620-Signature-Online-Learning-Compatible-Black/dp/B07R49KYCF/ref=sr_1_5?sr=8-5) (GAOMON S620 OSU Signature Graphics Tablet):  Dataset contains the top reviews of the product.
    - ![Kaggle](https://www.kaggle.com/code/jalesummak/amazon-reviews-topic-modeling-with-nlp-nmf-lda): This dataset is a randomly sampled corpus of Amazon product ratings and reviews. It contains nearly 1.6k customer reviews across a selection of products.

- Columns Used:
    - Amazon: `review_text` and `review_rating`
    - Amazon: `reviews_text` and `reviews_rating`


# Tech Stack
- Python, pandas, numpy, scikit-learn, nltk, Selenium

# Conclusion
This project showcases how one can transform free text data into a structured numerical feature ready for fundamental NLP machine learning tasks.  It highlights the necessity of preprocessing, feature engineering in NLP tasks as well as some of its quirks.