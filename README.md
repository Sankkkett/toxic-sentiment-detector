# Toxic Sentiment Detection App

The Toxic Sentiment Detector is a machine learning project designed to classify text into toxic and non-toxic categories. This tool helps moderate online conversations by detecting harmful or offensive language. It employs Natural Language Processing (NLP) techniques and machine learning algorithms for sentiment analysis and classification.

## Features
- **Preprocessing of Textual Data**: Includes tokenization, cleaning, and standardization of text.
- **Sentiment Classification**: Utilizes machine learning models for accurate classification.
- **Visualization**: Provides graphical insights into data distribution and model performance.
- **User-Friendly Deployment**: Accessible via a Streamlit-based web interface.
- **Toxicity Detection**: Classifies text as "Toxic" or "Non-toxic."
- **Interactive UI**: User-friendly interface built with Streamlit.
- **Custom Styling**: Enhanced visual presentation with CSS.

### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- Required Python libraries (listed in `requirements.txt`)
- Pre-trained model and vectorizer files:
  - `Toxicity_Sentiment_model.pkl`
  - `tf_idf.pkl`

## Technologies Used

### Programming Language
- **Python**: Core language for development.

### Libraries
- **Pandas** and **NumPy**: For data manipulation and analysis.
- **NLTK** and **Spacy**: For Natural Language Processing tasks.
- **Scikit-learn**: For building and evaluating machine learning models.
- **Matplotlib** and **Seaborn**: For data visualization.
- **Streamlit**: For deploying the web application.

## Dataset Used
- **Source**: [https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset)]
- **Description**: Contains labeled text examples categorized as toxic or non-toxic.

## Steps Performed

1. **Data Loading and Exploration**
   - Loaded and explored the dataset to understand its structure and label distribution.

2. **Data Preprocessing**
   - Cleaned the text by removing:
     - Stop words
     - Punctuation
     - Special characters
   - Tokenized and lemmatized the text for standardization.

3. **Feature Engineering**
   - Applied techniques such as:
     - TF-IDF (Term Frequency-Inverse Document Frequency)
     - Bag of Words

4. **Model Building and Evaluation**
   - Trained various machine learning models, including:
     - Logistic Regression
     - Random Forest
   - Evaluated models using metrics such as:
     - Accuracy
     - F1-Score
   - Selected the best-performing model.

5. **Visualization**
   - Created visualizations for:
     - Data distribution
     - Model performance

6. **Deployment**
   - Built a web application using Streamlit to allow users to:
     - Input text
     - View classification results
    


## How to Run

### Clone the Repository
```bash
git clone https://github.com/Sankkkett/toxic-Sentiment-detector.git
cd Toxic-Sentiment-Detector
```

### Install the Required Libraries
```bash
pip install -r requirements.txt
```

### Run the Streamlit App
```bash
streamlit run app.py
```
Or:
```bash
python -m streamlit run app.py
```

### Open the provided URL in your browser to use the application.



## Future Work
- Integrate deep learning models like LSTMs or Transformers for improved performance.
- Expand the dataset to include diverse and real-world text examples.
- Add multilingual support for analyzing text in various languages.

## Author
- **Name**: Sanket Pawar
- **Contact**: [sanketpawar24112001@gmail.com](mailto:sanketpawar24112001@gmail.com)
- **LinkedIn**: [Add LinkedIn profile link]

Feel free to fork the repository or contribute to its development!
