# Word Vector Analysis Project

## Project Overview

This project explores word vectors derived from the word2vec algorithm, examining the relationships between words captured via these vectors. The project includes finding similar words, solving word analogies, and using these vectors as features for a classification task. The primary dataset consists of Wikipedia biographies, and we leverage pre-computed word vectors to enhance our analyses.

## Table of Contents

1. [Project Description](#project-description)
2. [Data Import and Preparation](#data-import-and-preparation)
3. [Loading Word Vectors](#loading-word-vectors)
4. [Examining Word Vectors](#examining-word-vectors)
5. [Using Word Vectors as Classifier Features](#using-word-vectors-as-classifier-features)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Setup Instructions](#setup-instructions)
8. [Key Results](#key-results)
9. [Contact Information](#contact-information)

---

## Project Description

This project aims to explore and utilize word vectors to perform various NLP tasks, including finding similar words and solving analogies. Additionally, we use these word vectors as features in a nationality-classifier task.

## Data Import and Preparation

- **Data Source:** Pre-computed word vectors and Wikipedia biography dataset.
- **Data Cleaning:** Standardize and filter out infrequent labels.

## Loading Word Vectors

The project starts by loading pre-computed word vectors stored in a dictionary format (`wiki_wv`).

### Task 1.1: Load Pre-computed Word Vectors
```python
wiki_wv = {}
with open(r'C:/users/akama/downloads/word_vector_analysis/wiki_vocab.txt', encoding='utf-8') as f:
    wiki_wv['vocab'] = f.read().split('\n')
wiki_wv['wv'] = np.load(r'C:/users/akama/downloads/word_vector_analysis/wiki_vects.npy')
```

### Basic Facts about Our Word Vectors
```python
n_words = len(wiki_wv['wv'])
n_dims = wiki_wv['wv'].shape[1]
print(f'{n_words} words and {n_dims} dimensions')
```

### Task 1.2: Construct a Word Index
```python
word_index = dict(zip(wiki_wv['vocab'], range(len(wiki_wv['vocab']))))
wiki_wv['index'] = word_index
```

### Task 1.2.1: Get the Vector for a Given Word
```python
def get_word_vector(word, wv=wiki_wv):
    word_idx = wv['index'].get(word)    
    if word_idx is not None:
        return wv['wv'][word_idx]
    else:
        return None
```

## Examining Word Vectors

### Task 2.1: Get Similar Words
Define the `get_most_similar` function to find words similar to a given word:
```python
def get_vector_cossims(vect, wv=wiki_wv):
    return cosine_similarity(np.array([vect]), wv['wv']).flatten()

def get_most_similar(word, wv=wiki_wv, k=10):
    word_vector = get_word_vector(word, wv)
    if word_vector is None:
        return []
    cossims = get_vector_cossims(word_vector, wv)
    word_idx = wv['index'].get(word)
    if word_idx is not None:
        cossims[word_idx] = -1
    most_similar_indices = np.argsort(cossims)[-k:][::-1]
    most_similar_words = [wv['vocab'][idx] for idx in most_similar_indices]
    return most_similar_words
```

### Task 2.2: Examine Word Analogies
Define the `get_analogy` function:
```python
def normalize(vect):
    return vect / np.linalg.norm(vect)

def get_analogy(a, b, c, wv=wiki_wv):
    a_vec = get_word_vector(a, wv)
    b_vec = get_word_vector(b, wv)
    c_vec = get_word_vector(c, wv)
    if a_vec is None or b_vec is None or c_vec is None:
        return None
    norm_a = normalize(a_vec)
    norm_b = normalize(b_vec)
    norm_c = normalize(c_vec)
    d_prime = norm_b - norm_a + norm_c
    cossims = get_vector_cossims(d_prime, wv)
    for word in [a, b, c]:
        word_idx = wv['index'][word]
        cossims[word_idx] = -1
    most_similar_idx = np.argmax(cossims)
    most_similar_word = wv['vocab'][most_similar_idx]
    return most_similar_word
```

## Using Word Vectors as Classifier Features

### Task 3.1.1: Filter out Infrequent Labels
Filter the rows in `nationality_df` for labels occurring at least 500 times:
```python
nationality_df = pd.read_csv('../../assets/bio_name_nationality.tsv.gz', sep='\t', compression='gzip')
nationality_df = nationality_df.dropna()
nationality_df = nationality_df[:75000]

def standardize_nationality(nationality):
    parts = nationality.split(',')
    standardized_label = parts[-1].strip()
    return standardized_label

nationality_df['nationality'] = nationality_df['nationality'].apply(standardize_nationality)
cleaned_nationality_df = nationality_df[nationality_df.groupby('nationality')['nationality'].
                                        transform('count')>=MIN_NATIONALITY_COUNT]
print(len(cleaned_nationality_df), cleaned_nationality_df.nationality.nunique())
```

### Task 3.1.2: Create Train/Dev/Test Data Splits
Split the cleaned data into train, development, and test sets:
```python
train_dev_df, test_df = train_test_split(
    cleaned_nationality_df, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True
)

train_proportion_in_train_dev = TRAIN_SIZE / (TRAIN_SIZE + DEV_SIZE)

train_df, dev_df = train_test_split(
    train_dev_df, test_size=(1 - train_proportion_in_train_dev), random_state=RANDOM_SEED, shuffle=True
)

print(len(train_df), len(dev_df), len(test_df))

y_train = list(train_df.nationality)
y_dev = list(dev_df.nationality)
y_test = list(test_df.nationality)
```

### Task 3.2: Tokenize Text and Remove Stopwords
Tokenize each biography and remove stopwords:
```python
stop_words = ENGLISH_STOP_WORDS
tokenized_train_items = []
tokenized_dev_items = []

stop_words = set(ENGLISH_STOP_WORDS)
token_pattern = re.compile(r'(?u)\b\w\w+\b')

def tokenize_and_remove_stopwords(text, token_pattern, stop_words):
    tokens = token_pattern.findall(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

tokenized_train_items = [tokenize_and_remove_stopwords(bio, token_pattern, stop_words) for bio in train_df['bio']]
tokenized_dev_items = [tokenize_and_remove_stopwords(bio, token_pattern, stop_words) for bio in dev_df['bio']]

print(len(tokenized_train_items[0]), len(tokenized_dev_items[0]))
```

### Task 3.3: Compute Word Vector-Based Features
Compute the mean word vector for each document:
```python
def generate_word_vector_features(tokenized_texts, wv=wiki_wv):
    vocab = set(wv['vocab'])
    word_vectors = wv['wv']
    index = wv['index']
    features = np.zeros((len(tokenized_texts), word_vectors.shape[1]))
    for i, tokens in enumerate(tokenized_texts):
        valid_vectors = [word_vectors[index[token]] for token in tokens if token in vocab]
        if valid_vectors:
            mean_vector = np.mean(valid_vectors, axis=0)
        else:
            mean_vector = np.zeros(word_vectors.shape[1])
        features[i] = mean_vector
    return features

# Test with a small subset
X_sample = generate_word_vector_features(tokenized_train_items[:200], wiki_wv)
print(X_sample.shape)
print(X_sample)
```

### Task 3.3.1: Compute Features for the Entire Data
Generate word-vector-based features for the train and dev sets:
```python
X_train_wv = generate_word_vector_features(tokenized_train_items, wiki_wv)
X_dev_wv = generate_word_vector_features(tokenized_dev_items, wiki_wv)
print(X_train_wv.shape, X_dev_wv.shape)
```

### Task 3.4: Train and Evaluate Classifier
Train a Logistic Regression classifier and evaluate its performance:
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('clf', LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, solver='lbfgs', n_jobs=-1))
])

param_dist = {
    'clf__C': uniform(loc=0, scale=4),
}

random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=20, cv=3, scoring='f1_macro', n_jobs=None, random_state=RANDOM_SEED, verbose=2)
random_search.fit(X_train_wv, y_train)
best_model = random_search.best_estimator_
y_pred_dev = best_model.predict(X_dev_wv)
lr_wv_f1 = f1_score(y_dev, y_pred_dev, average='macro')

print(lr_wv_f1)
```

### Task 3.5: Consider Model Size
Compute the number of feature weights required for both tf-idf and word vector features:
```python
vectorizer = TfidfVectorizer(min_df=500, stop_words='english')
X_train = vectorizer.fit_transform(train_df.bio)
num_tfidf_features = X_train.shape[1]
num_wv_features = X_train_wv.shape[1]
num_classes = len(set(y_train))
num_tfidf_feature_weights = num_tfidf_features * num_classes
num_wv_feature_weights = num_wv_features * num_classes
print('num feature weights for tfidf', num_tfidf_feature_weights)
print('num feature weights for word vects', num_wv_feature_weights)
```

## Setup Instructions

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/akamal341/word-vector-analysis.git
    cd word-vector-analysis
    ```

2. **Install Dependencies**:
    ```sh
    pip install pandas numpy scikit-learn matplotlib tqdm
    ```

3. **Run the Jupyter Notebook**:
    Open and execute the provided Jupyter Notebook `word_vector_analysis_project.ipynb` to walk through the data analysis, feature engineering, model training, and evaluation.

## Key Results

1. **Analogies:** Successfully identified relationships such as capital cities and professional fields.
2. **Most Similar Words:** Identified words similar to a given word based on cosine similarity.
3. **Classifier Performance:** Logistic Regression model using word vectors achieved an F1 score of 0.727.

## Contact Information

For any questions or further information, please contact:
- **Name:** Asad Kamal
- **Email:** aakamal {/@/} umich {/dot/} edu
- **LinkedIn:** [LinkedIn Profile](https://linkedin.com/in/asadakamal)
