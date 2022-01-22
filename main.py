import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb

from naivebayes import *

if __name__ == '__main__':
    tf.keras.datasets.imdb.load_data(
        path="imdb.npz",
        num_words=None,
        skip_top=0,
        maxlen=None,
        seed=113,       # int. Seed for reproducible data shuffling.
        start_char=1,   # int. The start of a sequence will be marked with this character. Defaults to 1 because 0 is usually the padding character.
        oov_char=2,     # int. The out-of-vocabulary character. Words that were cut out bc of the num_words or skip_top limits will be replaced with this character.
        index_from=3,   # int. Index actual words with this index and higher.
        # **kwargs  # Used for backwards compatibility.
    )
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)

    tf.keras.datasets.imdb.get_word_index(path="imdb_word_index.json")



    # Retrieve the training sequences.
    (x_train, _), _ = keras.datasets.imdb.load_data()
    # Retrieve the word index file mapping words to indices
    word_index = keras.datasets.imdb.get_word_index()
    # Reverse the word index to obtain a dict mapping indices to words
    inverted_word_index = dict((i, word) for (word, i) in word_index.items())
    # Decode the first sequence in the dataset
    decoded_sequence = " ".join(inverted_word_index[i] for i in x_train[0])

    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    nb = naive_bayes()
    nb.run_id3(word_index, training_data, testing_data)
    #print(testing_targets)

    ##################################

    # pre-process dataset for training
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data()

    # shuffle dataset with sample
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    # df shape
    print(df.shape)
    # set features and target
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    # # split on train and test 0.7/0.3
    X_train, X_test, y_train, y_test = X[:100], X[100:], y[:100], y[100:]

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    ##

    # train the model
    x = naive_bayes()

    x.fit(X_train, y_train)
    predictions = x.predict(X_test)
    x.accuracy(y_test, predictions)

    y_test.value_counts(normalize=True)

    x.visualize(y_test, predictions, 'variety')

    emails = pd.read_csv("data/spambase.data")

    emails.head(2)

    emails['1'].value_counts()

    emails['spam'] = emails['1']
    emails = emails.drop(columns=['1'])
    emails['spam'] = emails['spam'].map({1: "spam", 0: "not_spam"})
    emails['spam'].value_counts()

    print(emails.shape)
    emails.head()
    X, y = emails.iloc[:, :-1], emails.iloc[:, -1]
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    model = NaiveBayesClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    y_train

    model.accuracy(y_test, preds)

    model.visualize(y_test, preds, 'spam')

    # compare to sklearn Naive Bayes Classifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    clf = GaussianNB()

    # iris dataset
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    clf.score(X_test, y_test)

    tr = pd.DataFrame(data=y_test, columns=['variety'])
    pr = pd.DataFrame(data=preds, columns=['variety'])

    fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(15, 6))

    sns.countplot(x='variety', data=tr, ax=ax[0], palette='viridis', alpha=0.7)
    sns.countplot(x='variety', data=pr, ax=ax[1], palette='viridis', alpha=0.7)

    fig.suptitle('True vs Predicted Comparison', fontsize=20)

    ax[0].tick_params(labelsize=12)
    ax[1].tick_params(labelsize=12)
    ax[0].set_title("True values", fontsize=18)
    ax[1].set_title("Predicted values", fontsize=18)
    plt.show()

    # emails dataset
    clf1 = GaussianNB()

    clf1.fit(X_train, y_train)

    preds1 = clf1.predict(X_test)
    # prediced better for emails classifications
    clf1.score(X_test, y_test)

    test_df = pd.DataFrame(data=y_test, columns=['spam'])
    pred_df = pd.DataFrame(data=preds1, columns=['spam'])

    fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(15, 6))

    sns.countplot(x='spam', data=test_df, ax=ax[0], palette='pastel', alpha=0.7)
    sns.countplot(x='spam', data=pred_df, ax=ax[1], palette='pastel', alpha=0.7)

    fig.suptitle('True vs Predicted Comparison', fontsize=20)

    ax[0].tick_params(labelsize=12)
    ax[1].tick_params(labelsize=12)
    ax[0].set_title("True values", fontsize=18)
    ax[1].set_title("Predicted values", fontsize=18)
    plt.show()



