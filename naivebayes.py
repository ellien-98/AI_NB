import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


class naive_bayes():
    '''
    Bayes Theorem form
    P(y|X) = P(X|y) * P(y) / P(X)
    '''

    def calc_prior(self, features, target):
        '''
        prior probability P(y)
        calculate prior probabilities
        '''
        self.prior = (features.groupby(target).apply(lambda x: len(x)) / self.rows).to_numpy()

        return self.prior

    def calc_statistics(self, features, target):
        '''
        calculate mean, variance for each column and convert to numpy array
        '''
        self.mean = features.groupby(target).apply(np.mean).to_numpy()
        self.var = features.groupby(target).apply(np.var).to_numpy()

        return self.mean, self.var

    def calc_posterior(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for i in range(self.count):
            prior = np.log(self.prior[i])  # use the log to make it more numerically stable
            conditional = np.sum(np.log(self.gaussian_density(i, x)))  # use the log to make it more numerically stable
            posterior = prior + conditional
            posteriors.append(posterior)
        # return class with highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def fit(self, features, target):
        # classes == 0 or 1
        self.classes = np.unique(target)
        self.count = len(self.classes)
        self.feature_nums = features.shape[1]
        self.rows = features.shape[0]

        self.calc_statistics(features, target)
        self.calc_prior(features, target)

    def predict(self, features):
        preds = [self.calc_posterior(f) for f in features.to_numpy()]
        return preds

    def accuracy(self, y_test, y_pred):
        accuracy = np.sum(y_test == y_pred) / len(y_test)
        return accuracy

    # for diagrams
    def visualize(self, y_true, y_pred, target):
        tr = pd.DataFrame(data=y_true, columns=[target])
        pr = pd.DataFrame(data=y_pred, columns=[target])

        fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(15, 6))

        sns.countplot(x=target, data=tr, ax=ax[0], palette='viridis', alpha=0.7, hue=target, dodge=False)
        sns.countplot(x=target, data=pr, ax=ax[1], palette='viridis', alpha=0.7, hue=target, dodge=False)

        fig.suptitle('True vs Predicted Comparison', fontsize=20)

        ax[0].tick_params(labelsize=12)
        ax[1].tick_params(labelsize=12)
        ax[0].set_title("True values", fontsize=18)
        ax[1].set_title("Predicted values", fontsize=18)
        plt.show()


    def gaussian_density(self, class_idx, x):
        '''
                    calculate probability from gaussian density function (normally distributed)
                    we will assume that probability of specific target value given specific class is normally distributed

                    probability density function derived from wikipedia:
                    (1/√2pi*σ) * exp((-1/2)*((x-μ)^2)/(2*σ²)), where μ is mean, σ² is variance, σ is quare root of variance (standard deviation)
                    '''
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp((-1 / 2) * ((x - mean) ** 2) / (2 * var))
        #         numerator = np.exp(-((x-mean)**2 / (2 * var)))
        denominator = np.sqrt(2 * np.pi * var)
        prob = numerator / denominator
        return prob

'''
                def gaussian_density(self, class_idx, x):
            
            calculate probability from gaussian density function (normally distributed)
            we will assume that probability of specific target value given specific class is normally distributed

            probability density function derived from wikipedia:
            (1/√2pi*σ) * exp((-1/2)*((x-μ)^2)/(2*σ²)), where μ is mean, σ² is variance, σ is quare root of variance (standard deviation)
            # 
            mean = self.mean[class_idx]
            var = self.var[class_idx]
            numerator = np.exp((-1 / 2) * ((x - mean) ** 2) / (2 * var))
            #         numerator = np.exp(-((x-mean)**2 / (2 * var)))
            denominator = np.sqrt(2 * np.pi * var)
            prob = numerator / denominator
            return prob
'''
