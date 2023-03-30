import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

def RFECV_RandomForest(x_train, y_train, step):
    y = np.reshape(y_train, (-1))
    x = np.reshape(x_train, (x_train.shape[0], -1))

    min_features_to_select=1
    # Init the transformer
    rfe = RFECV(estimator=RandomForestClassifier(), step=step, cv=3, min_features_to_select = min_features_to_select, verbose=1)

    # Fit to the training data
    rfe.fit(x, y)

    print("Feature ranking: ", rfe.ranking_)
    return rfe.ranking_

##

"""
predicte = forest.predict(x_test)
predicte = scaler.inverse_transform(np.reshape(predicte,(-1,1)))

len_x_test=len(x)

close = data.filter(['close'])
validation = close[int(len_x_test*0.7):len_x_test]

validation['Predictions'] = np.reshape(predicte,(-1))

import matplotlib.pyplot as plt
plt.figure(figsize=(14,7))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')

plt.plot(validation[['close', 'Predictions']])
plt.legend(['Val réelle', 'Predictions'], loc='lower right')

plt.grid()
plt.show()
"""

"""
print(f"Optimal number of features: {rfe.n_features_}")

n_scores = len(rfe.cv_results_["mean_test_score"])

import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.errorbar(
    range(min_features_to_select, n_scores + min_features_to_select),
    rfe.cv_results_["mean_test_score"],
    yerr=rfe.cv_results_["std_test_score"],
)
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()
"""

