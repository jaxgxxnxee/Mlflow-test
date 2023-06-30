from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from joblib import dump, load
import json
import sys

## Load Dataset
random_state = int(sys.argv[1]) if len(sys.argv) > 1 else 0
iris_data = datasets.load_iris()
target_index = {i: target for i, target in enumerate(iris_data['target_names'])}
X, y = iris_data.data, iris_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

## Train Dataset
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

accuracy = dict(ac_sc=accuracy_score(y_test, y_predict),
                rc_sc=recall_score(y_test, y_predict, average="weighted"),
                pr_sc=precision_score(y_test, y_predict, average="weighted"),
                f1_sc=f1_score(y_test, y_predict, average='micro')
                )

## Save Model
dump(clf, 's3/iris_classification.joblib')

### save etc
with open('s3/target_index.json', 'w') as f:
    json.dump(target_index, f)

with open('s3/accuracy.json', 'w') as f:
    json.dump(accuracy, f)
