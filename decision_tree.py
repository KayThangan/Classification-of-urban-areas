"""
This module is a program that simulates a decision tree classification
of the urban areas of the city of Rome.
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from matplotlib import pyplot as plt

# loading dataset
data = pd.read_csv("data.csv")
#split dataset in features and target variable
feature_cols = pd.read_csv("data.csv", index_col=1, nrows=0).columns.tolist()
feature_cols.remove("most_present_age")
feature_cols.remove("cell_id")
feature_cols.remove("Roads:total")
feature_cols.remove("Buildings:total")
feature_cols.remove("pois:total")
feature_cols.remove("ThirdPlaces:total")
X = data[feature_cols]  # Features
y = data.most_present_age  # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifier object
# decision_tree = DecisionTreeClassifier()
decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
# Train Decision Tree Classifier
decision_tree = decision_tree.fit(X_train, y_train)

# displaying the decision tree
tree_text_representation = tree.export_text(decision_tree, feature_names=feature_cols)
print(tree_text_representation)

# Predict the response for test dataset
y_pred = decision_tree.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

figure = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(
    decision_tree,
    feature_names=feature_cols,
    class_names="most_present_age",
    filled=True,
)
figure.savefig("decision_tree.png")
figure.show()
