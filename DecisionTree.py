import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import pydot


class DecisionTree:

    data = pd.read_csv("finalDataset.csv").drop(
        ['Unnamed: 0', 'Name', 'District', 'Address', 'Contact', 'Homepage', 'Upvote', 'Disabled Lodging Facility',
         'Disabled Ticket Office',
         'Service for the visually-impaired', 'Service for the hearing-impaired',
         'Information Service', 'Wheelchair Rental'], axis=1)

    data = pd.DataFrame(data)

    columns = ['Main Entrance Access Road', 'Handicap Parking', 'Main Entrance Slope Way', 'Disabled Escalator',
               'Disabled Toilet', 'Disabled Seat']

    # ----- Preparação dos dados -----

    def normalizeData(self):
        data = self.data.drop(columns=['ID'])

        d = {'Y': 1, 'N': 0}
        for i in self.columns:
            data[i] = data[i].map(d)

        X = data.drop(columns='Access')
        y = data['Access']

        return X, y

    # ----- Treino do modelo -----

    def trainModel(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        dtree = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=10)
        dtree = dtree.fit(X_train, y_train)

        # ----- Calculo da acurácia -----

        y_pred = dtree.predict(X_test)
        mape = metrics.accuracy_score(y_test, y_pred)
        accuracy = 100 * mape
        cm = metrics.confusion_matrix(y_test, y_pred)

        return dtree, cm, accuracy

    # ----- Gera imagem de uma árvore -----

    def generateTree(self, dtree):
        export_graphviz(dtree, out_file='tree.dot', feature_names=self.columns, rounded=True, precision=1)
        (graph,) = pydot.graph_from_dot_file('tree.dot')
        graph.write_png('DecisionTree.png')

    # ----- Definição dos dados mais relevantes -----

    def importantData(self, dtree):
        importances = list(dtree.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(self.columns, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

        # ----- Gera gráfico dos dados mais relevantes -----

        plt.figure(figsize=(25, 20))
        plt.style.use('fivethirtyeight')
        x_values = list(range(len(importances)))
        values = plt.bar(x_values, importances, orientation='vertical')
        plt.bar_label(values)
        plt.xticks(x_values, self.columns)
        plt.ylabel('Importance')
        plt.xlabel('Variable')
        plt.title('Variable Importances')
        plt.savefig('compDT.png')

        return feature_importances


dtObject = DecisionTree()

X, y = dtObject.normalizeData()
print(X,y)
dtree, cm, accuracy = dtObject.trainModel(X, y)
dtObject.generateTree(dtree)
feature_importances = dtObject.importantData(dtree)

print('Accuracy:', round(accuracy, 2), '%.')
print('Confusion matrix\n\n', cm)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
