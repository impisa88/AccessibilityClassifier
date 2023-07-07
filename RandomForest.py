import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
from matplotlib import pyplot as plt


class RandomForest:

    data = pd.read_csv('finalDataset.csv').drop(
        ['Unnamed: 0', 'Name', 'District', 'Address', 'Contact', 'Homepage', 'Upvote', 'Disabled Lodging Facility',
         'Disabled Ticket Office',
         'Service for the visually-impaired', 'Service for the hearing-impaired',
         'Information Service', 'Wheelchair Rental', 'ID'], axis=1)
    columns = ['Main Entrance Access Road', 'Handicap Parking', 'Main Entrance Slope Way', 'Disabled Escalator',
               'Disabled Toilet', 'Disabled Seat']

    # ----- Preparação dos dados -----

    def normalizeData(self):

        d = {'Y': 1, 'N': 0}
        for i in self.data.columns.drop('Access'):
            self.data[i] = self.data[i].map(d)

        labels = np.array(self.data['Access'])
        data = self.data.drop(columns='Access')
        data_list = list(data.columns)
        data = np.array(data)

        return data, labels, data_list

    # ----- Treino do modelo -----

    def trainModel(self, data, labels):

        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.25,
                                                                            random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(train_data, train_labels)
        predictions = rf.predict(test_data)
        errors = abs(predictions - test_labels)

        # ----- Calculo da acurácia -----

        mape = rf.score(test_data, test_labels)
        accuracy = 100 * mape

        return rf, accuracy

    # ----- Gera imagem de uma árvore da floresta -----

    def generateTree(self, rf, data_list):

        tree = rf.estimators_[10]
        export_graphviz(tree, out_file='RandomForest.dot', feature_names=data_list, rounded=True, precision=1)
        (graph,) = pydot.graph_from_dot_file('RandomForest.dot')
        graph.write_png('RandomForest.png')

    # ----- Definição dos dados mais relevantes -----

    def importantData(self):

        importances = list(rf.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(data_list, importances)]
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
        plt.savefig('compRF.png')

        return feature_importances


rfObject = RandomForest()
data, labels, data_list = rfObject.normalizeData()
rf, accuracy = rfObject.trainModel(data, labels)
rfObject.generateTree(rf, data_list)
feature_importances = rfObject.importantData()

print('Accuracy:', round(accuracy, 2), '%.')
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
