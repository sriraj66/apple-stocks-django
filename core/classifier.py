import pandas as pd
import pickle
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class ModelClassifier:
    def __init__(self):

        self.fields = ['Volume']
        self.pkl_path = 'model/stock.pkl'
        self.data = pd.read_csv("data/appledata.csv")

        self.x,self.y = self.ProcessData()
        self.model = None
        self.x_train, self.x_test,self.y_train,self.y_test = self.TrainData()
    
    def TrainData(self):
        return ( train_test_split(self.x,self.y , test_size = 0.9))


    def ProcessData(self):
        return (self.data.drop(self.fields,axis=1),self.data[self.fields[0]])
    
    def TrainModel(self):
        model = LogisticRegression()
        model.fit(self.x_train,self.y_train)

        predet = model.predict(self.x_test)
        print("Accuracy : ",metrics.accuracy_score(self.y_test,predet))

        pickle.dump(model, open(self.pkl_path,'wb'))
        self.model = model

    def PredictModel(self,array):
        model = pickle.load(open(self.pkl_path,'rb'))

        res = model.predict(np.array([array]))
        # print('Accuraccy : ',metrics.accuracy_score(self.y_test, res))
        return res

if __name__ == '__main__':
    model = ModelClassifier()
    model.TrainModel()

    test = model.PredictModel([7,150.72,146.43,151.48,146.15])
    print(test)