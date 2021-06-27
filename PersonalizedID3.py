import pandas as pd
from ID3 import ID3,train_with_cross_validation,LABEL_COL
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class PersonalizedID3(ID3):
    def __init__(self,data_size=300,par=4.2):
        super().__init__(data_size,par)
        self.pca_transformer=None
        self.standard_scaler_transformer = None
        
    def addPCAFeatures(self,data,train=True):
        if train==True:
            self.standard_scaler_transformer = StandardScaler().fit(data)
            X = self.standard_scaler_transformer.transform(data)
            pca = PCA(n_components=1)
            self.pca_transformer = pca.fit(X)
            principalComponents = self.pca_transformer.transform(X)
        else:
            X = self.standard_scaler_transformer.transform(data)
            principalComponents = self.pca_transformer.transform(X)
        
        return np.concatenate([data,principalComponents],axis=1)
    
    def select_features(self,data):
        feature_importance = np.array([False, False, False, False,  False, False,  False, False, False,
       False, False, False,  False,  False,  True, True, True, False,
        False,  False, False,  True,  True, True, False,  True,  False,
        True, False, False,  True, False])
        data = data[:,feature_importance]
        return data
        
    def preprocess(self,data,train=True):
        if train==True:
            X = data[:,LABEL_COL+1:]
            Y = data[:,[LABEL_COL]]
        else:
            X = data
        
        X = self.addPCAFeatures(X)
#         X = self.select_features(X)
        
        if train==True:
            data = np.concatenate([Y,X],axis=1)
        else:
            data = X
            
        return data
    
    def predict(self,tree,test_data):
        test_data = self.preprocess(test_data,train=False)
        return np.array([self.predict_row(tree,row) for row in test_data])
    
    def fit(self,data):
        data = self.preprocess(data)
        return super().fit(data)
    def is_pure_node(self,node):        
        return super().is_pure_node(node) or  node.num_m()/node.num_exampels()>8/9



if __name__=="__main__":
    start = time.time()
    df = pd.read_csv("train.csv",names = ["Y"]+["x_{i}".format(i=i) for i in range(30)])
#     df = pd.concat([df,principalDf],axis=1)
    data = df.to_numpy()
    results,losses = train_with_cross_validation(data,PersonalizedID3,4.2)
    average_accuracy = np.array(results).mean()
    average_spacial_loss = np.array(losses).mean()
    print("Trained ID3 with 5Fold Cross validation average accuracy of {}".format(average_accuracy))
    print("Trained ID3 with 5Fold Cross validation average spacial loss of {}".format(average_spacial_loss))
    end = time.time()
#     print(end-start)



