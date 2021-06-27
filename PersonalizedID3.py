import pandas as pd
from ID3 import ID3,train_with_cross_validation,LABEL_COL,KFold,get_targets_as_ints,spacial_loss
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class PersonalizedID3(ID3):
    def __init__(self,data_size=301,par=4.2):
        if data_size is None:
            data_size=301
        if par is None:
            par = 4.2
        super().__init__(data_size,par)
        self.pca_transformer=None
        self.standard_scaler_transformer_pca = None
        self.standard_scaler_transformer_kmeans = None
        self.kmeans_transformer = None
        
    def addPCAFeatures(self,data,train=True):
        if train==True:
            self.standard_scaler_transformer_pca = StandardScaler().fit(data)
            X = self.standard_scaler_transformer_pca.transform(data)
            pca = PCA(n_components=2)
            self.pca_transformer = pca.fit(X)
            principalComponents = self.pca_transformer.transform(X)
        else:
            X = self.standard_scaler_transformer_pca.transform(data)
            principalComponents = self.pca_transformer.transform(X)
        
        return np.concatenate([data,principalComponents],axis=1)
    
    def add2MeansFeature(self,data,train=True):
        if train: 
            kmeans = KMeans(init="random",
                            n_clusters=2,
                            n_init=10,
                            max_iter=300,
                            random_state=42)
            self.standard_scaler_transformer_kmeans = StandardScaler().fit(data)
            X = self.standard_scaler_transformer_kmeans.transform(data)
            self.kmeans_transformer = kmeans.fit(X)

        else:
            X = self.standard_scaler_transformer_kmeans.transform(data)
        
        kmean_distances_from_centroid = self.kmeans_transformer.transform(X)
        kmean_predictions = self.kmeans_transformer.predict(X)
        kmean_features = np.concatenate([kmean_distances_from_centroid,kmean_predictions[np.newaxis].T],axis=1)       
        return np.concatenate([data,kmean_features],axis=1)

            
        
        
    
    def select_features(self,data):
        feature_importance = np.array([True, True, True, True,  True, True,  True, True, True,
       True, True, False,  False,  True,  True, True, True, False,
        False,  False, True,  True,  True, True, True,  True,  True,
        True, True, True,  True,False])
        data = data[:,feature_importance]
        return data
        
    def preprocess(self,data,train=True):
        if train==True:
            X = data[:,LABEL_COL+1:]
            Y = data[:,[LABEL_COL]]
        else:
            X = data
#         X = self.add2MeansFeature(X)
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


def train_with_cross_validation(data,model,data_size=None,par=None):
    kf = KFold(n_splits=5, shuffle=True,random_state=302957113)
    acc_results = []
    loss_results = []
    for train_index, test_index in kf.split(data):
        num_row, num_col = data.shape
        id3 = model(data_size,par)
        train = data[train_index]
        test = data[test_index]
        test_x = test[:,1:]
        test_labels = test[:,LABEL_COL]
        predictions = id3.fit_predict(train,test_x)
        num_correct = len(predictions[predictions==get_targets_as_ints(test_labels)])
        num_total = len(test)
        accuracy = num_correct/num_total
        acc_results.append(accuracy)
        loss = spacial_loss(predictions,get_targets_as_ints(test_labels))
        loss_results.append(loss)
        
    return acc_results,loss_results



if __name__=="__main__":
    start = time.time()
    df = pd.read_csv("train.csv",names = ["Y"]+["x_{i}".format(i=i) for i in range(30)])
#     df = pd.concat([df,principalDf],axis=1)
    data = df.to_numpy()
    results,losses = train_with_cross_validation(data,PersonalizedID3,301,4.2)
    average_accuracy = np.array(results).mean()
    average_spacial_loss = np.array(losses).mean()
    print("Trained ID3 with 5Fold Cross validation average accuracy of {}".format(average_accuracy))
    print("Trained ID3 with 5Fold Cross validation average spacial loss of {}".format(average_spacial_loss))
    end = time.time()
#     print(end-start)


# +

X= df.iloc[:,1:]
# Standardizing the features
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['Y']]], axis = 1)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['M', 'B']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Y'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
# -


