import pandas as pd
from ID3 import ID3,train_with_cross_validation
import numpy as np


class PersonalizedID3(ID3):
    def is_pure_node(self,node):        
        return super().is_pure_node(node) or node.num_m()/node.num_exampels()>0.92


if __name__=="__main__":
    df = pd.read_csv("train.csv",names = ["Y"]+["x_{i}".format(i=i) for i in range(30)])
    data = df.to_numpy()
    results,losses = train_with_cross_validation(data,PersonalizedID3)
    average_accuracy = np.array(results).mean()
    average_spacial_loss = np.array(losses).mean()
    print("Trained ID3 with 5Fold Cross validation average accuracy of {}".format(average_accuracy))
    print("Trained ID3 with 5Fold Cross validation average spacial loss of {}".format(average_spacial_loss))


# %debug


