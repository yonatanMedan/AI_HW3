import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
LABEL_COL = 0
class Node:
    def __init__(self, data:np.ndarray):
        self.data:pd.DataFrame = data
        self.classes = np.unique(data[:,LABEL_COL])
        self.children:list[Node] = []
        self.value = None
        self.feature = None
        self.feature_criterion = None
    
    def num_exampels(self):
        return len(self.data)
    
    def num_m(self):
        target = self.data[:,0]
        return len(target[target=="M"])

    def add_child(self,child):
        self.children.append(child)
        
    def calc_entropy(self):
        entropy = 0
        len_data = self.num_exampels()
        for cls in self.classes:
            num_in_class = len(self.data[self.data[:,LABEL_COL]==cls])
            class_prob = num_in_class/len_data
            class_prob_log = math.log2(class_prob)
            if class_prob ==0:
                class_prob_log = 0
            entropy-= class_prob*class_prob_log
        return entropy
        
    def __str__(self):
        return self.value if self.value is not None else self.children




def get_targets_as_ints(labels):
    return (labels=="M").astype(int)


class ID3:
    def __init__(self,data_size=-1,par=-1):
        self.data_size = data_size
        self.par = par

    def fit_predict(self,data,test_data):
        tree = self.fit(data)
        return get_targets_as_ints(self.predict(tree, test_data))
    
    def fit(self,data:np.ndarray):
        tree = Node(data)
        self.grow(tree)
        return tree

    def get_pruned_val(self, data:np.ndarray):
        unique, counts = np.unique(data, return_counts=True)
        max_index = np.argmax(counts)
        return unique[max_index]

    def grow(self,node:Node):
        if self.is_pure_node(node):
            node.value = node.data[0,LABEL_COL]
            return
        if self.is_small_node(node):
            pruned_data = node.data[:,LABEL_COL]
            node.value = self.get_pruned_val(pruned_data)
            return
        feature_index,val = self.choose_feature(node)
        node.feature = feature_index
        node.feature_criterion = val
        left_node,right_node = self.split_by_feature(node,feature_index,val)
        self.grow(left_node)
        self.grow(right_node)
        node.children.append(left_node)
        node.children.append(right_node)
        
    def predict_row(self,tree,data_row):
        node = tree
        while node.feature is not None:
            #minus 1 as target is not included in data_row
            if data_row[node.feature-1]<node.feature_criterion:
                node = node.children[0]
            else:
                node = node.children[1]
            
        return node.value
    
    def predict(self,tree,test_data):
        return np.array([self.predict_row(tree,row) for row in test_data])


    def choose_feature(self,node):
        max_ig = 0
        choosen_val = None
        choosen_feature = None
        for feature in range(1,node.data.shape[1]):
            ig,val = self.calc_ig_continues(node,feature)
            if ig>=max_ig:
                max_ig = ig
                choosen_val = val
                choosen_feature = feature
                
        return choosen_feature, choosen_val
    
    def calc_ig_continues(self,node,feature_index):
        intermediat_values = self.get_intermediate_values(node.data,feature_index)
        max_ig = -1
        best_val = None
        parent_entropy = node.calc_entropy()
        for value in intermediat_values:
            child_nodes = self.split_by_feature(node,feature_index,value)
            ig = parent_entropy
            for child in child_nodes:
                ig-=(child.num_exampels()/node.num_exampels())*child.calc_entropy()
                
            if ig >= max_ig:
                max_ig = ig
                best_val = value
            
        return max_ig, best_val
    
    def get_intermediate_values(self,data,feature_index):
        feature_col_sorted = np.sort(data[:,feature_index])
        intermediat_values = (feature_col_sorted[:-1]+feature_col_sorted[1:])/2
        return intermediat_values
    
    def split_by_feature(self,node,feature_index,val):
        assert(feature_index!=0)
        data_left = node.data[node.data[:,feature_index]<val]
        data_right = node.data[node.data[:,feature_index]>=val]
        left_node = Node(data_left)
        right_node = Node(data_right)
        return left_node,right_node
    
    def is_pure_node(self,node):
        target = node.data[:,0]
        return (target==target[0]).all()

    def is_small_node(self,node:Node):
        if self.par<0 or self.par > 100:
            return False
        num_row, num_col = node.data.shape
        min_node_num = self.data_size * self.par / 100
        if num_row >= min_node_num:
            return False
        return True


def train_with_cross_validation(data,model,par=-1):
    kf = KFold(n_splits=5, shuffle=True,random_state=302957113)
    acc_results = []
    loss_results = []
    for train_index, test_index in kf.split(data):
        num_row, num_col = data.shape
        id3 = model(num_row,par)
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

# predicting M=1 where labels is B=0
def count_false_positive(predictions,labels):
    mask = (labels==0)&(predictions==1)
    return len(predictions[mask])
# predicting B=0 where labels is M=1
def count_false_negative(predictions,labels):
    mask = (labels==1)&(predictions==0)
    return len(predictions[mask])
# Expecting predictions and labels to be ints (1=M 0=B)
def spacial_loss(predictions, labels):
    return count_false_positive(predictions,labels)+8*count_false_negative(predictions,labels)


def calc_spacial_loss_on_all_M_prediciton(data):
    labels = get_targets_as_ints(data[:,LABEL_COL])
    predictions = np.ones(len(labels))
    return spacial_loss(predictions,labels)


if __name__=="__main__":
    id3 = ID3()
    df = pd.read_csv("train.csv",names = ["Y"]+["x_{i}".format(i=i) for i in range(30)])
    data = df.to_numpy()
    for i in range(0,100,1):
        results, losses = train_with_cross_validation(data, ID3,i)
        average_accuracy = np.array(results).mean()
        average_spacial_loss = np.array(losses).mean()
        print("Trained ID3 with 5Fold Cross validation and par = {} average accuracy of {}".format(str(i),average_accuracy))
        print("Trained ID3 with 5Fold Cross validation average spacial loss of {}".format(average_spacial_loss))
        print("ALL M naive classifier spacial loss: {}".format(calc_spacial_loss_on_all_M_prediciton(data)))



