from DecisionTree import *
import pandas as pd
import random
from sklearn import model_selection

header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
#header = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', header=None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
lst = df.values.tolist()
t = build_tree(lst, header, 0, 0)   #Assign-2 Added ID and Depth while building the tree.
print_tree(t)

print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header, 0, 0)                 #Assign-2 Added ID and Depth while building the tree.
print("*************Tree before pruning*******")
print_tree(t)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))

# Code to store all the node ID's in a list named, id_list
id_list = listID(t)
# Code to take a random sample of 2 ID's from the list created above
random_node_id = random.sample(id_list, 2)

## TODO: You have to decide on a pruning strategy
t_pruned = prune_tree(t, random_node_id)                    #Pruning two nodes randomly

print("*************Tree after pruning nodes " + str(random_node_id) + " randonly*******")
print_tree(t_pruned)
acc = computeAccuracy(test, t_pruned)
print("Accuracy on test = " + str(acc))

