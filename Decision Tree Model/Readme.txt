Language used is Python 3.x

Steps to compile:
1> Download the two files namely, DecisionTree.py and Driver.py.
2> Open python command line interface (or Anaconda command prompt or PyCharm) at the location where the python files (name: Driver.py and DecisionTree.py) are located.

3> type the following command to run the code:

	python Driver.py

4> Please give it some time to print the tree and stats


Note -- If you want to change the Dataset to be used.
1> Edit Driver.py file
2> Change the dataset url and header below :-
	header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])

