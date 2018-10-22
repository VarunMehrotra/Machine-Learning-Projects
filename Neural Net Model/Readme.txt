            Neural Network Report
            
            
1)	We have provided the summary of the results we obtained with various parameters in results.xml and attaching screenshot of the same here. 
We have used the Iris dataset. We ran different iterations using Sigmoid, relu and tanh activation function.

2)	Assumptions and Simplifications:
a.	last column will be the class label column
b.	training and test datasets will have the same format 
c.	We haven't implemented regularization, adaptive learning rate, or momentum factors.

3)	Preprocessing:
a.	We have replaced '?' from the data with NaN values and removed all the duplicate rows.
b.	We have used labeEncoder to convert all the categorical valued attribute to numerical value and normalized the data and scaled it using MinMaxScaler.
c.	Converted numpy array to dataframe.
d.	Data is split into test and train. 80% train and 20% test

4)	We tried various parameter for learning rate and number of iterations and put the best result obtained in report.



5)	Observations: 
Among all the activation function, sigmoid function performed best. Tanh and Sigmoid activation performs almost similar. Sigmoid output is not zero-centric but Tanh output are zero centric. I think the reason sigmoid performed better is because the data points are centered between 0 to 1. Also, ReLU tends to blow up activation as there is no mechanism to constrain the output of the neuron, as "X" itself is the output.
