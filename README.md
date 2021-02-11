# SVM (Support Vector Machine)

Support vector machine is a black box technique.It is imagined as a surface that creates boundary between points of data plotted in multidimensional that represents examples and their feature values.

The goal of SVM  is to create a flat boundary called hyperplane,which divides the space to create fairly homogenous partitions on either side.

SVM can be adopted for use with nearly any type of learning task,including both classification and numeric prediction.

In N dimension feature space,the dimension of decision boundary corresponds to N-1.

## SVM for non seperable data:

 This uses two concepts-
 
1)	Kernel Tricks:- 
              It utilizes existing features,applies some transformations and creates new features, this new features are key to find nonlinear decision boundary.Two most            popular kernels:-\
             a)	Polynomial Kernal\
             b)	Radial Basis Function (RBF).\
2)	Soft Margin:-\
             It is used for both linearly and non linearly seperable data.Applying soft margin,SVM tolerates few dots to get misclassified and tries to balance the trade-off between finding a line that maximizes the margin and minimizes the misclassification. 


## Data used :-
                   
                  Salary dataset :- To  Prepare a classification model using SVM for salary data.
                  Forestfires dataset :- To classify the Size_Category using SVM.
                  Letters dataset :- Letters classification

## Programming :- 
                   Python


**The Codes regarding SVM for *classification of salary data from salary dataset, classify letters from letters dataset, classify size category from forest fires dataset* are present in this Repository in detail.**



