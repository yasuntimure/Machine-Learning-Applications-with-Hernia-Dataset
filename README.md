# Machine-Learning-Applications-with-Hernia-Dataset
Supervised, Unsupervised Analysis and Development with Hernia Dataset

Part A: 

In this part, we used a large number of visualizations using the matplotlib library. Describe the Data, Grouping the Data, Histogram, Density Plots, Box and Whisker Plots, Correlation Matrix Plot. Apart from the drawings, you can see a detailed output on the output side, up to the grouping of each data type.


Part B:

In section B we have normalized both the method and our own algorithm and applied it to all columns later .In the Outlier section, we thought of using a z score, but decided not to add it to the code because it was not successful in practice.


Part C:

We applied PCA to the data we normalized in C section. As a result, we reduced the number of columns to 2 columns and trained them on two columns. Here, we thought that falling from 6 columns to 2 columns may cause problems in data processing and cause losses. For this reason, we paid attention to the variance ratio and looked at how much we lost by collecting these values. We believe that there is no loss since we have a value of 99%.


Part D:

In Part D, we used three methods to train the ai we have. These are Gaussian NBA, Decision Tree and Linear SVM. In part B we put MinMaxScaller and different scale methods. In this part we have also written a method, is Scale, that finds its own minimum and maximum values. Then, with the normalization formula, we took each column separately and replaced them with scale states. Afterwards, we carried out the so-called test and train stages.
