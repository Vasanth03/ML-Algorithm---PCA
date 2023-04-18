# ML-Algorithm---PCA
PCA is applied in pharmaceuticals dataset to study the accuracy of the model.
>> A pharmaceuticals manufacturing company is conducting a study on a new medicine to treat heart diseases. The company has gathered data from its secondary sources and would like you to provide high level analytical insights on the data. Its aim is to segregate patients depending on their age group and other factors given in the data. Perform PCA and clustering algorithms on the dataset and check if the clusters formed before and after PCA are the same and provide a brief report on your model. You can also explore more ways to improve your model. 
#### PCA: Principal Component Analysis is techniques used for reducing the dimensionality of data. It increases interpretability with minimal information loss,  captures most of the variances of the data.

1. Normalise or standardise the variables.
2. Compute the covariance matrix to identify correlation.
3. Compute eigenvectors and eigenvalues of the covariance matrix to identify the principal components.
4. Create a feature vector to decide which principal components to retain.                    
5. Principal components are new variables that are constructed as linear combinations or mixtures of the initial variables.

* Initially for 14 features in the dataset, we select 14 components and the cumulative variance of each is as follows 
33.06496383,  48.34957274,  58.87516129,  68.69543478, 75.77361519,  80.90605998,  85.89819375,  89.7578116 , 92.43267287,  94.93103747,  96.74681062,  98.04079758, 99.16599012, 100.    

* PCA is applied and 10 components are selected as given in the statement, thus the cumulative variance will be 94.93 %.

* For the first ten components, hierarchical and K-Means is applied, with the same threshold the clusters for hierarchical clustering becomes 8 from 10.
