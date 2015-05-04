# Supervised
## Continuous Y (Regression)
* Linear Regression (parametric) Challenges: 
	+ Non-linearity of true f(x)
	+ Correlation of error terms
	+ Non-constant variance of error terms
	+ Outliers
	+ High leverage points
	+ Collinearity of predictors


## Classification
* Logistic Regression: probability Y belongs t a particular category. Generally used for two class problems. Produces linear decision boundary. Can outperform LDA if Gaussian assumptions are not met.
* Linear Discriminant Analysis/Quadratic Discriminant Analysis: Often used for multi-class problems because parameter estimates in logistic regression can be unstable. Produces linear decision boundary. parametric. Requires Gaussian distribution of observations and a common covariance matrix.
	+ Confusion matrix
	+ Sensitivity and Specificity and the ROC curve: True positive rate vs. false positive rate

* Quadratic Discriminant Analysis: allows each class to have its own covariance matrix. parametric. Can work if decision boundary is somewhat non-linear and if there are limitations on the number of training observations.
* K-Nearest Neighbor. non-parametric, so does not show which predictors are. Best if decision boundary is highly non-linear. Critical to select the right level of smoothness. May not perform as well if p is large

## Resampling

* Cross Validation helps estimate test error. Might be used to tune model to lowest est test MSE.
	+ Validation Set Approach might be used with repeated sampling-but results will vary
	+ Leave One Out Cross-Validation: Less bias than Validation Set but can be computationally expensive except for least squares linear regression models.
	+ k-Fold Cross-Validation. Less computationally expensive than LOOCV. Typically k=5 or k=10 to balance bias and variability

* Bootstrap: repeatedly draw samples with replacement from the same data set.

## Linear Model Selection and Regularization

* Subset selection (pick the subset of p predictors that is most related to the response)
	+ Best subset selection- consider all combinations of p predictors
	+ Stepwise selection (very large p) add or subtract 1 predictor at each step and keep the model that gives most improvement. May be forward (add 1 at a time) or backward
	+ Hybrid: Add strongest and remove weakest. Similar to best but less computation

* Shrinkage (fit the model with all p predictors but shrink estimated coefficients to reduce variance)
	+ Ridge Regression: minimize RSS + "shrinkage penalty" (lambda) applied to other coefficients. Disadvantage is it include all parameters in the final model
	+ The Lasso similar to Ridge, but some parameters are forced to 0
* Optimal Model Selection: Requires estimate of test error

	+ Cp, Akaike Information Criterion (AIC) and Baysian...(BIC)
	+ Adjusted R-squared: Less theoretically sound
	+ Validation/Cross Validation

* Dimension reduction (use M-dimensional subspace of p predictors to fit models) useful when p is large relative to n.
	+ Principal Components Regression
	+ Partial Least Squares. Supervised


## High-Dimensional Problems (p>n)

* Subset and shrinkage methods can be used.
* Multicollinearity is a big problem: resulting predictors are not necessarily "the right ones"
* Never use calculated statistical measures of test error, use test set or cross-validation.

## Non-Linear Methods

* Polynomial Regression: linear model with polynomial terms. f(x) is global
* Step Functions: break X into bins C1 to Cn. Linear Regression with dummy variables based on bin.
* Basis Functions: generalizes the first two methods. b1 through bK basis functions. Each can be polynomial, step, or another function.
* Regression Splines: fit low degree polynomials over different regions of X. A knot K is a break in the range of X. Fits K+1 models. Often give more stable estimates than polynomial regression.
	+ Constrain the function so: 1) The whole function must be continuous and 2) the first and second derivatives of the piecewise polynomials are continuous across knots. Also 3) Linear at boundary for more stable estimates at boundaries: "natural spline"
		* Select K based on desired degrees of freedom
		* Often use cross validation to determine the best K


* Smoothing Splines: Loss + penalty function like ridge regression and lasso. Lambda controls the smoothness. Effective degrees of freedom based on lambda. No "knots", every x is a potential break, but choose lambda to minimize RSS using cross validation, LOOCV.
* Local Regression: weights nearest points to x0 and fits weighted least squares regression.

## Generalized Additive Models (GAMs)

* GAMs for Regression Problems: Replace linear coefficient for each predictor with a separate function. Each function can be estimated with any of the non-linear methods above. Add the functions together.
	+ However the functions have to be additive and interactions among predictors can be missed
	+ Manually add interaction terms like (Xj x Xk)
* GAMs for Classification: Use same method with logistic regression

## Tree-Based Methods
Apply to both regression and classification problems. Internal and terminal nodes, connected by branches.

* Regression: Divide predictor space into regions, predicted value is mean of y in each region. 
	+ Use recursive binary splitting for each predictor cut point s of each predictor Xj to minimize RSS
	+ Prune tree based on cost complexity as function of alpha
	+ Use K-fold cross validation to choose alpha. 

* Classification: Use most commonly occurring class in each region.
	+ Minimize Gini index or cross-entropy to measure "purity" (in lieu of classification error rate) to grow the tree
	+ Use classification error rate to prune the tree (although any of the three would do)
* Assessment:
	+ Easy to explain and intuitive
	+ Can be displayed and explained graphically
	+ Handle qualitative predictors without using dummy variables
	+ But...lower predictive accuracy than some other approaches
	+ High variance versus other methods (but see below)
* Bagging (bootstrap aggregation) reduces the variance
	+ Take B repeated samples from training set,grow a tree for each (no pruning)
		* Regression: average the results
		* Classification: majority vote
	+ Often B in the range of 100
	+ Use Out of Bag error (OOB) to estimate test error
	+ Improved prediction vs. a simple tree, but lose interpretability, but the relative importance of predictors can be determined by which RSS or the Gini index is decreased by splits over a given predictor

* Random Forest limits the number of predictors considered at each split to a random sample of m predictors from the population p.  
	+ If m=p, RF is the same as bagging. Here m= sqrt(p)
	+ Performs better than bagging with a large number of correlated predictors

* Boosting
	+ Grows multiple trees sequentially based on information from previous trees
	+ Fits small tree first, updates f(x) with lambda x previous and same for errors. Effectively fits to the residuals of the prior estimate (for regression).
	+ Parameters are number of trees (B), shrinkage parameter (lambda), typically .01 or .001, Number of splits (d) or the interaction depth. If d=1 it is an additive model.
	+ Usually results in smaller trees
	+ Can be used for many methods, not just decision trees

## Support Vector Machine (SVM)
* Maximal Margin Classifier
	+ Separating Hyperplane: classification based on which side of a defined hyperplane x falls. By definition linear. The margin is the distance a given point x is from the hyperplane.
	+ Maximal Margin Classifier: maximizes the distance of the point(s) with the smallest margin. These points are the "support vector(s)"
	+ Can be very sensitive to small changes in observations
	+ However, it may be that no separating hyperplane exists
	+ Linear
* Support Vector Classifier (soft margin classifier)
	+ Allow some points to be the "wrong" side of the hyperplane. More robust with different observations and better classification of most training observations
	+ C is a non-negative tuning parameter for the maximum margin that will be tolerated. C=0 is same as maximal margin classifier. Generally a tuning parameter determined with cross-validation. Small C low bias/high variance and fewer support vectors
	+ More robust than other methods when there are misclassified outliers far from the hyperplane.
	+ Linear
* Support Vector Machines (single class): Enlarge the feature space to support a non-linear boundary with quadratic, cubic or other predictors. However, in practice only inner products of terms is required
	+ Kernel (K) is a general function that quantifies the similarity of two observations. (A linear kernel is equivalent to the Support Vector Classifier). 
	+ A polynomial kernel  fits support vector classifier in a higher degree polynomial=support vector machine
	+ A radial kernel defines a boundary that forms a loop.

* SVM with multiple classes
	+ One Versus One Classification: Perform pairwise classifications and use the class to which an observation was most frequently assigned
	+ One Versus All Classification: Look at each class k versus the collective of other classes.

* Comparison to other methods: computationally more similar than they seem.
	+ Well separated classes favor SVM
	+ More overlapping regimes favor logistic regression
* Support Vector Regression is an extension


# Unsupervised
Often performed as part of exploratory data analysis.

## Principal Components Analysis (PCA)

* Scale factors X
* Look for the linear combination of X that has the highest variance Z1 (coefficients are "loading factors")
* Then repeat for all linear combinations uncorrelated with Z1
* Often used to generate a biplot of first two principal components
* How many principal components? Use the proportion of variance explained (PVE). Ultimately requires judgement.


## Clustering
Find subgroups in a data set that are "similar" Often the meaning of similar is domain specific. Depending on the problem, we may cluster features based on observations, or observations based on features (used here).

* K-Means Clustering: Pre-specified number of non-overlapping clusters, K. 
	+ Minimize the distance between observations within a cluster.
	+ Squared euclidean distance is often used, but there are others.
	+ Algorithm: 
		* Randomly assign each observation to a cluster 1 to K.
		* Iterate until assignment stops changing:
			+ Compute the cluster centroid (vector of feature means for the cluster)
			+ Assign each observation to the cluster whose centroid is closest.


	+ Stops at a local optimum. Important to run multiple times with different initial configurations and select the best.
	+ Better than hierarchical clustering if there are "natural" clusters

* Hierarchical Clustering: Bottom-up or agglomerative. Tree-like visual representation, dendrogram.
	+ Similarity is based on where an observation is in the (vertical) hierarchy
	+ The final number of clusters is based on a horizontal cutoff
	+ Algorithm:
		* Begin with n observations and a measure of distance (e.g., euclidean distance)
		* Take all pairs and fuse those with the minimum distance into clusters of 2.
		* Repeat, fusing each cluster based on linkage with other clusters, until you are at the top
		* Linkage options: Complete, Single, Average, Centroid. Average or Complete are preferred. Centroid can result in undesirable inversions. Dendrogram can vary a lot between different choices of linkage.

	+ Other choices to make based on the nature of the problem. Alternatives to Euclidean Distance:
		* Distance based on corrleation (of features) between two observations in stead of euclidean distance
		* Scaling of features: has the effect of forcing them to have equal weight


* No consensus on a methodology to "validate" the clusters obtained
* Overall: small decisions about the methodology can have a big impact on the results. 
	+ Try out several to see what patterns are consistent
	+ Experiment with subsets of the data
	+ Be cautious not to oversell the reported results as truth



# General Concepts
Bias (Accuracy) vs. Variance (Sensitivity to alternative data sets)
Bayes Classifier: assign to class based on highest probaility predicted based on X

***
Source: An Introduction to Statistical Learning, Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, (Corrected 4th printing 2014)

