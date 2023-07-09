##########################################################################
##########################################################################
#
#
#   STATISTICS FROM BASIC TO ADVANCED
#
#
##########################################################################

getwd()
library(usethis)
library(tidyverse)
library(data.table)
library(janitor)
library(arrow)
library(DataExplorer)
library(datasets)

# This tutorial will cover basic to advanced statistics ####
# This tutorial will explain about modelling and prediction techniques in R
# hypothesis testing, linear regression, logistic regression, classification, market basket analysis
# random forest, ensemnle techniques, clustering, and etc. 

# Basic Statistics ####
# There are two variable: Categorical and numerical
# Categorical consist of nominal and ordinal
# Nominal can not be ordered (Female, and Male), while ordinal it can be ordered (ranking)(Agree, neutral, disagree)
# Numerical has two different types: Interval and ratio
# Interval does not has true zero (Temperature), while ratio has true zero (height or weight)


# Descriptive statistics ####
# It deals with description of value in the datasets
# How much dataset spreads from its average
# What is the min and max number in dataset
# Central tendency (Mean, median, mode, SD, Variance, Skewness, etc)

# Measure central tendecy
# Describe the whole dataset with single value represent the centre of its distribution
# There are three main measures: mean, median, mode
# mean (sensitive to outlier), but it's good if data is normally distributed
# media is robust measure if there is extreme values or outlier
# mode is useful when it deals with categoriacl variable

# Measure of dispersion 
# It measure the spread or dispersion of the datasets. 
# There are four measure of variability; range, inter quartile range, SD, and variance
# Range: Difference between min and max ==> very sensitive to outliers
# Standard deviation: average distance of value in distribution from their mean
# variance: square root of standard deviation
# skewness: degree to which scores in distribution are spread out => measure of symmetry
# kurtosis: flatness or peakness of the curve

# Standardize a variable (normalization and scaling)
# It required as the continuous independent variables are measured at different scales.
# These variable do not give equal contribution to the analysis
# The idea is to transform the data to comparable scales. and rescale a original
# variable to have equal range and variance

# Method of standardization or normalization
# There are four main methods for standardization
# 1. Z score
# This is one of the most popular method to normalize the data. 
# We rescale an original variable to have a mean of zero and SD of 1. 
# it works by subtracting mean of original variable from each individual of raw data and then 
# divide it by standard deviation of the original variable

# Creating a sample data
set.seed(123)
X =data.frame(k1 = sample(100:1000,1000, replace=TRUE),
              k2 = sample(10:100,1000, replace=TRUE))
X.scaled = scale(X, center= TRUE, scale=TRUE) # scale is used for standardization or normalization
# center = true, mean subtracting the mean from its original variable
# the scale = true, mean dividing the centered column by its standard deviations
head(X, 5)
head(X.scaled, 5)
colMeans(X.scaled)
var(X.scaled)

# Min max scaling
# it is also called 0-1 scaling
# formula: x-min(x)/(max(x)-min(x))

# This method is used to make equal range but different means and standard deviations
library(dplyr)
mins= as.integer(summarise_all(X, min))
mins
rng = as.integer(summarise_all(X, function(x) diff(range(x))))
rng
X.scaled = data.frame(scale(X, center= mins, scale=rng))
head(X.scaled, 5)

summarise_all(X.scaled, funs(min, max))

# Standard deviation method
# We divided each value by standard deviation. 
# The idea is to have equal variance, but different means and range
# formula; x/stdev(x)

X.scaled = data.frame(scale(X, center= FALSE , scale=apply(X, 2, sd, na.rm = TRUE)))
head(X.scaled, 5)
summarise_all(X.scaled, var)

# Range method
# We divide each value by the range.
# Formula; x/(max(x)-min(x))
# mean, variance and range of the variables are still different but at least
# the ranges are likely to be more similar

library(dplyr)
rng = as.integer(summarise_all(X, function(x) diff(range(x))))
X.scaled = data.frame(scale(X, center= FALSE, scale=rng))
summarise_all(X.scaled, var)

# Centering
# Centering means subtracting constant value from every value of a variable
# Constant value an be average, min or max
X=sample(1:100,1000, replace=TRUE)
scale(X,center = TRUE, scale=FALSE)
# by default, scale() function with center = TRUE, it substract mean value from values of a variable


# When it is important to standardize variables?
# It is important to standardize variables before running cluster analysis. 
# It is because cluster analysis techniques depend on the concept of measuring the distance between the different observations we're trying to cluster.
# 
# Prior to principle component analysis, it is critical to standardize variables. 
# it is because PCA gives more weightage to those variables that have higher variances
# than to those variables that have very low variances. 

# It is required to standardize variable before using k-nearest neighbors with an euclidean distance measure
# standardization makes all variables to contribute equally

# All SVM (Support Vector Machine) method are based on distance so it is required to scale
# variables prior to running final support vector machine (SVM) model

# it is necessary to standardize variables before using lasso and ridge regression
# LASSO regression puts constraints on the size of the coefficient associated to 
# each variable
# The result of centering the variables means that there is no longer an intercept. This applies equally to ridge regression.

# In regression, we can calculate the importance of variables by ranking independent
# variables based no the descending order of absolute value of standardized coefficient
# In regression, when an interaction is created from two variables that are not centered on 0, some amount of collinearity will be induced. 
# Centering first addresses this potential problem

# When it is not required to standardize variables
# Standardize does not change RMSE, R-squared value, adjusted-R-squared value, 
# p-value of coefficients. 
# Example and prove as below:
# Create Sample Data
set.seed(123)
train <- data.frame(X1=sample(1:100,1000, replace=TRUE),
                    X2=1e2*sample(1:500,1000, replace=TRUE),
                    X3=1e-2*sample(1:100,1000, replace=TRUE))
train$y <- with(train,2*X1 + 3*1e-2*X2 - 5*1e2*X3 + 1 + rnorm(1000,sd=10))
head(train, 4)

#Fit linear regression model
fit  <- lm(y~X1+X2+X3,train)
summary(fit)

# Predict on test datasets
# in the program below, we are first preparing a sample test dataset which is used
# later for prediction
# create test dataset
set.seed(456)
test <- data.frame(X1=sample(-5:5,100,replace=TRUE),
                   X2=1e2*sample(-5:5,100, replace=TRUE),
                   X3=1e-2*sample(-5:5,100, replace=TRUE))
# predict y based on test data without standardization
pred   <- predict(fit,newdata=test)
head(cbind(test, pred))

# With standardization 
# In the script below, we are first storing mean and standard deviation of variables
# of training dataset in two separate numeric vectors. Later, these vectors are used to
# standardize training dataset
# Standardize predictors
(means   <- sapply(train[,1:3],mean))
(stdev <- sapply(train[,1:3],sd))
train.scaled <- as.data.frame(scale(train[,1:3],center=means,scale=stdev))
head(train.scaled)
train.scaled$y <- train$y

# Check mean and Variance of Standardized Variables
library(dplyr)
summarise_at(train.scaled, vars(X1,X2,X3), funs(round(mean(.),4)))
summarise_at(train.scaled, vars(X1,X2,X3), var)
# Result : Mean is 0 and Variance is 1 for all the standardized variables.

#Fit Scaled Data
fit.scaled <- lm(y ~ X1 + X2 + X3, train.scaled)
summary(fit.scaled)
summary(fit)

# Compare Coefficients, R-Squared and Adjusted R-Squared 
# value of coefficients are not same when we run regression analysis with and without standardizing independent variables
# It does not mean they are affected by scaling / standardization
# The values are different because of these are the slopes - how much the target variable changes if you change independent variable by 1 unit. In other words, standardization can be interpreted as scaling the corresponding slopes.
# The adjusted r-squared and multiple r-squared value is exactly same

# How to standardize validation / test dataset
# To standardize validation and test dataset, we can use mean and standard deviation of independent variables from training data. 
# Later we apply them to test dataset using Z-score formula
z = (X_test - Xbar_training) / Stdev_training

# Script for standardized test data
#  we are using mean and standard deviation of training data which is used to calculate Z score on test data.
test.scaled <- as.data.frame(scale(test,center=means,scale=stdev))
head(test.scaled)

# Compare Prediction - Scaled vs Unscaled
# predict y based on new data scaled, with fit from scaled dataset
pred.scaled   <- predict(fit.scaled,newdata=test.scaled)

# Compare Prediction - unscaled vs. scaled fit
all.equal(pred,pred.scaled) # the result is true, means that there is no difference

head(cbind(pred,pred.scaled),n=10)
# As you can see above both prediction values are exact same.

# Compare RMSE score
# RMSE on train data with un-scaled fit
pred_train   <- predict(fit,newdata=train)
rmse <- sqrt(mean((train$y - pred_train)^2))
rmse

# RMSE on train data with scaled fit
pred_train.scaled   <- predict(fit.scaled,newdata=train.scaled)
rmse.scaled <- sqrt(mean((train$y - pred_train.scaled)^2))
rmse.scaled

# Compare RMSE
all.equal(rmse,rmse.scaled) 
# RMSE is the same 
# It is because RMSE is associated with scale of Y (target variable). 
# Prediction is also unchanged.

# Interpretation of Standardized Regression Coefficient
# Most of modern statistical softwares automatically produces standardized regression coefficient. 
# It is important metrics to rank predictors. Its interpretation is slightly different from unstandardized estimates. 
# Standardized coefficients are interpreted as the number of standard deviation units Y changes with an increase in one standard deviation in X.

# Correlation with or without Centering / Standardization
# The correlation score does not change if you perform correlation analysis on centered and uncentered data. 
X=sample(1:100,1000, replace=TRUE)
Y=1e2*sample(1:500,1000, replace=TRUE)
cor(X,Y)
cor(X-mean(X),Y-mean(X))
X

# Standardization after missing imputation and outlier treatment
# Centering and Scaling data should be done after imputing missing values. 
# It is because the imputation could influence correct center and scale to use. 
# Similarly, outlier treatment should be done prior to standardization.

# Standardize Binary (Dummy) Variables 
# Standardizing binary variables makes interpretation of binary variables vague as it cannot be increased by  a standard deviation. 
# The simplest solution is : not to standardize binary variables but code them as 0/1, 
# and then standardize all other continuous variables by dividing by two standard deviation. 
# It would make them approximately equal scale. The standard deviation of both the variables would be approx. 0.5

# Standardization and Tree Algorithms and Logistic Regression
# Standardization does not affect logistic regression, 
# decision tree and other ensemble techniques such as random forest and gradient boosting.





