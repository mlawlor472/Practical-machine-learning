Practical Machine Learning - Prediction Assignment Writeup


Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

Data
The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

Goal
The goal of this project is to predict the manner in which they did the exercise. This is the “classe” variable in the training set. You will see a report describing how we built our model, how we used cross validation, what we think the expected out of sample error is, and why we made the choices we did. We use our prediction model to predict 20 different test cases in this project.

Preliminary Work
First, we set the seed at 1234 for reproduceability. We then downloaded, installed and loaded the required packages that are going to be used in this project which are caret, randomForest, rpart and rpart.plot.

This report outcome variable is classe and factor variable with 5 levels. For this dataset, “participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in 5 different fashions:

Class A = really according to the specification Class B = elbowed to the front Class C = lifting the dumbbell only halfway Class D = dropping the dumbbell only halfway Class E = throwing the hips to the front

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes." Prediction evaluations will be based on maximizing the accuracy and minimizing the out-of-sample error. All other available variables after cleaning will be used for prediction. Two models will be tested using Decision Tree and Random Forest algorithms. The model with the highest accuracy will be selected.

Expected sample error
The expected out-of-sample error will correspond to the quantity: 1-accuracy in the cross-validation data. Accuracy is the proportion of correct classified observation over the total sample in the subTesting data set. Expected accuracy is the expected accuracy in the out-of-sample data set (i.e. original testing data set). Thus, the expected value of the out-of-sample error will correspond to the expected number of missclassified observations/total observations in the Test data set, which is the quantity: 1-accuracy found from the cross-validation data set.

Cross-validation
We will perform cross-validation by subsampling our training data set randomly without replacement into 2 subsamples: subTraining data (75% of the original Training data set) and subTesting data (25%). We will fit our models on the subTraining data set, and then test them on the subTesting data. Once the most accurate model is choosen, we will test it on the original Testing data set.

Reasons for my choices
Our outcome variable “classe” is an unordered factor variable. Thus, we can choose our error type as 1-accuracy. Large sample size with N = 19622 in the Training data set. So it were divided into subTraining and combine with subTesting to allow cross-validation. Columns or fields with all missing values will be discarded as well as features that are irrelevant. All other features will be kept as relevant variables. Random Forest and Decision Tree algorithms are known for their ability of detecting the features that are important for classification.

Preprocessing
We set the working directory and seed, and load the installed libraries.

setwd("~/Coursera/Assignments/RProgramming/Module8-Assignment")
set.seed(1234)
library(caret)
## Warning: package 'caret' was built under R version 3.2.3
## Loading required package: lattice
## Loading required package: ggplot2
## Warning: package 'ggplot2' was built under R version 3.2.2
library(randomForest) #Random forest for classification and regression
## Warning: package 'randomForest' was built under R version 3.2.3
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
library(rpart) #Regressive Partitioning and Regression trees
Now we load the data sets into R, interpreting the miscellaneous NA, #DIV/0! and empty fields as “NA”.

trainingData <- read.csv("pml-training.csv", header=T, na.strings=c("NA","#DIV/0!", ""))
testingData <- read.csv('pml-testing.csv', header=T, na.strings=c("NA","#DIV/0!", ""))
Cleaning the data
We partition and preprocess the training data using the code described below. We exclude all variables with at least one ‘NA’ from the analysis, as well as variables related to time and user information, where a total of 51 variables and 19622 class measurements were excluded. We maintain the same variables in the test data set (Validation dataset) to be used for predicting the 20 test cases provided.

## NA exclusion for all available variables
noNATrainingData<-trainingData[, apply(trainingData, 2, function(x) !any(is.na(x)))] 
dim(noNATrainingData)
## [1] 19622    60
## variables with user information, time and undefined
cleanTrainingData<-noNATrainingData[,-c(1:8)]
dim(cleanTrainingData)
## [1] 19622    52
## 20 test cases provided clean info - Validation data set
cleanTestingData<-testingData[,names(cleanTrainingData[,-52])]
dim(cleanTestingData)
## [1] 20 51
Partitioning the data and prediction process
Now we subset the cleaned downloaded data set to generate a test set independent from the 20 cases provided set. We perform data partitioning to obtain a 75% training set and a 25% test set.

#data cleaning
inTrain<-createDataPartition(y=cleanTrainingData$classe, p=0.75,list=F)
training<-cleanTrainingData[inTrain,] 
test<-cleanTrainingData[-inTrain,] 
#Training and test set dimensions
dim(training)
## [1] 14718    52
dim(test)
## [1] 4904   52
Results and Conclusions
We generate random forest trees for the training dataset using cross-validation. Then we examine the generated algorithm under the partitioned training set to examine the accuracy and estimated error of prediction. By using 51 predictors for five classes using cross-validation at a 5-fold an accuracy of 99.2% with a 95% CI [0.989-0.994] was achieved accompanied by a Kappa value of 0.99.

fitControl2<-trainControl(method="cv", number=5, allowParallel=T, verbose=T)
rffit<-train(classe~.,data=training, method="rf", trControl=fitControl2, verbose=F)
## + Fold1: mtry= 2 
## - Fold1: mtry= 2 
## + Fold1: mtry=26 
## - Fold1: mtry=26 
## + Fold1: mtry=51 
## - Fold1: mtry=51 
## + Fold2: mtry= 2 
## - Fold2: mtry= 2 
## + Fold2: mtry=26 
## - Fold2: mtry=26 
## + Fold2: mtry=51 
## - Fold2: mtry=51 
## + Fold3: mtry= 2 
## - Fold3: mtry= 2 
## + Fold3: mtry=26 
## - Fold3: mtry=26 
## + Fold3: mtry=51 
## - Fold3: mtry=51 
## + Fold4: mtry= 2 
## - Fold4: mtry= 2 
## + Fold4: mtry=26 
## - Fold4: mtry=26 
## + Fold4: mtry=51 
## - Fold4: mtry=51 
## + Fold5: mtry= 2 
## - Fold5: mtry= 2 
## + Fold5: mtry=26 
## - Fold5: mtry=26 
## + Fold5: mtry=51 
## - Fold5: mtry=51 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 26 on full training set
predrf<-predict(rffit, newdata=test)
confusionMatrix(predrf, test$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393    7    0    0    0
##          B    2  940    9    0    0
##          C    0    2  845    6    1
##          D    0    0    1  798    1
##          E    0    0    0    0  899
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9941         
##                  95% CI : (0.9915, 0.996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9925         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9986   0.9905   0.9883   0.9925   0.9978
## Specificity            0.9980   0.9972   0.9978   0.9995   1.0000
## Pos Pred Value         0.9950   0.9884   0.9895   0.9975   1.0000
## Neg Pred Value         0.9994   0.9977   0.9975   0.9985   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2841   0.1917   0.1723   0.1627   0.1833
## Detection Prevalence   0.2855   0.1939   0.1741   0.1631   0.1833
## Balanced Accuracy      0.9983   0.9939   0.9930   0.9960   0.9989
pred20<-predict(rffit, newdata=cleanTestingData)
pred20
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
So now we run a boosting algorithm to confirm and be able to compare predictions. Data is not shown but the boosting approach presented less accuracy (96%) (Data not shown). However, when the predictions for the 20 test cases were compared match was same for both ran algorimths.

fitControl2<-trainControl(method="cv", number=5, allowParallel=T, verbose=T)
gmbfit<-train(classe~.,data=training, method="gbm", trControl=fitControl2, verbose=F)
## Loading required package: gbm
## Warning: package 'gbm' was built under R version 3.2.3
## Loading required package: survival
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: splines
## Loading required package: parallel
## Loaded gbm 2.1.1
## Loading required package: plyr
## Warning: package 'plyr' was built under R version 3.2.1
gmbfit$finalModel
class(gmbfit)
predgmb<-predict(gmbfit, newdata=test)
confusionMatrix(predgmb, test$classe)
predtrain<-predict(gmbfit, newdata=training)
confusionMatrix(predtrain, training$classe)
predtrain<-predict(gmbfit, newdata=training)
confusionMatrix(predtrain, training$classe)
Once we obtain the predictions for the 20 test cases provided, we use the script below to obtain single text files to be uploaded to the courses web site to comply with the submission assigment. 20 out of 20 hits also confirmed the accuracy of the obtained models.

getwd()
## [1] "C:/Users/Farrah/Documents/Coursera/Assignments/RProgramming/Module8-Assignment"
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(pred20)

