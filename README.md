# Practical Machine Learning - Course Project

The purpose of this repository is the project of the course *Practical Machine Learning* which is part of the Coursera Data Science Specialization. The context of the project is the Human Activity Recognition (HAR).

A training dataset from three sensors on the belt, forearm, arm, and dumbell of 6 participants is provided. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal is to predict the manner in which they did the exercise. Another dataset of 20 new cases is also provided to test the accuracy of our prediction model.

The details and dataset of the original experiment is available here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 
 

## Getting and cleaning data

The dataset provided is a subset of the [original](http://groupware.les.inf.puc-rio.br/static/WLE/WearableComputing_weight_lifting_exercises_biceps_curl_variations.csv), approximately 50%. It has 19622 observations and 160 variables. Only 52 of the 160 variables are from the three sensors (accelerometer, gyroscope and magnetometer) of the four devices (belt, arm, forearm and dumbbell), the rest are derived variables (mean, variance, standard deviation, max, min, amplitude, kurtosis, skewness, etc) computed from a sliding window of 2.5 seconds.

I only use the 52 variables from the sensors because we have to predict the class based on only one observation. The values of the derived variables in the test dataset are nulls. 



```r
train <- read.csv("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
test <- read.csv("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")


belt <- c("roll_belt", "pitch_belt", "yaw_belt", "total_accel_belt", "gyros_belt_x", 
    "gyros_belt_y", "gyros_belt_z", "accel_belt_x", "accel_belt_y", "accel_belt_z", 
    "magnet_belt_x", "magnet_belt_y", "magnet_belt_z")

arm <- c("roll_arm", "pitch_arm", "yaw_arm", "total_accel_arm", "gyros_arm_x", 
    "gyros_arm_y", "gyros_arm_z", "accel_arm_x", "accel_arm_y", "accel_arm_z", 
    "magnet_arm_x", "magnet_arm_y", "magnet_arm_z")

dumbbell <- c("roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", "total_accel_dumbbell", 
    "gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z", "accel_dumbbell_x", 
    "accel_dumbbell_y", "accel_dumbbell_z", "magnet_dumbbell_x", "magnet_dumbbell_y", 
    "magnet_dumbbell_z")

forearm <- c("roll_forearm", "pitch_forearm", "yaw_forearm", "total_accel_forearm", 
    "gyros_forearm_x", "gyros_forearm_y", "gyros_forearm_z", "accel_forearm_x", 
    "accel_forearm_y", "accel_forearm_z", "magnet_forearm_x", "magnet_forearm_y", 
    "magnet_forearm_z")

classe <- c("classe")


train <- train[, c(belt, arm, dumbbell, forearm, classe)]
test <- test[, c(belt, arm, dumbbell, forearm)]
```



## Training

I use the caret package for training the model. The type of model is a random forest and the resampling method is cross-validation with 10 folds.



```r
library(caret, quietly = TRUE, verbose = FALSE)

control <- trainControl(method = "cv", number = 10)

# reproducible analysis
set.seed(825)

fit <- train(classe ~ ., data = train, method = "rf", importance = TRUE, trControl = control)
```



```
## Random Forest 
## 
## 19622 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 17660, 17659, 17659, 17660, 17660, 17660, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.002        0.003   
##   30    1         1      0.002        0.002   
##   50    1         1      0.003        0.004   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```



```r
library(caret, quietly = TRUE, verbose = FALSE)
varImp(fit)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```
## rf variable importance
## 
##   variables are sorted by maximum importance across the classes
##   only 20 most important variables shown (out of 52)
## 
##                         A     B    C    D    E
## pitch_belt           26.3 100.0 64.5 44.4 40.0
## roll_belt            71.6  79.5 81.8 72.2 97.4
## pitch_forearm        66.5  70.5 91.3 55.5 58.1
## magnet_dumbbell_y    68.2  65.6 83.7 60.6 53.0
## magnet_dumbbell_z    77.7  54.8 67.5 48.9 48.8
## yaw_belt             59.5  62.4 67.6 68.8 45.0
## roll_forearm         46.5  36.5 43.3 30.8 33.1
## accel_forearm_x      17.0  35.2 31.9 43.2 34.9
## gyros_belt_z         21.2  28.9 32.2 21.5 41.5
## accel_dumbbell_y     31.4  30.9 39.8 27.6 32.6
## gyros_dumbbell_y     34.3  27.4 39.5 25.6 19.6
## yaw_arm              39.2  35.2 27.1 28.6 18.8
## accel_dumbbell_z     22.1  29.5 19.7 28.2 33.4
## gyros_arm_y          26.1  31.2 23.3 32.1 21.2
## gyros_forearm_y      12.9  29.5 29.2 18.5 14.9
## roll_dumbbell        17.4  29.5 17.7 23.4 28.4
## magnet_belt_y        12.8  28.9 24.0 17.2 23.1
## total_accel_dumbbell 11.0  27.6 15.4 24.0 28.8
## magnet_belt_z        18.2  28.8 22.6 26.4 24.9
## magnet_belt_x        16.0  28.6 23.3 14.0 22.2
```



## Prediction

The prediction of the 20 cases of the test dataset provided are true which confirm the output of the training (accuracy 1).


```r
predict(fit, newdata = test)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```



## Conclusion

Although the results obtained using the random forest are fine, the training process takes a while so I wonder if there is another method faster.


