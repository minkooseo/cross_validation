# Minimal but more intuitive to use cross validation tool.
require(cvTools)
require(plyr)

# Function for cross validation.
cv <- function(K,  # How many folds do you want?
               type,  # type of folds to be generated. 'random', 'consecutive', 'interleaved'.
               samp_size=1.0,  # What fraction of data to be used as validation.
               seed,  # Seed for random number generator.
               preprocessor,
               modeller,
               predictor,
               evaluator,
               data, 
               ...) {
  set.seed(seed)
  data <- sample(data)
  data <- data[1:(NROW(data) * samp_size),]
  folds <- cvFolds(NROW(data), K, R=1, type)
  llply(1:K, function(i) {
    cat('Fold #', i, '\n')
    sub_train <- data[folds$subsets[folds$which != i], ]
    sub_valid <- data[folds$subsets[folds$which == i], ]
    time.preprocess <- system.time(
      lst <- preprocessor(sub_train, sub_valid))
    sub_train <- lst$train
    sub_valid <- lst$valid
    sub_preprocessed <- lst$preprocessed
    # TODO: return prediction accuracy for training as well.
    time.modelling <- system.time(
      m <- modeller(sub_train))
    time.prediction <- system.time(
      p <- predictor(m, sub_preprocessed, sub_valid))
    return(list(
      time.preprocess=time.preprocess, 
      time.modelling=time.modelling, 
      time.prediction=time.prediction,
      evaluation=evaluator(p, sub_valid)))
  }, ...)
}

cv.demo.classification <- function() {
  # Example for predicting Species using Petal.Length.
  require(caret)
  require(rpart)
  data(iris)
  
  preprocessor <- function(train, valid) {
    preProcValues <- preProcess(train[, 1:4], method = c('center', 'scale'))
    train[, 1:4] <- predict(preProcValues, train[, 1:4])
    valid[, 1:4] <- predict(preProcValues, valid[, 1:4])
    return(list(train=train, valid=valid))
  }
  
  modeller <- function(train) {
    return(rpart(Species ~ Petal.Length, data=train))
  }
  
  predictor <- function(model, preprocessed, valid) {
    return(predict(model, newdata=valid, type='class'))
  }
  
  evaluator <- function(predicted, valid) {
    return(sum(predicted == valid$Species) / NROW(valid))
  }
  
  result <- cv(10, 'consecutive', 1.0, 1234, 
               preprocessor, modeller, predictor, evaluator, 
               iris)
  cat('Result: ')
  print(result)
}


cv.demo.regression <- function() {
  # Example for predicting Sepal.Length from Sepal.Width
  data(iris)
  
  preprocessor <- function(train, valid) {
    sepal.length.mean <- mean(train$Sepal.Length)
    sepal.width.mean <- mean(train$Sepal.Width)
    
    train$Sepal.Length <- train$Sepal.Length - sepal.length.mean
    train$Sepal.Width <- train$Sepal.Width - sepal.width.mean
    valid$Sepal.Width <- valid$Sepal.Width - sepal.width.mean
    
    return(list(train=train, 
                valid=valid, 
                preprocessed=sepal.length.mean))
  }
  
  modeller <- function(train) {
    return(lm(Sepal.Length ~ Sepal.Width, data=train))
  }
  
  predictor <- function(model, preprocessed, valid) {
    return(predict(model, newdata=valid) + preprocessed)
  }
  
  evaluator <- function(predicted, valid) {
    return(sum((valid$Sepal.Length - predicted)^2) / NROW(valid))
  }
  
  result <- cv(10, 'consecutive', 1.0, 1234, 
               preprocessor, modeller, predictor, evaluator, 
               iris)
  cat('Result: ')
  print(result)
}
