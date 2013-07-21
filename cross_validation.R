# Minimal but more intuitive to use cross validation tool.
require(cvTools)
require(foreach)

# TODO: Stratified sampling is necessary so that minority class is not
# undersampled.

# Function for cross validation.
cv <- function(K,  # How many folds do you want?
               R,  # Number of replications
               type,  # type of folds to be generated. 'random', 'consecutive', 'interleaved'.
               samp_size=1.0,  # What fraction of data to be used as validation.
               seed,  # Seed for random number generator.
               preprocessor,
               modeller,
               predictor,
               evaluator,
               data,
               setting, # Algorithm parameter, e.g., number of trees.
               ...) {
  set.seed(seed)
  data <- data[sample(1:NROW(data)), ]
  data <- data[1:(NROW(data) * samp_size),]
  if (type != 'random') {
    warning(paste("Replication is not supported if type is not random. ",
            "No replication was made."))
    R=1
  }
  folds <- cvFolds(NROW(data), K, R, type)
  foreach(r=1:R, .combine='c') %do% {
    kfold_result <- foreach(i=1:K) %dopar% {
      cat('\nRepeat #', r, ' Fold #', i, '\n')
      sub_train <- data[folds$subsets[folds$which != i, r], ]
      sub_valid <- data[folds$subsets[folds$which == i, r], ]
      cat('training:', NROW(sub_train), 'rows.\n')
      cat('validation:', NROW(sub_valid), 'rows.\n')
      cat('-- Preprocessing\n')
      time.preprocess <- system.time(
        lst <- preprocessor(sub_train, sub_valid, setting, ...))
      print(time.preprocess)
      sub_train <- lst$train
      sub_valid <- lst$valid
      sub_preprocessed <- lst$preprocessed
      # TODO: return prediction accuracy for training as well.
      cat('-- Modelling\n')
      time.modelling <- system.time(
        m <- modeller(sub_train, setting, ...))
      print(time.modelling)
      cat('-- Prediction\n')
      time.prediction <- system.time(
        p <- predictor(m, sub_preprocessed, sub_valid, setting, ...))
      print(time.prediction)
      cat('-- Evaluation\n')
      eval_result <- evaluator(p, sub_valid, setting, ...)
      print(eval_result)
      return(list(
        time.preprocess=time.preprocess, 
        time.modelling=time.modelling, 
        time.prediction=time.prediction,
        evaluation=eval_result))
    }
    return(kfold_result)
  }
}

cv.demo.classification <- function() {
  # Example for predicting Species using Petal.Length.
  require(caret)
  require(foreach)
  require(randomForest)
  data(iris)
  
  preprocessor <- function(train, valid, setting) {
    preProcValues <- preProcess(train[, 1:4], method = c('center', 'scale'))
    train[, 1:4] <- predict(preProcValues, train[, 1:4])
    valid[, 1:4] <- predict(preProcValues, valid[, 1:4])
    return(list(train=train, valid=valid))
  }
  
  modeller <- function(train, setting) {
    return(randomForest(Species ~ Petal.Length, data=train,
                        ntree=setting$ntree))
  }
  
  predictor <- function(model, preprocessed, valid, setting) {
    return(predict(model, newdata=valid, type='class'))
  }
  
  evaluator <- function(predicted, valid, setting) {
    return(confusionMatrix(predicted, valid$Species))
  }
  
  result <- cv(10, 3, 'random', 1.0, 1234, 
               preprocessor, modeller, predictor, evaluator, 
               iris, setting=list(ntree=100))  
  eval_total <- foreach(r=result, .combine='+') %do% {
    r$evaluation$table
  }
  cat('\n\nResult: \n')
  print(prop.table(eval_total, margin=1))
}


cv.demo.regression <- function() {
  # Example for predicting Sepal.Length from Sepal.Width
  data(iris)
  
  preprocessor <- function(train, valid, setting) {
    sepal.length.mean <- mean(train$Sepal.Length)
    sepal.width.mean <- mean(train$Sepal.Width)
    
    train$Sepal.Length <- train$Sepal.Length - sepal.length.mean
    train$Sepal.Width <- train$Sepal.Width - sepal.width.mean
    valid$Sepal.Width <- valid$Sepal.Width - sepal.width.mean
    
    return(list(train=train, 
                valid=valid, 
                preprocessed=sepal.length.mean))
  }
  
  modeller <- function(train, setting) {
    return(lm(Sepal.Length ~ Sepal.Width, data=train))
  }
  
  predictor <- function(model, preprocessed, valid, setting) {
    return(predict(model, newdata=valid) + preprocessed)
  }
  
  evaluator <- function(predicted, valid, setting) {
    return(sum((valid$Sepal.Length - predicted)^2) / NROW(valid))
  }
  
  result <- cv(10, 3, 'random', 1.0, 1234, 
               preprocessor, modeller, predictor, evaluator, 
               iris, NULL)
  cat('Result: \n')
  print(result)
}
