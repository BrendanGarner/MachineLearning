library(readxl)
library(dplyr)
library(caret)
library(ranger)
library(mlr)
library(ggplot2)

library(tensorflow)
tf$random$set_random_seed(1234)

library(keras)
use_session_with_seed(1234)

###########################################
# Only required once
#install python packages  
library(reticulate)
#create a new environment
conda_create("r-reticulate")
#install tensorflow
conda_install("r-reticulate", "tensorflow")
###########################################

# Question 2a) - prepare data
# Import the spreadsheet into an R data frame
# Skip the first row and use the second row as headers
col_names <- array(read_excel('AUS_Data.xlsx', skip=1, n_max = 1, col_names = FALSE))
data <- data.frame(read_excel('AUS_Data.xlsx', skip = 2, col_names = FALSE))
colnames(data) <- col_names

# Abbreviate the column names
colnames(data) <- c("period", "unemployment_rate", "GDP", 
                    "government_consumption", "all_sectors_consumption",
                    "terms_of_trade", "CPI", "job_vacancies", "population")

# Move the period column to the right of the data frame
# Change column to date format
data <- data %>% select(-period, period)
data$period <- as.Date(data$period, origin="1899-12-30")

# Question 2b) - descriptive statistics
# Use box plot to visualise the data distribution for each variable
data_stacked <- stack(data[1:4])
ggplot(data_stacked, aes(x="", y=values)) +
  ggtitle ("Distribution of data") +
  theme(plot.title = element_text(hjust = 0.5), 
        axis.title.x=element_blank()) + 
  geom_boxplot(outlier.color="red", outlier.shape=3) + 
  geom_jitter(width=0.1, alpha=0.05, color="blue") +
  facet_wrap( ~ ind, scales="free")

# Question 3. Apply one supervised ML algorithm - random trees
# Split data: train (June 81 - Dec 2017), test (Mar 18 - Sep 20)
train <- subset(data, period < as.Date("2018-03-01"))
test <- subset(data, period >= as.Date("2018-03-01"))

# Remove period data
train <- subset(train, select = -(period))
test <- subset(test, select = -(period))

# Define task and learner
train.task <- makeRegrTask(data = train, target = "unemployment_rate")
learner <- makeLearner("regr.ranger")

set.seed(1234)
# Choose resampling strategy and define grid
rdesc <- makeResampleDesc("RepCV", folds = 10, reps = 10)
ps <- makeParamSet(makeDiscreteParam("mtry", 5),
                   makeDiscreteParam("importance", "permutation"),
                   makeDiscreteParam("min.node.size", 2),
                   makeDiscreteParam("num.trees", 950))

# Tune parameters
res = tuneParams(learner, train.task, rdesc, par.set = ps,
                 control = makeTuneControlGrid())

# Train on dataset using best hyperparameters
lrn = setHyperPars(makeLearner("regr.ranger"), par.vals = res$x)
m <- mlr::train(lrn, train.task)

# Make predictions for unemployment rate
cat("OOB prediction error (MSE) on training data is ", m$learner.model$prediction.error)
pred_test_RF <- predict(m$learner.model, data = test)
print(pred_test_RF$predictions)
print(test$unemployment_rate)

# Implement artificial neural network
# Apply normalisation to data
mean <- apply(train, 2, mean)
std <- apply(train, 2, sd)
train <- scale(train, center = mean, scale = std)
test <- scale(test, center = mean, scale = std)

# Separate unemployment rate as target
train_targets <- subset(train, select = unemployment_rate)
train <- subset(train, select = -(unemployment_rate))
test_targets <- subset(test, select = unemployment_rate)
test <- subset(test, select = -(unemployment_rate))

# Build neural network
build_model <- function() {
  k_clear_session()
  set.seed(1234)
  tf$random$set_random_seed(1234)
  model <- keras_model_sequential() %>%
    layer_dense(units = 7, activation = "relu",
                input_shape = dim(train)[[2]]) %>%
    layer_dense(units = 7, activation = "relu") %>%
    layer_dense(units = 7, activation = "relu") %>%
    layer_dense(units = 7, activation = "relu") %>%
    layer_dense(units = 7, activation = "relu") %>%
    layer_dense(units = 7, activation = "relu") %>%
    layer_dense(units = 7, activation = "relu") %>%
    layer_dense(units = 1)
  
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae")
  )
}

k <- 4
indices <- sample(1:nrow(train))
folds <- cut(indices, breaks = k, labels = FALSE)
num_epochs <- 1000
all_mae_histories <- NULL
for (i in 1:10) {
  for (i in 1:k) {
    cat("processing fold #", i, "\n")
    
    # Prepare the validation data: data from partition # k
    val_indices <- which(folds == i, arr.ind = TRUE)
    val_data <- train[val_indices,]
    val_targets <- train_targets[val_indices]
    
    # Prepare the training data: data from all other partitions
    partial_train_data <- train[-val_indices,]
    partial_train_targets <- train_targets[-val_indices]
    
    # Build the Keras model (already compiled)
    model <- build_model()
    
    # Train the model (in silent mode, verbose=0)
    history <- model %>% fit(
      partial_train_data, partial_train_targets,
      validation_data = list(val_data, val_targets),
      epochs = num_epochs, batch_size = 1, verbose = 0
    )
    mae_history <- history$metrics$val_mean_absolute_error
    all_mae_histories <- rbind(all_mae_histories, mae_history)
  }
}

# compute the average of the per-epoch MAE scores for all folds
average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories, 2, mean)
)

# Plot validation scores
ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_smooth() +
  ggtitle ("14 neurons") +
  theme(plot.title = element_text(hjust = 0.5))

mea_test <- 0
mea_train <- 0
average_predictions <- 0
k <- 50
# Fit final model and make predictions on test data
for (i in 1:k) {
  k_clear_session()
  set.seed(1234)
  tf$random$set_random_seed(1234)
  model <- build_model()
  set.seed(1234)
  tf$random$set_random_seed(1234)
  model %>% fit(train, train_targets,
                epochs = 300, batch_size = 16, verbose = 0)
  result_test <- model %>% evaluate(test, test_targets)
  mea_test <- mea_test + result_test$mean_absolute_error
  result_train <- model %>% evaluate(train, train_targets)
  mea_train <- mea_train + result_train$mean_absolute_error
  
  predictions <- model %>% predict(test)
  predictions_unscaled <- t(predictions) * std[1] + mean[1]
  average_predictions <- average_predictions + predictions_unscaled
}

cat("average mea_train is", mea_train/k)
cat("average mea_test is", mea_test/k)
cat("average predicitons are", average_predictions/k)

