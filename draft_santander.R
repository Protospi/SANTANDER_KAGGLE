##############################################################################################################

# Draft Santader Data Mining Project

##############################################################################################################

# Load libraries
library(tidyverse)
library(caret)
library(pROC)
library(naniar)
library(reshape2)
library(matrixStats)

##############################################################################################################

# Load data
train <- read_csv("data/train.csv") %>% select(target:var_199) %>% rename(class = target)
test <- read_csv("data/test.csv")
submission <- read_csv("data/sample_submission.csv")


##############################################################################################################

# Inspect data meta information and missing values

##############################################################################################################

# Print dimensins of the data frames
dim(train)
dim(test)
dim(submission)

# Print missing values
miss_var_summary(train)
miss_var_summary(test)

##############################################################################################################

# Preporcess data

##############################################################################################################

# Define X atributes variable
X <- train %>% select(var_0:var_199) 

# Define Y target variable
Y <- train$class

# Scale X
X_scale <- X %>% scale()

# Apply pca
pca <- prcomp(X)

# Summary of pca
summary(pca)

# Extract X dimmesion reduced
X_pca <- pca$x %>% as_tibble()

##############################################################################################################

# Explore data

##############################################################################################################

# Compute number of transactions on train set
pct_target_df <- (table(Y) / length(Y)) %>%
  as_tibble() %>% 
  mutate(Y = c("Não", "Sim")) %>% 
  rename(`Transação` = Y, Percentual = n)

# Draw colum plot of transactions
ggplot(pct_target_df, aes(x = `Transação`, y = Percentual, fill = `Transação`)) +
  geom_col()+
  geom_text(aes(label = round(Percentual,4)), vjust = -0.5)+
  labs(title = "Percentual de Transações do Banco de Treino Completo")

##############################################################################################################

# Gradient boosted machine Classifier

##############################################################################################################

# Define X atributes variable
X_scale <- train %>% select(var_0:var_199) %>%  scale() %>% as_tibble()

# Define Y target variable
Y_df <- tibble(class = Y)

# Define df_model
df_model <- Y_df %>%  bind_cols(X_scale)

# Define models train and test
split <- round(nrow(df_model) * 0.8)
train_train <- df_model[1:split,]
train_test <- df_model[split:nrow(df_model),]

# Check percentages of split
prop.table(table(df_model$class))
prop.table(table(train_train$class))
prop.table(table(train_test$class))

# Define tibble of percentage target split train   
pct_train_df <- tibble(`Transação` = c("Não", "Sim"),
                       Percentual = as.vector(prop.table(table(train_train$class))))

# Draw colum plot of transactions
ggplot(pct_train_df, aes(x = `Transação`, y = Percentual, fill = `Transação`)) +
  geom_col()+
  geom_text(aes(label = round(Percentual,4)), vjust = -0.5)+
  labs(title = "Percentual de Transações do Banco de Treino Treino")

# Define tibble of percentage target split train   
pct_test_df <- tibble(`Transação` = c("Não", "Sim"),
                      Percentual = as.vector(prop.table(table(train_test$class))))

# Draw colum plot of transactions
ggplot(pct_test_df, aes(x = `Transação`, y = Percentual, fill = `Transação`)) +
  geom_col()+
  geom_text(aes(label = round(Percentual,4)), vjust = -0.5)+
  labs(title = "Percentual de Transações do Banco de Treino Teste")

##############################################################################################################

# Correlation Matrix

##############################################################################################################

# Define correlation matrix
cormat <- round(cor(X),2)
melted_cormat <- melt(cormat) %>%  
  mutate(Var1 = as.numeric(str_remove(Var1, "var_")),
         Var2 = as.numeric(str_remove(Var2, "var_")))

# Define breaks
dist_breaks <- seq(0,200, by = 10)

# PLot correlation matrix
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()+
  scale_x_continuous( breaks= dist_breaks)+
  scale_y_continuous( breaks = dist_breaks)+
labs(title = "Mapa de calor de Correlações das Variáveis",
     x = "Variáveis",
     y = "Variáveis")

##############################################################################################################

# Gradiente machine boost

##############################################################################################################

# Setup train controls with 10 folds and 5 repetitions
ctrl <- trainControl(method = "repeatedcv",
                     number = 2,
                     repeats = 1,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     verboseIter = TRUE,
                     seeds = set.seed(3008))

# Define weights
model_weights <- ifelse(train_train$class == 0,
                        (1/table(train_train$class)[1]) * 0.5,
                        (1/table(train_train$class)[2]) * 0.5)

# Define gbm dataframe
gbm_df <- train_train %>% 
  mutate(class = as.factor(ifelse(class == 0, "Não", "Sim")))

##############################################################################################################

# Define classifier
weighted_fit <- train(class ~ .,
                      data = gbm_df,
                      method = "gbm",
                      verbose = TRUE,
                      weights = model_weights,
                      metric = "ROC",
                      trControl = ctrl)

##############################################################################################################

# Predict to weghted fit
predicted_weight <- predict(weighted_fit, train_test, type = "prob")

# Define roc_curve
roc_curve <- roc(train_test$class, predicted_weight[, "Sim"])

# Print AUC
auc(roc_curve)

# Regenerate prediction to submit
test_sub <- test %>% select(var_0:var_199) %>%  scale() %>% as_tibble()
pred_sub <- predict(weighted_fit, test_sub, type = "prob")

# Calculate confusion matrix
conf_mat <- confusionMatrix(as.factor(ifelse(predicted_weight$Sim > 0.5, 1, 0)),
                           as.factor(train_test$class))

# Functio to draw matrix
draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('Matriz de Confus�o das Predi��es de Transa��es', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'N�o', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Sim', cex=1.2)
  text(125, 370, 'Predito', cex=1.3, srt=90, font=2)
  text(245, 450, 'Observado', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'N�o', cex=1.2, srt=90)
  text(140, 335, 'Sim', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "M�tricas", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}  

# Draw the matrix
draw_confusion_matrix(conf_mat)

# Update submissions  
submission$target <- ifelse(pred_sub$Sim > 0.5, 1, 0)

# Write csv to submission
write_csv(submission, "data/sample_submission.csv")

##############################################################################################################

# Build orig model
orig_fit <- train(class ~ .,
                   data = gbm_df,
                   method = "gbm",
                   verbose = FALSE,
                   metric = "ROC",
                   trControl = ctrl)

##############################################################################################################

# Build down-sampled model
ctrl$sampling <- "down"
down_fit <- train(class ~ .,
                  data = gbm_df,
                  method = "gbm",
                  verbose = FALSE,
                  metric = "ROC",
                  trControl = ctrl)

##############################################################################################################

# Build up-sampled model
ctrl$sampling <- "up"
up_fit <- train(class ~ .,
                data = gbm_df,
                method = "gbm",
                verbose = FALSE,
                metric = "ROC",
                trControl = ctrl)

##############################################################################################################

# Define function test ROC
test_roc <- function(model, data) {
  
  # Compute roc values for predictions
  roc(data$class, predict(model, data, type = "prob")[, "Sim"])
  
}

##############################################################################################################

# Define a list of models results
model_list <- list(ponderado = weighted_fit,
                   reduzido = down_fit,
                   ampliado = up_fit,
                   original = orig_fit)

# Inpect results
model_list_roc <- model_list %>%
  map(test_roc, data = train_test)

# Inspect auc results
model_list_roc %>%
  map(auc)

##############################################################################################################

# Define list of results
results_list_roc <- list(NA)

# Define counter number of models
num_mod <- 1

# Loop over the modelsa
for(the_roc in model_list_roc){
  
  # Populate list of results
  results_list_roc[[num_mod]] <- tibble(tpr = the_roc$sensitivities,
                                        fpr = 1 - the_roc$specificities,
                                        model = names(model_list)[num_mod])
  
  # Increment counter
  num_mod <- num_mod + 1
  
}

# Stack rows of results
results_df_roc <- bind_rows(results_list_roc)


# Define custom colors
custom_col <- c("#0078d7", "#009E73", "#0072B2", "#0078d7")

# Plot ROC curve for all 4 models
ggplot(aes(x = fpr,  y = tpr, group = model), data = results_df_roc) +
  geom_line(aes(color = model)) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  labs(title = "Curva ROC dos Modelos Ajustados")


# Stack rows of results
results_df_roc <- bind_rows(results_list_roc) %>%
  filter(model == "weighted" ) %>% 
  select(tpr, fpr) %>% 
  rename(`Taxa de Verdadeiro Positivo` = tpr,
         `Taxa de Falso Positivo` = fpr)


# Define custom colors
custom_col <- c("#0078d7", "#009E73", "#0072B2", "#D55E00", "#CC79A7")

# Plot ROC curve for all 5 models
ggplot(aes(x = `Taxa de Falso Positivo`,  y = `Taxa de Verdadeiro Positivo`),
       data = results_df_roc) +
  geom_line(color = "#0078d7") +
  geom_abline(intercept = 0, slope = 1, color = "#151E3D", size = 1) +
  labs(title = "Curva ROC do Modelo Ajustado")

##############################################################################################################

# Count variables

##############################################################################################################

# Transform data frame with 200.000 x 200 to vactor with 40000000 observations
data_vec <- tibble(class = rep(Y, 200),
                   var = as.vector(unlist(X_scale)))

# Find unique values in all variables
uniques <- unique(data_vec)

# Count unique values
table_data_vec <- table(data_vec)

# Melt table
df_train <- melt(table_data_vec) %>% as_tibble()


##############################################################################################################

# Logistic Regression

##############################################################################################################

# Define models train and test
split <- round(nrow(df_model) * 0.8)
train_train <- df_model[1:split,]
train_test <- df_model[split:nrow(df_model),]

# Print Model
model <- glm(class ~ .,
             data = train_train,
             family = "binomial")

##############################################################################################################

# Define target anf feature
predicted <- predict(model, newdata = train_test[,-1], type = "response")
targets <- train_test$class

# Calculate confusion matrix
conf_mat <- confusionMatrix(as.factor(ifelse(predicted > 0.5, 1, 0)),
                            as.factor(targets))


# Draw the matrix
draw_confusion_matrix(conf_mat)

##############################################################################################################

# Calculate roc curve
roc_curve <- roc(train_test$class,
                 predict(model,
                         newdata = train_test[,-1],
                         type = "response"))

# Populate list of results
results_roc <- tibble(tpr = roc_curve$sensitivities,
                      fpr = 1 - roc_curve$specificities)


# Plot ROC curve for all 4 models
ggplot(aes(x = fpr,  y = tpr), data = results_roc) +
  geom_line(color = "#0078d7") +
  geom_abline(intercept = 0, slope = 1, color = "#151E3D", size = 1) +
  labs(title = "Curva ROC do Modelo Ajustado")

##############################################################################################################

# Define X atributes variable
Y_scale <- test %>% select(var_0:var_199) %>%  scale() %>% as_tibble()

# Prediction
predicted_test <- predict(model, newdata = Y_scale, type = "response")

# Update submissions  
submission$target <- ifelse(predicted_test > 0.5, 1, 0)

# Write csv to submission
write_csv(submission, "data/sample_submission.csv")



##############################################################################################################






