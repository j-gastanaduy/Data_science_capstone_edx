
##libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(Rborist)) install.packages("Rborist", repos = "http://cran.us.r-project.org")
if(!require(ggrepel)) install.packages("ggrepel", repos = "http://cran.us.r-project.org")
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")


# IBM HR Analytics Employee Attrition & Performance Dataset
tb<- read_csv("https://github.com/j-gastanaduy/Data_science_capstone_edx/raw/master/HR-Employee-Attrition.csv",
              col_types = cols(Attrition = col_factor(levels = c("No", "Yes")),
                               BusinessTravel = col_factor(levels = c("Non-Travel", "Travel_Rarely", "Travel_Frequently")),
                               Department = col_factor(),
                               EducationField= col_factor(),
                               Gender= col_factor(levels = c("Female", "Male")),
                               JobRole= col_factor(),
                               MaritalStatus= col_factor(),
                               Over18= col_factor(),
                               OverTime= col_factor(levels = c("No", "Yes"))))
tb<- tb%>%
  mutate(Attrition = fct_recode(Attrition, "0"="No", "1"="Yes"))


##data exploration
prop.table(table(tb$Attrition))


##create train and validation sets
set.seed(1)
index<- createDataPartition(y= tb$Attrition, times = 1, p = 0.8, list = FALSE)
train_set<- tb[index, ]
test_set<- tb[-index, ]


##preprocessing train_set
nearZeroVar(tb)

train_set<- train_set%>%
  select(-c("EmployeeCount", "Over18", "StandardHours"))


##knn
set.seed(5)
train_knn<- train(Attrition ~ ., data = train_set,
                  method = "knn",
                  tuneGrid = data.frame(k = seq(5, 100, 5)),
                  trControl= trainControl(method = "cv", number = 10, p = .9))
confusionMatrix(predict(train_knn, test_set), test_set$Attrition, positive = "1")


##classification tree
train_rpart<- train(Attrition ~ ., data= train_set,
                    method= "rpart",
                    tuneGrid = data.frame(cp = seq(0, 0.1, length.out = 25)))

ggplot(train_rpart, highlight = TRUE)
confusionMatrix(predict(train_rpart, test_set), test_set$Attrition, positive = "1")


##random forest
set.seed(51)
train_rf<- train(Attrition ~., data = train_set,
                 method= "Rborist",
                 nTree= 50,
                 trControl= trainControl(method = "cv", number = 5, p = .9),
                 tuneGrid= data.frame(predFixed = seq(2, 20, 2),
                                      minNode = c(12)))
ggplot(train_rf, highlight = TRUE)
confusionMatrix(predict(train_rf, test_set), test_set$Attrition, positive = "1")



set.seed(1023)
fit_rf <- train(Attrition ~., data = train_set,
                method= "Rborist",
                nTree= 1000,
                tuneGrid= data.frame(predFixed = c(12), minNode = c(12)))

confusionMatrix(predict(fit_rf, test_set), test_set$Attrition, positive = "1")


##logistic regression
train_glm<- glm(Attrition ~., data= train_set, family = "binomial")
p_hat_glm<- predict(train_glm, test_set, type = "response")
y_hat_glm<- ifelse(p_hat_glm > 0.5, "1", "0")%>%
  factor(levels = c("0", "1"))
confusionMatrix(y_hat_glm, test_set$Attrition, positive = "1")

train_glm2<- glm(Attrition ~ .-Department-Education-EmployeeNumber-JobLevel- HourlyRate
                 -MonthlyIncome-MonthlyRate - PercentSalaryHike- PerformanceRating- StockOptionLevel, 
                 data= train_set, family = "binomial")

p_hat_glm2<- predict(train_glm2, test_set, type = "response")
y_hat_glm2<- ifelse(p_hat_glm2 > 0.5, "1", "0")%>%
  factor(levels = c("0", "1"))
confusionMatrix(y_hat_glm2, test_set$Attrition, positive = "1")


##Precision-Recall(logistic regression)
probs <- seq(0.001, 1, length.out = 10)

method_logistic<- sapply(probs, function(p){
  y_hat<- ifelse(p_hat_glm2 > p, "1", "0")%>%
    factor(levels = c("0", "1"))
  list(method = "Logistic Regression",
       cutoff= p,
       FPR = 1-specificity(y_hat, test_set$Attrition),
       recall_TPR = sensitivity(y_hat, test_set$Attrition),
       precision = precision(y_hat, test_set$Attrition))})

tb_logistic<- data.frame(t(method_logistic))

pr_plot_logistic<-
  tb_logistic%>%
  unnest()%>%
  ggplot(aes(recall_TPR, precision, label = cutoff)) +
  geom_line() +
  geom_point()+
  geom_text_repel(nudge_x = 0.001, nudge_y = -0.001)

y_hat_glm2.2<- ifelse(p_hat_glm2 > 0.5, "1", "0")%>%
  factor(levels = c("0", "1"))
confusionMatrix(y_hat_glm2.2, test_set$Attrition, positive = "1") 


###optimizing probability cutoff in logistic regression
probs <- seq(0.001, 0.35, length.out = 12)

method_logistic<- map_df(probs, function(p){
  y_hat<- ifelse(p_hat_glm2 > p, "1", "0")%>%
    factor(levels = c("0", "1"))
  list(method = "Logistic Regression",
       cutoff= round(p,3),
       FPR = 1-specificity(y_hat, test_set$Attrition),
       recall_TPR = sensitivity(y_hat, test_set$Attrition),
       precision = precision(y_hat, test_set$Attrition))})

tb_logistic<- data.frame(t(method_logistic))

pr_plot_logistic<-
  tb_logistic%>%
  unnest()%>%
  ggplot(aes(recall_TPR, precision, label = cutoff)) +
  geom_line() +
  geom_point()+
  geom_text_repel(nudge_x = 0.001, nudge_y = -0.001)


##LDA
train_lda<- train(Attrition ~ ., data = train_set, 
                  method = "lda")
p_hat_lda<- predict(train_lda, test_set, type = "prob")
y_hat_lda<- predict(train_lda, test_set)
confusionMatrix(y_hat_lda, test_set$Attrition, positive = "1")


##Precision-Recall (lda)
probs <- seq(0.001, 0.35, length.out = 12)

method_lda<- map_df(probs, function(p){
  y_hat<- ifelse(p_hat_lda[,2] > p, "1", "0")%>%
    factor(levels = c("0", "1"))
  list(method = "LDA",
       cutoff= round(p,3),
       FPR = 1-specificity(y_hat, test_set$Attrition),
       recall_TPR = sensitivity(y_hat, test_set$Attrition),
       precision = precision(y_hat, test_set$Attrition))})


##ensemble lda + logistic
p_ensemble<- (p_hat_glm2+p_hat_lda[,2])/2

probs <- seq(0.001, 0.35, length.out = 12)

method_ensemble<- map_df(probs, function(p){
  y_hat<- ifelse(p_ensemble> p, "1", "0")%>%
    factor(levels = c("0", "1"))
  list(method = "Ensemble",
       cutoff= round(p,3),
       FPR = 1-specificity(y_hat, test_set$Attrition),
       recall_TPR = sensitivity(y_hat, test_set$Attrition),
       precision = precision(y_hat, test_set$Attrition))})
 

###precision-recall plots
pr_plots<-
bind_rows(method_logistic, method_lda, method_ensemble)%>%
  ggplot(aes(recall_TPR, precision, color= method))+
  geom_line() +
  geom_point()


##choosing probability cutoffs
method_ensemble[4,]
method_ensemble[5,]
method_logistic[6,]
method_lda[3,]
method_lda[7,]


##evaluation metrics of final models
y_hat_glm2.3<- ifelse(p_hat_glm2 > 0.16, "1", "0")%>%
  factor(levels = c("0", "1"))
metrics_logistic<- confusionMatrix(y_hat_glm2.3, test_set$Attrition, positive = "1") 

y_hat_lda2<- ifelse(p_hat_lda[2] > 0.191, "1", "0")%>%
  factor(levels = c("0", "1"))
metrics_lda<- confusionMatrix(y_hat_lda2, test_set$Attrition, positive = "1")

y_hat_ensemble<- ifelse(p_ensemble > 0.128, "1", "0")%>%
  factor(levels = c("0", "1"))
metrics_ensemble<- confusionMatrix(y_hat_ensemble, test_set$Attrition, positive = "1")


