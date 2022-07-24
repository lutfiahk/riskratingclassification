#library importing
library(e1071)
library(caret)
library(dplyr)

#data importing
df<-credit_datas
dim(df)
str(df)
summary(df)

#data cleaning
index<-c("pendapatan_setahun_juta", "durasi_pinjaman_bulan", "jumlah_tanggungan", "risk_rating")
datarisk<-df[index]
datarisk$risk_rating<-as.factor(datarisk$risk_rating)
str(datarisk)
head(datarisk)

#datasplitting
set.seed(123)
datasample<-sample(1:nrow(datarisk), size = 0.8*nrow(datarisk)) #80%data digunakan sbg data train
datatrain<-datarisk[datasample,]
datatest<-datarisk[-datasample,]
dim(datatrain)
dim(datatest)

#classification modeling
##using naive bayes method
modelnaive<-naiveBayes(x=datatrain %>% select(-risk_rating), y=datatrain$risk_rating)

###testing the model
predictnaive<-predict(modelnaive, datatest, type="class")
tbpredictnaive <- data.frame(datatest,predictnaive)
head(tbpredictnaive)

###evaluating prediction
naiveeval<- confusionMatrix(data=predictnaive, reference=datatest$risk_rating)
naiveeval

##using support vector machine
####classification modeling
modelsvm<-svm(risk_rating~.-risk_rating,datatrain, gamma=0.5, type="C-classification")

###testing the model
predictsvm<-predict(modelsvm, datatest, type="class")

###evaluating prediction
svmeval<- confusionMatrix(data=predictsvm, reference=datatest$risk_rating)
svmeval

##using random forest
controlrf <- trainControl (method="repeatedcv", number=5, repeats=2) #k-fold cross validation
modelrf <- train(risk_rating ~., data=datatrain, method="rf", trControl=controlrf)

###testing the model
predictrf <- predict(modelrf, newdata = datatest, type = "raw")

###evaluating prediction
rfeval<- confusionMatrix(data=predictrf, reference=datatest$risk_rating)
rfeval

##using k-nearest neighbor
###classification modeling
library(caret)
controlknn <- trainControl (method="repeatedcv", number=5, repeats=2) 
modelknn<-train(risk_rating ~ ., data = datatrain, method = "knn", trControl = controlknn, preProcess = c("center","scale"), tuneLength = 20)

##testing the model
predictknn <- predict(modelknn, newdata = datatest, type = "raw")

###evaluating the model
confusionMatrix(data = predictknn, reference=datatest$risk_rating)

##using decision tree
###classification modeling
library(rpart)
modeldt <-rpart(risk_rating~., data=datatrain, method="class")

##testing the model
predictdt <- predict(modeldt, newdata = datatest, type = "class")

###evaluating the model
confusionMatrix(data = predictdt, reference=datatest$risk_rating)
