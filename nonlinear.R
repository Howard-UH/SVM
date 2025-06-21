library(caTools)
library(e1071)
library(rminer)
library(ggplot2)
library(lattice)
library(caret)
library(survival)
library(pROC)
library(kernlab)


dat = read.csv("AIDS_DATA.csv", header = TRUE)
dat<-dat[,-1]

is.na(dat)#檢查遺失值#rm移除
# 統計第20欄位的遺失值數量
missing_values_count <- sum(is.na(dat[, 20]))

# 輸出遺失值數量
print(paste("第20欄位的遺失值數量：", missing_values_count))

na.row<-is.na(dat[,20])
mean.age<-mean(dat[,20], na.rm=T)
dat[na.row,20]<-round(mean.age)#平均數放回遺失值


dat$cens=as.factor(dat$cens)
table(dat$cens)

# 設定隨機種子，以便重現結果
set.seed(1234)

split_data <- sample.split(dat$cens,SplitRatio = 0.7)

# 取出 70% 的觀測值作為訓練集
data_train <- subset(dat, split_data == T)

# 將剩下的 30% 觀測值作為測試集
data_test <- subset(dat, split_data == F)

dim(data_train);dim(data_test)

#values that cost will iterate over
cost_list <- as.list(c(10^(-5:5)))


#create an empty list that will store the fraction of prediction that match the actual prediction corresponding to the value of C
prediction_fraction_non_linear <- c()

for(i in seq_along(cost_list)){
  c <- cost_list[[i]]
  
  model_non_linear <- ksvm(as.matrix(data_train[,-24]),data_train[,24],
                           type="C-svc",kernel="rbfdot",C=c,scaled=TRUE)
  
  a_nonlinear <- colSums(model_non_linear@xmatrix[[1]] *model_non_linear@coef[[1]] )
  
  a0_nonlinear <- model_non_linear@b
  
  pred_non_linear <- predict(model_non_linear,data_test[,-24])
  
  prediction_fraction_non_linear[i] <- sum(pred_non_linear==data_test[,24])/nrow(data_test)
  
  
}

warnings()
#output
do.call(rbind, Map(data.frame, C=cost_list,
                   pred_ksvm_non_linear = prediction_fraction_non_linear))

sprintf("c= %f provides the best prediction of %f",
        cost_list[which(prediction_fraction_non_linear ==
                          max(prediction_fraction_non_linear))[1]],max(prediction_fraction_non_linear))

best_model_non_linear <- ksvm(as.matrix(dat[,-24]),
                              dat[,24],type="C-svc",kernel='rbfdot',
                              C=cost_list[which(prediction_fraction_non_linear == max(prediction_fraction_non_linear))[1]],scaled=TRUE)
a_nonlinear_best <- colSums(best_model_non_linear@xmatrix[[1]] *best_model_non_linear@coef[[1]] )

print(a_nonlinear_best)


a0_nonlinear_best <- best_model_non_linear@b

print(a0_nonlinear_best)


#非線性1
# 預測模型
predictions <- predict(best_model_non_linear, data_test[,-24])

# 計算混淆矩陣
confusion_matrix <- confusionMatrix(predictions, data_test[,24])

# 計算第一類目標變數（0）的Recall和F1-score
precision_class_0 <- confusion_matrix$byClass['Pos Pred Value'][1]
recall_class_0 <- confusion_matrix$byClass['Sensitivity'][1]
f1_score_class_0 <- confusion_matrix$byClass['F1'][1]

# 計算第二類目標變數（1）的Recall和F1-score 
precision_class_1 <- confusion_matrix$byClass['Neg Pred Value'][1]
recall_class_1 <- confusion_matrix$byClass['Specificity'][1]
f1_score_class_1 <- 2 * precision_class_1 * recall_class_1 / (precision_class_1 + recall_class_1)

print(paste("For Class 0 Precision:", precision_class_0))
print(paste("For Class 0 Recall (召回率):", recall_class_0))
print(paste("For Class 0 F1-score:", f1_score_class_0))

print(paste("For Class 1 Precision:", precision_class_1))
print(paste("For Class 1 Recall (召回率):", recall_class_1))
print(paste("For Class 1 F1-score:", f1_score_class_1))



#ROC
# 使用非線性模型預測機率
predictions <- predict(best_model_non_linear, data_test[,-24])
class(predictions)
length(predictions)
predictions <- as.numeric(as.character(predictions))

# 檢查test$cens的數據類型和長度
class(data_test$cens)
length(data_test$cens)

# 將test$cens轉換為數值型
data_test$cens <- as.numeric(as.character(data_test$cens))


# 繪製ROC曲線
par(pty = 's')
roc(data_test$cens, predictions,plot=TRUE,legacy.axes=TRUE,percent=TRUE, main = "ROC Curve",xlab = "False Positive Percentage", ylab = "True Positive Percentage",col="#377eb8",lwd=4,print.auc=TRUE)

# 添加模型類型的標籤
text(40, 0.7, "Model: Nonlinear", col = "red")

# 繪製ROC曲線
library(ROCR)
rocplot=function(pred,truth,...){
  predob = prediction(pred,truth)
  perf = performance(predob,"fpr","tpr")
  plot(perf,...)
}
svmfit.opt=svm(cens~.,data=data_train,kernel="radial",gamma=0.0001,cost=10,decision.values=T)

fitted=attributes(predict(svmfit.opt, data_train, decision.values=TRUE)) $decision.values

par(mfrow=c(1,2))

rocplot (fitted, data_train[,24],main="Training Data",xlab = "False Positive Rate", ylab = "True Positive Rate",print.auc=TRUE)

svmfit.flex=svm(cens~., data=data_train, kernel="radial",gamma=0.000001, cost=10, decision.values=T)

fitted=attributes(predict(svmfit.flex, data_train,decision.values=T))$decision.values

rocplot (fitted,data_train[,24], add=T, col="red",print.auc=TRUE)

fitted=attributes (predict (svmfit.opt, data_test,decision.values=T)) $ decision.values

rocplot (fitted, data_test[,24],main="Test Data",xlab = "False Positive Rate", ylab = "True Positive Rate")

fitted=attributes(predict (svmfit.flex, data_test,decision.values=T))$decision.values

rocplot (fitted,data_test[,24], add=T, col="red")


