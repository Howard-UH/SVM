library(caTools)
library(e1071)
library(rminer)
library(ggplot2)
library(lattice)
library(caret)
library(survival)
library(pROC)
library(kernlab)

dat <- read.csv("AIDS_DATA.csv", header = TRUE)
dat <- dat[,-1]

is.na(dat)  # 檢查遺失值#rm移除
# 統計第20欄位的遺失值數量
missing_values_count <- sum(is.na(dat[, 20]))

# 輸出遺失值數量
print(paste("第20欄位的遺失值數量：", missing_values_count))

na.row <- is.na(dat[, 20])
mean.age <- mean(dat[, 20], na.rm = TRUE)
dat[na.row, 20] <- round(mean.age)  # 平均數放回遺失值

dat$cens <- as.factor(dat$cens)

# 設定隨機種子，以便重現結果
set.seed(1234)

split_data <- sample.split(dat$cens, SplitRatio = 0.7)

# 取出 70% 的觀測值作為訓練集
data_train <- subset(dat, split_data == TRUE)

# 將剩下的 30% 觀測值作為測試集
data_test <- subset(dat, split_data == FALSE)

dim(data_train); dim(data_test)

# values that cost will iterate over
cost_list <- as.list(c(10^(-5:5)))

# create an empty list that will store the fraction of prediction that match the actual prediction corresponding to the value of C
prediction_fraction_linear <- c()

for(i in seq_along(cost_list)){
  c <- cost_list[[i]]
  
  model_linear <- ksvm(as.matrix(data_train[,-24]), data_train[,24],
                       type = "C-svc", kernel = "vanilladot", C = c, scaled = TRUE)
  
  a_linear <- colSums(model_linear@xmatrix[[1]] * model_linear@coef[[1]])
  
  a0_linear <- model_linear@b
  
  pred_linear <- predict(model_linear, data_test[,-24])
  
  prediction_fraction_linear[i] <- sum(pred_linear == data_test[,24]) / nrow(data_test)
  
}

warnings()
# output
do.call(rbind, Map(data.frame, C = cost_list,
                   pred_ksvm_linear = prediction_fraction_linear))

sprintf("c= %f provides the best prediction of %f",
        cost_list[which(prediction_fraction_linear ==
                          max(prediction_fraction_linear))[1]], max(prediction_fraction_linear))

best_model_linear <- ksvm(as.matrix(dat[,-24]),
                          dat[,24], type = "C-svc", kernel = 'vanilladot',
                          C = cost_list[which(prediction_fraction_linear == max(prediction_fraction_linear))[1]], scaled = TRUE)
a_linear_best <- colSums(best_model_linear@xmatrix[[1]] * best_model_linear@coef[[1]])

print(a_linear_best)

a0_linear_best <- best_model_linear@b

print(a0_linear_best)
# 上面是線性

# 線性
# 預測模型
predictions <- predict(best_model_linear, data_test[,-24])

# 計算混淆矩陣
confusion_matrix <- confusionMatrix(predictions, data_test[,24])
confusion_matrix
# 計算第一類目標變數（0）的Recall和F1-score
precision_class_0 <- confusion_matrix$byClass['Pos Pred Value'][1]
recall_class_0 <- confusion_matrix$byClass['Sensitivity'][1]
f1_score_class_0 <- confusion_matrix$byClass['F1'][1]

# 計算第二類目標變數（1）的Recall和F1-score 
precision_class_1 <- confusion_matrix$byClass['Neg Pred Value'][1]
recall_class_1 <- confusion_matrix$byClass['Specificity'][1]
f1_score_class_1 <- 2 * precision_class_1 * recall_class_1 / (precision_class_1 + recall_class_1)

print(paste("對於類別 0 精確度：", precision_class_0))
print(paste("對於類別 0 召回率：", recall_class_0))
print(paste("對於類別 0 F1-score：", f1_score_class_0))

print(paste("對於類別 1 精確度：", precision_class_1))
print(paste("對於類別 1 召回率：", recall_class_1))
print(paste("對於類別 1 F1-score：", f1_score_class_1))

# ROC
# 使用線性模型預測機率
predictions <- predict(best_model_linear, data_test[,-24])
class(predictions)
length(predictions)
predictions <- as.numeric(as.character(predictions))

# 檢查 test$cens 的數據類型和長度
class(data_test$cens)
length(data_test$cens)

# 將 test$cens 轉換為數值型
data_test$cens <- as.numeric(as.character(data_test$cens))

# 計算真正率和假正率
roc_obj <- roc(data_test$cens, predictions)

# 繪製ROC曲線
par(pty = 's')
roc(data_test$cens, predictions, plot = TRUE, legacy.axes = TRUE, percent = TRUE, main = "ROC Curve", xlab = "False Positive Percentage", ylab = "True Positive Percentage", col = "#377eb8", lwd = 4, print.auc = TRUE)

# 添加模型類型的標籤
text(40, 0.7, "Model: Linear", col = "red")


