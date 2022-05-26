library(reshape2)
library(sdamr)
library(lme4)
library(afex)
library(corrplot)

data = read.csv('C:\\Users\\vbtes\\CompProjects\\vbtCogSci\\features_of_agency\\data\\datasets_prolific_id\\properties_1\\datasets_csv\\task_data.csv')
colnames(data)


mod = lm(agency ~ movement + energy + replication + complexity + learning + reaction + mistakes + communication + variety + monitoring, data=data)
summary(mod)

mod = lm(goal_setting ~ movement + energy + replication + complexity + learning + reaction + mistakes + communication + variety + monitoring, data=data)
summary(mod)

mod = lm(goal_directedness ~ movement + energy + replication + complexity + learning + reaction + mistakes + communication + variety + monitoring, data=data)
summary(mod)

mod = lm(freewill ~ movement + energy + replication + complexity + learning + reaction + mistakes + communication + variety + monitoring, data=data)
summary(mod)



properties = cor(data[colnames(data)[6:19]])
corr_prop = cor(properties)
cor.mtest <- function(mat, ...) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}
p.mat = cor.mtest(properties)
corrplot(corr_prop,p.mat = p.mat, sig.level = 0.01)


