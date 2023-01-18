library(reshape2)
library(sdamr)
library(lme4)
library(afex)
library(corrplot)
library(CAM)

data = read.csv('C:\\Users\\vbtes\\CompProjects\\vbtCogSci\\features_of_agency\\data\\datasets_prolific_id\\properties_1\\datasets_csv\\task_data.csv')
colnames(data)
dim(data)
data$cluster = as.factor(data$cluster)


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
p.mat = cor.mtest(data[colnames(data)[6:19]])
corrplot(properties,p.mat = p.mat, sig.level = 0.01)


# LMERs

# WORDS
mod = afex::mixed(agency ~ movement + energy + replication + complexity + 
                    learning + reaction + mistakes + communication + variety + 
                    monitoring + 
                    (movement + energy + replication + complexity + 
                       learning + reaction + mistakes + communication + variety + 
                       monitoring||Participant.Public.ID) + 
                    (movement + energy + replication + complexity + 
                       learning + reaction + mistakes + communication + variety + 
                       monitoring||word), data=data, method = "S", expand_re = TRUE)
mod
summary(mod)

# CLUSTERS
mod = afex::mixed(agency ~ movement + energy + replication + complexity + 
                    learning + reaction + mistakes + communication + variety + 
                    monitoring +
                    (movement + energy + replication + complexity + 
                       learning + reaction + mistakes + communication + variety + 
                       monitoring||Participant.Public.ID) +
                    (movement + energy + replication + complexity + 
                       learning + reaction + mistakes + communication + variety + 
                       monitoring||cluster), data=data, method = "S", expand_re = TRUE)
mod
summary(mod)
mod$full_model



mod = afex::mixed(freewill ~ movement + energy + replication + complexity + 
                    learning + reaction + mistakes + communication + variety + 
                    monitoring + 
                    (movement + energy + replication + complexity + 
                       learning + reaction + mistakes + communication + variety + 
                       monitoring||Participant.Public.ID) + 
                    (movement + energy + replication + complexity + 
                       learning + reaction + mistakes + communication + variety + 
                       monitoring||word), data=data, method = "S", expand_re = TRUE)
mod
summary(mod)

mod = afex::mixed(agency ~ goal_directedness + goal_setting + freewill + 
                    (goal_directedness + goal_setting + freewill||Participant.Public.ID) + 
                    (goal_directedness + goal_setting + freewill||word), data=data, method = "S", expand_re = TRUE)
mod
summary(mod)

mod = afex::mixed(movement ~ agency + energy + replication + complexity + learning + reaction + mistakes + communication + variety + monitoring + (1|word), data=data)
mod

mod = afex::mixed(agency ~ movement + energy + replication + complexity + learning + reaction + mistakes + communication + variety + monitoring + (1+movement + energy + replication + complexity + learning + reaction + mistakes + communication + variety + monitoring|Participant.Public.ID), data=data)
mod

