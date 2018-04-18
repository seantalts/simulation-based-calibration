#setwd("~/Documents/sbc/Rcode")
set.seed(666)


library(doParallel)
library(foreach)

cl <- makeCluster(50)
registerDoParallel(cl)

print(getDoParWorkers())

load("sbc_inla.RData")
source("generate_space.R")
rrr=foreach(dummy=c(1:1000), .combine = rbind) %dopar% {
library(INLA)
sbc()
}

stopCluster(cl)

save(rrr, file = "output.RData")

# 
# n_rep=10
# 
# 
# for (j in 6:95){
# ranks = matrix(NA,n_rep,length(points.mc))
# print(j)
# for (i in 1:n_rep) {
# 
#   inla_seed = as.integer(runif(1)*.Machine$integer.max)
#  
#   dat = generate_data(spde, A.gen, N, m)
#   logit_prev_true = dat$logit_intercept+ dat$field
#   functional_true = functionals(logit_prev_true,mesh,points.mc)
#   ranks[i,] = rank_functionals(dat$y,N,spde,A.est,m,functional_true,0L)
# }
# 
# assign(paste("ranks_",j,"_10",sep=""),ranks)
# rr = rbind(rr,ranks)
# save(list=paste("ranks_",j,"_10",sep=""),file=paste("rank",j,"_10.RData",sep=""))
# }
