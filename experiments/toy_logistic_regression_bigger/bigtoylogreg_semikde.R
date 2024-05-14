#!/usr/bin/env Rscript
library(parallelMCMCcombine)
library(reticulate)
np = import("numpy")

set.seed(1)
times = c()
for (i in 0:4){
  shard_samples = np$transpose(np$load(sprintf("experiments/toy_logistic_regression_bigger/data/shard_samples_seed%s.npy", i)))
  set.seed(10+i)
  t = Sys.time()
  semikde_samps = semiparamDPE(shard_samples)
  samptime = difftime(Sys.time(), t, units='secs')[[1]]
  print(samptime)
  times = c(times, samptime)
  
  np$save(sprintf("experiments/toy_logistic_regression_bigger/data/samples_semikde_seed%s.npy", i), np$transpose(semikde_samps))
}

np$save("experiments/toy_logistic_regression_bigger/data/times_semikde.npy", times)
