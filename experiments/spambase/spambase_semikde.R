#!/usr/bin/env Rscript
library(parallelMCMCcombine)
library(reticulate)
np = import("numpy")

set.seed(1)
times = c()
for (i in 10:14){
  shard_samples = np$transpose(np$load(sprintf("experiments/spambase/data/shard_samples_seed%s.npy", i)))
  set.seed(10+i)
  t = Sys.time()
  semikde_samps = semiparamDPE(shard_samples)
  samptime = difftime(Sys.time(), t, units='secs')[[1]]
  print(samptime)
  times = c(times, samptime)
  
  np$save(sprintf("experiments/spambase/data/samples_seed%s_semikde.npy", i), np$transpose(semikde_samps))
}

np$save("experiments/spambase/data/times_semikde.npy", times)
