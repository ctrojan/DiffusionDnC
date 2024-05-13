#!/usr/bin/env Rscript
library(parallelMCMCcombine)
library(reticulate)
np = import("numpy")

shard_samples = np$transpose(np$load("experiments/mixture_of_gaussians/data/shard_samples.npy"))

set.seed(1)
t = Sys.time()
semikde_samps = semiparamDPE(shard_samples)
samptime = Sys.time()-t
np$save("experiments/mixture_of_gaussians/data/samples_semikde.npy", np$transpose(semikde_samps))
np$save("experiments/mixture_of_gaussians/data/time_semikde.npy", samptime)