#!/bin/bash
conda activate pvyis

GPU=1
dataset=biased_mnist
optim=biased_mnist

for p in 0.95; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=resnet18 \
  trainer=csgmcmc_trainer \
  dataset=${dataset} \
  dataset.p_bias=${p} \
  optimizer=${optim} \
  expt_suffix=p_${p}

  # CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  # model.name=occam_resnet18 \
  # trainer=occam_trainer \
  # dataset=${dataset} \
  # dataset.p_bias=${p} \
  # optimizer=${optim} \
  # expt_suffix=p_${p}
done