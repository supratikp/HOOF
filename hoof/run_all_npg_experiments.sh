#!/bin/bash
# a2c script

EXP=$1

# Vanilla TRPO
for i in {0..9}
do
	python npg_experiments.py $EXP TRPO $i &
done
wait

# HOOF NPG - gamma, lambda, KL constraint
for i in {0..9}
do
	python npg_experiments.py $EXP HOOF_All $i &
done
wait


# HOOF NPG - gamma, lambda, KL constraint without gamma/lambda conditioned vf
for i in {0..9}
do
	python npg_experiments.py $EXP HOOF_no_lamgam $i &
done
wait
