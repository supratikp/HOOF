#!/bin/bash
# a2c script

EXP=$1

for i in {0..9}
do
	python exp_TRPO_TNPG.py $EXP Baseline_TRPO $i &
done
wait

for i in {0..9}
do
	python exp_TRPO_TNPG.py $EXP TNPG_HOOF_All $i &
done
wait