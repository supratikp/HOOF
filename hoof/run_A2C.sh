#!/bin/bash
# a2c script

EXP=$1

for i in {0..9}
do
	python exp_A2C.py $EXP RMSProp_Baseline_A2C $i 40 &
done
wait

for i in {0..9}
do
	python exp_A2C.py $EXP SGD_Baseline_A2C $i 40 &
done
wait

for i in {0..9}
do
	python exp_A2C.py $EXP A2C_RMSProp_HOOF_LR $i 40 &
done
wait

for i in {0..9}
do
	python exp_A2C.py $EXP A2C_SGD_HOOF_LR $i 40 &
done
wait