#!/bin/bash
# a2c script

EXP=$1

# vanilla A2C with RMSProp
for i in {0..9}
do
	python a2c_experiments.py $EXP Vanilla_A2C RMSProp -1 $i &
done
wait

# vanilla A2C with SGD
for i in {0..9}
do
	python a2c_experiments.py $EXP Vanilla_A2C SGD -1 $i &
done
wait

# HOOF LR A2C SGD with max KL = 0.03
for i in {0..9}
do
	python a2c_experiments.py $EXP HOOF_A2C_LR SGD 0.03 $i &
done
wait

# HOOF LR A2C RMSProp with different KL constraints
for kl in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 -1.0
do
    for i in {0..9}
    do
    	python a2c_experiments.py $EXP HOOF_A2C_LR RMSProp $kl $i &
    done
    wait
done

# HOOF ENt+LR A2C RMSProp with max KL = 0.03
for i in {0..9}
do
	python a2c_experiments.py $EXP HOOF_A2C_Ent_LR RMSProp 0.03 $i &
done
wait
