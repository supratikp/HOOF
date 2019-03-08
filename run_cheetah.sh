#!/bin/bash
# a2c script

EXP='HalfCheetah-v2'
KL=0.03

	# algo_type max_kl(ignored for vanilla) random_start
	python run_mod_a2c.py $EXP vanilla 1.0 0 &
	python run_mod_a2c.py $EXP vanilla 1.0 1 &
	python run_mod_a2c.py $EXP vanilla 1.0 2 &
	python run_mod_a2c.py $EXP vanilla 1.0 3 &
	python run_mod_a2c.py $EXP vanilla 1.0 4 &
	python run_mod_a2c.py $EXP vanilla 1.0 5 &
	python run_mod_a2c.py $EXP vanilla 1.0 6 &
	python run_mod_a2c.py $EXP vanilla 1.0 7 &
	python run_mod_a2c.py $EXP vanilla 1.0 8 &
	python run_mod_a2c.py $EXP vanilla 1.0 9 &
	
	wait
	 
	python run_mod_a2c.py $EXP hoof_full $KL 0 &
	python run_mod_a2c.py $EXP hoof_full $KL 1 &
	python run_mod_a2c.py $EXP hoof_full $KL 2 &
	python run_mod_a2c.py $EXP hoof_full $KL 3 &
	python run_mod_a2c.py $EXP hoof_full $KL 4 &
	python run_mod_a2c.py $EXP hoof_full $KL 5 &
	python run_mod_a2c.py $EXP hoof_full $KL 6 &
	python run_mod_a2c.py $EXP hoof_full $KL 7 &
	python run_mod_a2c.py $EXP hoof_full $KL 8 &
	python run_mod_a2c.py $EXP hoof_full $KL 9 &

	wait

	python run_mod_a2c.py $EXP hoof_additive $KL 0 &
	python run_mod_a2c.py $EXP hoof_additive $KL 1 &
	python run_mod_a2c.py $EXP hoof_additive $KL 2 &
	python run_mod_a2c.py $EXP hoof_additive $KL 3 &
	python run_mod_a2c.py $EXP hoof_additive $KL 4 &
	python run_mod_a2c.py $EXP hoof_additive $KL 5 &
	python run_mod_a2c.py $EXP hoof_additive $KL 6 &
	python run_mod_a2c.py $EXP hoof_additive $KL 7 &
	python run_mod_a2c.py $EXP hoof_additive $KL 8 &
	python run_mod_a2c.py $EXP hoof_additive $KL 9 &

	wait

done
