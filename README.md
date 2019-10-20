# Fast Efficient Hyperparameter Tuning for Policy Gradients (https://arxiv.org/abs/1902.06583)
Implementation of HOOF for A2C and TNPG.
The code is based on OpenAI Baselines implementation: https://github.com/openai/baselines

To run the code: 
1. Add your MuJoCo key to the folder 
2. Build the docker with build.sh and then run it with run.sh
3. The parameters for each environment is in the yaml file
3. Run the code with run_all_a2c_experiments.sh or run_all_npg_experiments.sh with the (shortened) env name as argument (see the yaml file)
4. Use plots_a2c.py or plot_tnpg_and_hypers.py to plot the results
And that's it!
