# Fast Efficient Hyperparameter Tuning for Policy Gradients (https://arxiv.org/abs/1902.06583)
Implementation of HOOF for A2C and TNPG 
The code is based on OpenAI Baselines implementation: https://github.com/openai/baselines

To run the code: 
1. Add your MuJoCo key to the folder 
2. Build the docker with build.sh and then run it with run.sh
3. The parameters for each environment is in the yaml file
3. Run the code with run_A2C.sh or run_TRPO_TNPG.sh with the yaml filename as argument
4. For other environments, just create a new yaml file and add it to the folder
And that's it!
