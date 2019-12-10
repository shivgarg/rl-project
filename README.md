# Environment Setup
Source code is compatible with python3.
## Requirements

pip install -r requirements.txt

## Training and Testing Models
```
python -u <algorithm>.py | tee <logfile>
```
algorithm can be baseline, dqn, ppo, sac, sac_sep. The following steps are executed by each of the files.
- Each run creates 100 training games with a predefined seed. 
- Then the agent is trained for 2000 epsiodes.
- 20 testing games are created
- Test results are printed

## Visualizing training run
To visualize the training run, run the following command:
```
python visualize_log.py --logfile <training log file > --out <out graph file name prefix>
```