import argparse
import matplotlib.pyplot as plt
import pickle

args = argparse.ArgumentParser()
args.add_argument("--logfile",type=str,required=True,help="Log file to parse")
args.add_argument("--out",type=str,required=True,help="output file names")
args = args.parse_args()

logfile = open(args.logfile)

max_scores = []
avg_score = []
len_episode = []

for line in logfile:
    if line.startswith("ep"):
        line = line.split()
        length = int(line[1].split(":")[1])
        len_episode.append(length)
        average = float(line[3].split(":")[1])
        avg_score.append(average)
        maximum = float(line[2].split(":")[1])
        max_scores.append(maximum)
    elif line.startswith("Trained in"):
        break

max_scores_run_avg = []
len_episode_run_avg = []
avg = 0
epi_len = 0
WINDOW=50
for i in range(WINDOW):
    avg = (avg*i + max_scores[i])/(i+1)
    max_scores_run_avg.append(avg)
    epi_len = (epi_len*i + len_episode[i])/(i+1)
    len_episode_run_avg.append(epi_len)

for i in range(WINDOW,len(max_scores)):
    avg = (avg*WINDOW  - max_scores[i-WINDOW] + max_scores[i])/WINDOW
    max_scores_run_avg.append(avg)
    epi_len = (epi_len*WINDOW - len_episode[i-WINDOW] + len_episode[i])/WINDOW
    len_episode_run_avg.append(epi_len)

plt.figure()
plt.plot(range(len(max_scores)), max_scores_run_avg)
plt.xlabel("No. of Episodes")
plt.ylabel("Max Scores")
plt.title("Max Scores")
plt.savefig(args.out+"_max.png")

plt.figure()
plt.plot(range(len(avg_score)), avg_score)
plt.xlabel("No. of Episodes")
plt.ylabel("Avg Scores")
plt.title("Avg Scores")
plt.savefig(args.out+"_avg.png")

plt.figure()
plt.plot(range(len(len_episode)), len_episode_run_avg)
plt.xlabel("No. of Episodes")
plt.ylabel("Episode Length")
plt.title("Episode Length")
plt.savefig(args.out+"_len.png")

with open(args.out+".pkl",'wb') as f:
    pickle.dump((max_scores_run_avg, avg_score, len_episode_run_avg),f)
