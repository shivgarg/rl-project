import argparse
import matplotlib.pyplot as plt
import pickle

args = argparse.ArgumentParser()
args.add_argument("--out",type=str,required=True,help="output file names")
args = args.parse_args()


max_scores = {}
avg_scores = {}
len_episode = {}

FILE_LIST = ['dqn.pkl','ppo.pkl','sac.pkl','sac_sep.pkl','baseline.pkl']
for f in FILE_LIST:
  max_s, avg_s, epi_s = pickle.load(open(f,'rb'))
  max_scores[f.replace(".pkl","")] = max_s
  avg_scores[f.replace(".pkl","")] = avg_s
  len_episode[f.replace(".pkl","")] = epi_s

plt.figure()
legend = []
for key in max_scores:
    plt.plot(range(len(max_scores[key])), max_scores[key])
    legend.append(key)
plt.legend(legend, loc='lower right')
plt.xlabel("No. of Episodes")
plt.ylabel("Max Scores")
plt.title("Max Scores")
plt.savefig(args.out+"_max.png")

plt.figure()
for key in avg_scores:
    plt.plot(range(len(avg_scores[key])), avg_scores[key])
plt.legend(legend, loc='lower right')
plt.xlabel("No. of Episodes")
plt.ylabel("Avg Scores")
plt.title("Avg Scores")
plt.savefig(args.out+"_avg.png")

plt.figure()
for key in len_episode:
    plt.plot(range(len(len_episode[key])), len_episode[key])
plt.legend(legend, loc='lower right')
plt.xlabel("No. of Episodes")
plt.ylabel("Episode Length")
plt.title("Episode Length")
plt.savefig(args.out+"_len.png")

