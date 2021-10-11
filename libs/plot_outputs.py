import numpy as np
import matplotlib.pyplot as plt
import loaddata as d

cores = [
                "black",
                "blue",
                "green",
                "red",
                "grey",
                "brown",
                "violet",
                "seagreen",
                "dimgray",
                "darkorange",
                "royalblue",
                "darkviolet",
                "maroon",
                "yellowgreen",
                "peru",
                "steelblue",
                "crimson",
                "silver",
                "magenta",
                "yellow",
            ]

fig, ax = plt.subplots()
initial_scores = np.load(d.path+"output/initial_scores.npy")
final_scores = np.load(d.path+"output/final_scores.npy")
f= 0.2
x = [bn -f for bn in range(d.nbnd)]
x_ = [bn +f for bn in range(d.nbnd)]
ax.bar(x,initial_scores/d.nks,width=2*f,color="lightblue",label='Initial State')
ax.bar(x_,final_scores/d.nks,width=2*f,color="darkorange",label='Final State')
ax.set_ylim(0.5,1)
ax.set_xticks(range(d.nbnd))
ax.set_xticklabels([f'band ${i}$' for i in range(d.nbnd)],rotation=45)
ax.legend()

fig2,ax2 = plt.subplots()
initial_bandscores = np.load(d.path+"output/initial_bandScores.npy")
final_bandscores = np.load(d.path+"output/final_bandScores.npy")
f= 0.2
x = [bn -f for bn in range(d.nbnd)]
x_ = [bn +f for bn in range(d.nbnd)]
ax2.bar(x,initial_bandscores,width=2*f,color="lightblue",label='Initial State')
ax2.bar(x_,final_bandscores,width=2*f,color="darkorange",label='Final State')
ax2.set_xticks(range(d.nbnd))
ax2.set_xticklabels([f'band ${i}$' for i in range(d.nbnd)],rotation=45)
ax2.legend()
plt.show()
fig.savefig(d.path+'output/scores.pgf')
fig2.savefig(d.path+'output/bandscores.pgf')
