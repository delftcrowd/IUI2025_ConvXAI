import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util import load_user_data
import sys

valid_users, tp_data, _ = load_user_data(filename="../data/xailabdata_all.csv")
valid_users_2, tp_data_2, _ = load_user_data(filename="../data/xailabdata_llm_agent.csv")
valid_users = valid_users | valid_users_2
tp_data.update(tp_data_2)
variable_dict = {}
variable_dict["condition"] = []
dimensions = ["completeness", "coherence", "clarity", "usefulness_explanation"]

condition_count = {}
for user in valid_users:
    tp_condition = tp_data[user]["condition"]
    if tp_condition not in condition_count:
        condition_count[tp_condition] = 0
    condition_count[tp_condition] += 1
print(condition_count)

name_map = {
    "completeness": "Explanation\nCompleteness",
    "coherence": "Explanation\nCoherence",
    "clarity": "Explanation\nClarity",
    "usefulness_explanation": "Explanation\nUsefulness",
}
condition_map = {
    "control": 'Control',
    "dashboard": 'Dashboard',
    "chatxai": 'CXAI',
    "chatxaiboost": 'ECXAI',
    "chatxaiAuto": 'LLM Agent',
}
data_long_format = {
    "condition": [],
    "measure": [],
    "value": []
}
for user in valid_users:
    if tp_data[user]["condition"] == "control":
        continue
    explanation_understanding = tp_data[user]["explanation_understanding"]
    for variable in dimensions:
        data_long_format["condition"].append(condition_map[tp_data[user]["condition"]])
        data_long_format["measure"].append(name_map[variable])
        data_long_format["value"].append(explanation_understanding[variable])
df = pd.DataFrame(data_long_format, dtype=float)

size=24
params = {'axes.labelsize': size,
            'axes.titlesize': size,
            'xtick.labelsize': size*0.75,
            'ytick.labelsize': size*0.75}
plt.rcParams.update(params)
sns.set_theme(style="whitegrid")
fig = plt.gcf()

color_palette = sns.color_palette("Spectral_r", n_colors=5)[1:]
print(color_palette)
ax = sns.barplot(data=df, x="measure", y="value", hue="condition", hue_order=['Dashboard', 'CXAI', 'ECXAI', 'LLM Agent'], palette=color_palette, errwidth=0.5, capsize=.1)
# , ci=None


ax.set_ylim([3.0, 4.4])
ax.tick_params(labelsize=13)
ax.set_xlabel("Explanation Utility", fontsize = 24)
ax.set_ylabel("Value", fontsize = 24)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()
fig.savefig("explanation_utility_bar_plot.pdf", format='pdf')