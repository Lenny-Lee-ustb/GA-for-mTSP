from scipy import stats
import numpy as np
import pandas as pd

data_name = ["mtsp51", "mtsp100", "mtsp150", "pr76", "pr152", "pr226"]
df = pd.DataFrame(columns=['data', 'ours_dis_mean', 'baseline_dis_mean',
                           'ours_time_mean', 'baseline_time_mean', 'time_Wlcoxon_p_value',
                           "over_2CPU_time?", 'dis_Wilcoxon_p_value', 'ours_less_dis?', 'improved_by %'])
k = 0
for data in data_name:
    print("Doing Wilcoxon Sum Rank Test on the data " + data + ": ")
    df_baseline = pd.read_csv("./data/baseline_" + data + ".csv", sep=",",
                              header=0, names=['initial_distance', 'time_cost', 'global_min_dis'])
    df_ours = pd.read_csv("./data/ours_" + data + ".csv", sep=",",
                          header=0, names=['initial_distance', 'time_cost', 'global_min_dis'])

    mean_basline_time = df_baseline['time_cost'].mean()
    mean_ours_time = df_ours['time_cost'].mean()
    mean_baseline_dis = df_baseline['global_min_dis'].mean()
    mean_ours_dis = df_ours['global_min_dis'].mean()

    baseline_dis = df_baseline.iloc[:, 2].values
    ours_dis = df_ours.iloc[:, 2].values
    baselin_time = df_baseline.iloc[:, 1].values
    ours_time = df_ours.iloc[:, 1].values

    res_time = stats.wilcoxon(2*ours_time, baselin_time, alternative='greater')
    res_dis = stats.wilcoxon(baseline_dis, ours_dis, alternative='greater')
    df.loc[k] = [data, round(mean_ours_dis, 2), round(mean_baseline_dis, 2), round(mean_ours_time, 2), round(mean_basline_time, 2),
                 res_time.pvalue, True if res_time.pvalue > 0.05 else False,
                 res_dis.pvalue, True if res_dis.pvalue < 0.05 else False,
                 round((mean_baseline_dis-mean_ours_dis)/mean_baseline_dis*100, 2)]
    k = k+1
df.to_csv("analysis.csv")


'''practice'''
# x = np.arange(30)
# y = np.arange(30, 60, 1)

# print(x)
# print(y)

# res = stats.mannwhitneyu(x, y, alternative='two-sided')
# res2 = stats.wilcoxon(x, y)
# print(res)
# print(res2)

'''practice2'''
# x = np.array([310, 350, 370, 377, 389, 400, 415, 425,
#               440, 295, 325, 296, 250, 340, 298, 365, 375, 360, 385])
# y = np.array([320]*len(x))

# print(x)
# print(np.mean(x))
# print(y)

# res = stats.mannwhitneyu(x, y, alternative='two-sided') # mannwhitenyu 秩和检验考虑到了不等样本的情况
# res2 = stats.wilcoxon(x, y, correction=True, alternative='greater') # Wilcoxon 秩和检验主要针对两样本量相同的情况
# print(res)
# print(res2)
