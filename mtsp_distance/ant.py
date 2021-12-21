#蚁群算法解决TSP
from globals import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import random


class AntColony:
    def __init__(self):
        self.path_best = []
        self.distance_best = []

    def iterate(self, max_iter=20):
        MAX_iter = max_iter  # 最大迭代值
        AntCount = 100  # 蚂蚁数量
        # 信息素
        alpha = 1  # 信息素重要程度因子
        beta = 2  # 启发函数重要程度因子
        rho = 0.1  # 挥发速度

        Q = 1
        # 城市列表
        citys = []
        # 去掉第一个城市
        for i in range(1, len(X)):
            citys.append([X[i], Y[i]])
        city_count = len(citys)

        # 预计算距离
        # Distance = np.zeros((city_count, city_count))
        # for i in range(city_count):
        #     for j in range(city_count):
        #         if i != j:
        #             Distance[i][j] = math.sqrt((citys[i][0] - citys[j][0]) ** 2 + (citys[i][1] - citys[j][1]) ** 2)
        #         else:
        #             Distance[i][j] = 100000
        Distance = dist_map[1:,1:]

        # 初始信息素矩阵，全是为1组成的矩阵
        pheromonetable = np.ones((city_count, city_count))

        # 候选集列表,存放所有蚂蚁的路径
        candidate = np.zeros((AntCount, city_count)).astype(int)

        # path_best存放的是相应的，每次迭代后的最优路径，每次迭代只有一个值
        path_best = np.zeros((MAX_iter, city_count))

        # 存放每次迭代的最优距离
        distance_best = np.zeros( MAX_iter)

        # 倒数矩阵
        etable = 1.0 / Distance
        iter = 0  # 迭代初始值

        while iter < MAX_iter:
            # first：蚂蚁初始点选择
            # print(iter)
            if AntCount <= city_count:
                candidate[:, 0] = np.random.permutation(range(city_count))[:AntCount]
            else:
                m = AntCount - city_count
                n = 2
                candidate[:city_count, 0] = np.random.permutation(range(city_count))[:]
                while m > city_count:
                    candidate[city_count*(n - 1):city_count*n, 0] = np.random.permutation(range(city_count))[:]
                    m = m - city_count
                    n = n + 1
                candidate[city_count*(n-1):AntCount, 0] = np.random.permutation(range(city_count))[:m]
            length = np.zeros(AntCount)  # 每次迭代的N个蚂蚁的距离值

            # second：选择下一个城市选择
            for i in range(AntCount):
                # 移除已经访问的第一个元素
                unvisit = list(range(city_count))  # 列表形式存储没有访问的城市编号
                visit = candidate[i, 0]  # 当前所在点,第i个蚂蚁在第一个城市
                unvisit.remove(visit)  # 在未访问的城市中移除当前开始的点
                for j in range(1, city_count):  # 访问剩下的city_count个城市，city_count次访问
                    protrans = np.zeros(len(unvisit))  # 每次循环都更改当前没有访问的城市的转移概率矩阵1*30,1*29,1*28...
                    # 下一城市的概率函数
                    for k in range(len(unvisit)):
                        # 计算当前城市到剩余城市的（信息素浓度^alpha）*（城市适应度的倒数）^beta
                        # etable[visit][unvisit[k]],(alpha+1)是倒数分之一，pheromonetable[visit][unvisit[k]]是从本城市到k城市的信息素
                        protrans[k] = np.power(pheromonetable[visit][unvisit[k]], alpha) * np.power(
                            etable[visit][unvisit[k]], (alpha + 1))

                    cumsumprobtrans = (protrans / sum(protrans)).cumsum()
                    cumsumprobtrans -= np.random.rand()
                    # 求出离随机数产生最近的索引值
                    k = unvisit[list(cumsumprobtrans > 0).index(True)]
                    # 下一个访问城市的索引值
                    candidate[i, j] = k
                    unvisit.remove(k)
                    length[i] += Distance[visit][k]
                    visit = k  # 更改出发点，继续选择下一个到达点
                length[i] += Distance[visit][candidate[i, 0]]  # 最后一个城市和第一个城市的距离值也要加进去

            # 如果迭代次数为一次，那么无条件让初始值代替path_best,distance_best.
            if iter == 0:
                distance_best[iter] = length.min()
                path_best[iter] = candidate[length.argmin()].copy()
            else:
                # 如果当前的解没有之前的解好，那么当前最优还是为之前的那个值；并且用前一个路径替换为当前的最优路径
                if length.min() > distance_best[iter - 1]:
                    distance_best[iter] = distance_best[iter - 1]
                    path_best[iter] = path_best[iter - 1].copy()
                else:  # 当前解比之前的要好，替换当前解和路径
                    distance_best[iter] = length.min()
                    path_best[iter] = candidate[length.argmin()].copy()

            # 信息素的增加量矩阵
            changepheromonetable = np.zeros((city_count, city_count))
            for i in range(AntCount):
                for j in range(city_count - 1):
                    changepheromonetable[candidate[i, j]][candidate[i][j + 1]] += Q / length[i]
                # 最后一个城市和第一个城市的信息素增加量
                changepheromonetable[candidate[i, j + 1]][candidate[i, 0]] += Q / length[i]
            # 信息素更新
            pheromonetable = (1 - rho) * pheromonetable + changepheromonetable
            iter += 1
        self.path_best = path_best
        self.distance_best = distance_best


if __name__ == "__main__":
    ant_colony = AntColony()
    max_iter = 2
    ant_colony.iterate(max_iter)
    path_best = ant_colony.path_best
    distance_best = ant_colony.distance_best
    print("蚁群算法的最优路径", path_best[-1]+1)
    print("迭代", max_iter, "次后", "蚁群算法求得最优解", distance_best[-1])

    # 路线图绘制
    fig = plt.figure()
    plt.title("Best roadmap")
    x = []
    y = []
    path = []
    for i in range(len(path_best[-1])):
        x.append(X[int(path_best[-1][i])])
        y.append(Y[int(path_best[-1][i])])
        path.append(int(path_best[-1][i]+1))
    x.append(x[0])
    y.append(y[0])
    path.append(path[0])
    for i in range(len(x)):
        plt.annotate(path[i], xy=(x[i], y[i]), xytext=(x[i] + 0.3, y[i] + 0.3))
    plt.plot(x, y, '-o')

    # 距离迭代图
    fig = plt.figure()
    plt.title("Distance iteration graph")  # 距离迭代图
    plt.plot(range(1, len(distance_best) + 1), distance_best)
    plt.xlabel("Number of iterations")  # 迭代次数
    plt.ylabel("Distance value")  # 距离值
    plt.show()
