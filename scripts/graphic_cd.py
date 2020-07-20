import matplotlib.pyplot as plt
import Orange, orngStat
import pandas
import numpy

code = 'PETR3'

df = pandas.read_csv(f'../results/posthoc_{code}.csv', index_col=0)

initial_rank = 1
names = ['cells_1', 'cells_50', 'cells_80', 'cells_100', 'cells_150', 'cells_200']
table_ranks = []

for index, row in df.iterrows():
    initial_rank = 1
    rank_row_dict = {}
    sorted = row.sort_values(ascending=False)
    for i in sorted.index.tolist():
        rank_row_dict.update({i: initial_rank})
        initial_rank += 1

    rank_row_list = []
    for name in names:
        rank_row_list.append(rank_row_dict[name])

    table_ranks.append(rank_row_list)

print(table_ranks)
avranks = numpy.mean(table_ranks, axis=0)
print(avranks)

names = ['1 Unidade', '50 Unidades', '80 Unidades', '100 Unidades', '150 Unidades', '200 Unidades']
cd = orngStat.compute_CD(avranks, 10) #tested on 30 datasets
orngStat.graph_ranks(f'../results/final_posthoc_{code}.eps', avranks, names, cd=cd, width=6, textspace=1.5)
# plt.show()
# plt.savefig('../graphics/cd_digram.png')
