import sys
sys.path.append('../')

from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np
from insiders.models.denselstm import DenseLSTM
from b3data.stocks import Stocks, CLOSING, OPENING, MAX_PRICE
from b3data.stocks import MIN_PRICE, MEAN_PRICE, VOLUME
from b3data.utils.smote import duplicate_data
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines

code = 'VALE3'
look_back = 6
lstm_cells = 50
optimizer = 'rmsprop'
batch_size = 2

stocks = Stocks(year=2014, cod=code, period=6)
dataset = stocks.selected_fields([CLOSING])
dataset = duplicate_data(dataset)

model = DenseLSTM(input_shape=dataset.shape[1],
                  look_back=look_back, lstm_cells=lstm_cells, optimizer=optimizer)
model.create_data_for_fit(dataset)
result = model.fit_and_evaluate(batch_size=2, epochs=5000)

average_dataset = [dataset[i-look_back:i].mean() for i in range(look_back, len(dataset))]
prediction = model.model.predict(model.test_x)
prediction_labels = [1 if Decimal(i.item()) >= Decimal(0.50) else 0 for i in prediction]

begin_test = int(len(dataset) - len(model.test_x))
correct_labels = [1 if dataset[i] >= dataset[i-look_back:i].mean() else 0 for i in range(begin_test, len(dataset))]

plt.figure(figsize=(8, 3))

x = list(range(len(model.test_x)))
y = average_dataset[begin_test-look_back:len(average_dataset)]
y_d = dataset[begin_test:len(dataset)]
plt.ylabel("Preço do Ativo")
plt.xlabel("Dias")
plt.title(code)
score = 0
for i in range(len(x)-1):
    if prediction_labels[i] == 1:
        if prediction_labels[i] == correct_labels[i]:
            h = plt.scatter(x[i], y[i], marker='^', c='green', zorder=10)
            score += 1
        else:
            hx = plt.scatter(x[i], y[i], marker='x', c='green', zorder=10)
    else:
        if prediction_labels[i] == correct_labels[i]:
            l = plt.scatter(x[i], y[i], marker='v', c='red', zorder=10)
            score += 1
        else:
            lx = plt.scatter(x[i], y[i], marker='x', c='red', zorder=10)

    plt.plot([x[i],x[i]], [y_d[i], y[i]], 'k--', linewidth=1)

d_label, = plt.plot(y, color='#70a2ff')
a_label, = plt.plot(y_d, '-o',color='#ff756b', markersize=3)

# extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
# plt.legend((extra, d_label), ("Precisão: {}%".format(round(score/(len(x)-1), 2)*100), 
#                               "Preço médio últimos " + str(look_back) + " dias"), fontsize=10)

plt.legend((lx, hx, l, h), ("Precisão: {}%".format(round(score/(len, 
                            "Preço médio últimos " + str(look_back) + " dias"), fontsize=10)

figure = legend.figure
fig.canvas.draw()
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(filename, dpi="figure", bbox_inches=bbox)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
