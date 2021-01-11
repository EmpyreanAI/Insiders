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
from numpy.random import seed
import tensorflow

tensorflow.random.set_seed(33)
seed(9)

code = 'VALE3'
look_back = 6
lstm_cells = 50
optimizer = 'rmsprop'
batch_size = 2

stocks = Stocks(year=2014, cod=code, period=6)
dataset = stocks.selected_fields([CLOSING])
dataset = duplicate_data(dataset)

stocks_y = Stocks(year=2014, start_month=7, period=6)
dataset_y = stocks_y.selected_fields([CLOSING])
print(dataset_y)

model = DenseLSTM(input_shape=dataset.shape[1],
                  look_back=look_back, lstm_cells=lstm_cells, optimizer=optimizer)
model.create_data_for_fit(dataset_y)
X = np.append(model.train_x, model.test_x, 0)
model.create_data_for_fit(dataset)

result = model.fit_and_evaluate(batch_size=2, epochs=5000)

result['model'].save(code+'.h5')

average_dataset = [dataset_y[i-look_back:i].mean() for i in range(look_back, len(dataset_y))]
prediction = model.model.predict(X)
prediction_labels = [1 if Decimal(i.item()) >= Decimal(0.50) else 0 for i in prediction]

# begin_test = int(len(dataset_y) - len(model.test_x))
begin_test = 0
correct_labels = [1 if dataset_y[i] >= dataset_y[i-look_back:i].mean() else 0 for i in range(look_back, len(dataset_y))]

print(correct_labels)

plt.figure(figsize=(10, 3))

# x = list(range(len(model.test_x)))
x = list(range(len(X)))
y = average_dataset[begin_test:len(average_dataset)]
y_d = dataset_y[look_back:len(dataset_y)]
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

a_label, = plt.plot(y, color='#70a2ff')
d_label, = plt.plot(y_d, '-o',color='#ff756b', markersize=3)

extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
# plt.legend((extra, d_label), ("Precisão: {}%".format(round(score/(len(x)-1), 2)*100), 
#                               "Preço médio últimos " + str(look_back) + " dias"), fontsize=10)

plt.legend((extra, l, h, lx, hx, a_label, d_label), 
           ("Acurácia: {}%".format(round(score/(len(x)-1), 2)*100), 
            "Predição correta de baixa",
            "Predição correta de alta",
            "Predição incorreta de baixa",
            "Predição incorreta de alta",
            "Preço médio dos últimos " + str(look_back) + " dias",
            "Preço real do ativo"), fontsize=10, bbox_to_anchor=(1.05, 1))

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
