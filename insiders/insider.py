from keras.models import load_model
from decimal import Decimal
from pkg_resources import resource_filename, Requirement
from models.denselstm import DenseLSTM
import numpy
from b3data.utils.stock_util import StockUtil
import pandas as pd


class Insider:

    def __init__(self, code, look_back, cells):
        self.model = self._create_model(code, look_back, cells)
        # self.model = load_model(resource_filename(Requirement.parse("insiders"), 'saved_models/' + code + '.h5'))

    def _create_model(self, code, look_back, cells):
        lstm = DenseLSTM(look_back=look_back, lstm_cells=cells, optimizer='rmsprop')
        lstm.model.load_weights('insiders/saved_models/' + code + '.h5') 
        
        return lstm

    def predict(self, dataset):
        dataset = numpy.array(dataset)
        dataset = dataset.reshape(-1, 1)
        self.model.create_data_for_fit(dataset)
        predictions = self.model.model.predict(numpy.append(self.model.train_x, self.model.test_x, 0))
        prediction_labels = [1 if Decimal(i.item()) >= Decimal(0.50) else 0 for i in predictions]
        score = 0
        for i in range(len(prediction_labels)):
            if prediction_labels[i] == numpy.append(self.model.train_y, self.model.test_y)[i]:
                score += 1
        
        print('[Insider] Accuracy: ' + str(score/len(prediction_labels)*100))
        return prediction_labels

    @staticmethod
    def stock_and_predicts(codes=['PETR3'], windows=[6], start_year=2014, end_year=2014,
                          start_month=1,
                          period=6):

        s = StockUtil(codes, windows)
        prices, _ = s.prices_preds(2014, 2014)
        preds = []
        for index, code in enumerate(codes):
            ins = Insider(code, windows[index], 50)
            preds.append(ins.predict(prices[index]))
            prices[index] = prices[index][windows[index]:]
        
        return prices, preds

prices, preds = Insider.stock_and_predicts(['PETR3', 'VALE3', 'ABEV3'], [6, 6, 9], start_month=7)


df = pd.DataFrame({'prices': prices[1], 'preds': preds[1]})
df.to_csv('VALE3.csv')

