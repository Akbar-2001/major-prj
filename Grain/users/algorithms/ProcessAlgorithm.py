from django.conf import settings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
class Algorithms:
    from sklearn import preprocessing
    from sklearn.datasets import make_regression
    #le = preprocessing.LabelEncoder()
    path = settings.MEDIA_ROOT + "\\" + "grain_dataset.csv"

    data = pd.read_csv(path, delimiter=',')
# 100% 2types train and test  train =80% test=20%
    x = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    x,y = make_regression(random_state=0)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train) # x= (2,3)
    x_test = sc.fit_transform(x_test) # x= (1)

    x_train = pd.DataFrame(x_train)
    x_train.head()

    def processSVR(self):
        from sklearn.svm import SVR
        from sklearn.metrics import confusion_matrix

        print('x_train:',self.x_train)
        print('y_train:', self.y_train)

        regressor = SVR()
        regressor.fit(self.x_train,self.y_train)
        print('x_test:',self.x_test)
        y_pred = regressor.predict(self.x_test)
        print('y_pred:', y_pred)

        
        #score = regressor.score(self.x_test, self.y_test)
        #print(score)

        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        return mae,mse, r2
        
     

    def RandomForestRegressor(self):
        from sklearn.ensemble import RandomForestRegressor
        #from sklearn.datasets import make_regression

        #x,y = make_regression(self.x, self.y, random_state=0, shuffle=False)

        regr = RandomForestRegressor()
        regr.fit(self.x_train, self.y_train)
        y_pred = regr.predict(self.x_test)
        #score = regr.score([self.y_test], y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        return mae,mse, r2

    def GradientBoosting(self):
        from sklearn.ensemble import GradientBoostingRegressor

        reg = GradientBoostingRegressor(random_state=0)
        reg.fit(self.x_train, self.y_train)
        y_pred = reg.predict(self.x_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        #score = reg.score([self.y_test], y_pred)

        return mae,mse, r2

    def GRSVR(self):
        from sklearn.ensemble import RandomForestRegressor
       
        regr = RandomForestRegressor()
        regr.fit(self.x_train, self.y_train)
        y_pred = regr.predict(self.x_test)
        mae1 = mean_absolute_error(self.y_test, y_pred)
        mse1 = mean_squared_error(self.y_test, y_pred)

        from sklearn.ensemble import GradientBoostingRegressor

        reg = GradientBoostingRegressor(random_state=0)
        reg.fit(self.x_train, self.y_train)
        y_pred = reg.predict(self.x_test)

        mae2 = mean_absolute_error(self.y_test, y_pred)
        mse2 = mean_squared_error(self.y_test, y_pred)

        mae = np.array([[mae1],[mae2]])
        mae = np.mean(mae)
        mse = np.array([[mse1],[mse2]])
        mse = np.mean(mse)

        return mae, mse



