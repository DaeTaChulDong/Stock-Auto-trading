import pandas as pd
from pykrx import stock
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

#삼성전자 주식 데이터 불러오기
start_date='2024-01-01'
end_date='2025-08-28'
stock_code='005930'
df=stock.get_market_ohlcv_by_date(fromdate=start_date,todate=end_date,ticker=stock_code)

#다음날 고가 예측을 위해 하루씩 미룬 고가 컬럼 생성
df['Next_High']=df['고가'].shift(-1)

#마지막 행은 다음날 데이터가 없으므로 제거
df=df[:-1]

#특성과 타깃 분리
X=df.drop('Next_High', axis=1)
y=df['Next_High']

#훈련 세트와 테스트 세트 분리
X_train=X[:'2025-08-01']
y_train=y[:'2025-08-01']
X_test=X['2025-08-02':]
y_test=y['2025-08-02':]

#모델 학습
model=RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#예측
predictions=model.predict(X_test)

#실제 데이터와 예측 데이터 시각화
plt.figure(figsize=(14,7))
plt.plot(df[:'2025-08-01'], df['고가'][:'2025-08-02'], label='Actual High Price', color='blue')
plt.plot(df['2025-08-02':], y_test, 'gx-', label='Actual High Price (Test)', color='green')
plt.plot(df['2025-08-02':], predictions, 'rx-', label='Predicted High Price', color='red')
plt.xlabel('Date')
plt.ylabel('High Price')
plt.title('Samsung Electronics Stock Price Prediction')
plt.legend()
plt.show()