import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import mean_absolute_error
import csv

def MBE(y_true, y_pred):
    '''
    Parameters:
        y_true (array): Array of observed values
        y_pred (array): Array of prediction values

    Returns:
        mbe (float): Bias score
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_true = y_true.reshape(len(y_true),1)
    
    y_pred = y_pred.reshape(-1, 1)   
    diff = (y_true-y_pred)
    mbe = diff.mean()
    print('MBE = ', mbe)
    
    
# Load data
train = pd.read_csv(r"Training Set.csv")
test = pd.read_csv(r"Test Set.csv")

# Normalize data
scalerx = MinMaxScaler()
scalery = MinMaxScaler()
time_min = test['Minute']
actual_res = test['Actual Temperature Middle ((T[n])']
actual_res = np.array(actual_res)

train_x = scalerx.fit_transform(train[['Solar Radiation (W/m^2)', 'Outside Temperature (T[n-1])','Inside Temperature Middle (T[n-1])', 'Fan on/off']])
train_y = scalery.fit_transform(train[['Actual Temperature Middle ((T[n])']])
train_y = train_y.flatten()

test_x = scalerx.transform(test[['Solar Radiation (W/m^2)','Outside Temperature (T[n-1])', 'Inside Temperature Middle (T[n-1])', 'Fan on/off']])
test_y = scalery.transform(test[['Actual Temperature Middle ((T[n])']])
test_y = test_y.flatten()

# reshape input to be [samples, time steps, features]
X_train = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))
X_test = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))
    
# Optimize variables baseed on grid cross validation
      
svr_regr = SVR(kernel="rbf", epsilon=0.02894736842105263, C=7.63157894736842, gamma=1.3526315789473684)    # 1 day simulation with time (BEST)

svr_regr.fit(X_train, train_y)

# 5-minute ahead predictions (not simulated)

test_predict = svr_regr.predict(X_test)
test_predict = scalery.inverse_transform(np.array(test_predict).reshape(-1,1))
predict_score = svr_regr.score(X_test, test_y[0:len(test_y)])
svr_r2_sim = round(r2_score(scalery.inverse_transform(test_y.reshape(-1,1)), test_predict),2)
svr_mse_sim = round(math.sqrt(mean_squared_error(scalery.inverse_transform(test_y.reshape(-1,1)), test_predict)),2)
predict_mae = mean_absolute_error(scalery.inverse_transform(test_y.reshape(-1,1)), test_predict)
print("MAE=" + str(predict_mae))
MBE(scalery.inverse_transform(test_y.reshape(-1,1)), test_predict)
print("R2=" + str(svr_r2_sim))
print("RMSE: " + str(svr_mse_sim))

# initialise variables for simulation
prev_fan = 0
inside_temp_sim = []
x_test = X_test
input_sim = X_test[0,:]      
r2_list = []
param_list= []
inside_temp_sim.append(scalery.inverse_transform(test_y[0].reshape(1,-1)))
prev_delta_temp = 0
prev_sol = 0
prev_fan = 0
prev_inside_temp = 0
prev_outside_temp = 0
prev_time = 0
datapoint_no = 288
r2_hour_sim = 0
rmse_hour_sim = 0
pred_vec = []
count_col = 0
count_row = 0
j = 0
temp_pred = []
plot_array = []
fan_pred = []
error_array = np.zeros((len(time_min)-12,12))
temp_counter = 0

# Simulate until one hour before the end of test set is reached
while len(time_min)-j > 12:
    pred_vec = []
    input_sim = X_test[j,:]
    temp_counter += 1
    for i in range(12):
        
        temp_fan = 0
        input_sim = input_sim.reshape(1,-1)
        predicted_scaled_temperature = svr_regr.predict(input_sim)   
        predicted_temperature = scalery.inverse_transform(svr_regr.predict(input_sim).reshape(1,-1))
        pred_vec.append(scalery.inverse_transform(svr_regr.predict(input_sim).reshape(1,-1)))
        temp_pred.append(scalery.inverse_transform(svr_regr.predict(input_sim).reshape(1,-1)))       
        error_array[j][i] = round(math.sqrt((actual_res[j+i] - pred_vec[i])**2),2)
        
        prev_fan = input_sim[0,3]
        
        if prev_fan == 1 and predicted_temperature > 22:
            temp_fan = 1
        elif prev_fan == 1 and predicted_temperature <= 22:
            temp_fan = 0

        if prev_fan == 0 and predicted_temperature < 30:
            temp_fan = 0
        elif prev_fan == 0 and predicted_temperature >= 30:
            temp_fan = 1
        prev_fan = temp_fan      
        input_sim = np.array([X_test[j+i,0], X_test[j+i,1], predicted_scaled_temperature[0], prev_fan], dtype='object').reshape(1,-1)     
        count_col += 1
    plot_array.append(pred_vec[-1])
    fan_pred.append(prev_fan*5)

    j += 1

output_array = np.array(plot_array)
output_array = output_array.reshape(-1,1)
np.savetxt('simulated.csv', output_array, delimiter=',')
output_fan = np.array(fan_pred)
output_fan = output_fan.reshape(-1,1)
np.savetxt('simulated_fan.csv', output_fan, delimiter=',')
np.savetxt('simulated_error.csv', error_array, delimiter=',')

# Output to files and plotting

x_labels = np.arange(0,datapoint_no,1)
inside_temp_sim = np.array(inside_temp_sim).flatten()
predictions_svr = inside_temp_sim

prev_fan = prev_fan*5 # Makes the fan output appear larger on the plots
plot_array = np.array(plot_array).flatten()
svr_r2_sim = round(r2_score(actual_res[0:len(time_min)-12], plot_array),2)

svr_mse_sim = round(math.sqrt(mean_squared_error(actual_res[:-12], plot_array)),2)
predict_mae = mean_absolute_error(actual_res[:-12], plot_array)
print("MAE=" + str(predict_mae))
MBE(actual_res[:-12], plot_array)
print("R2=" + str(svr_r2_sim))
print("RMSE: " + str(svr_mse_sim))


x_labels = np.arange(0,len(time_min),1)

plt.plot(x_labels[0:len(time_min)-12],actual_res[0:len(time_min)-12], color='blue', alpha = 0.8)
plt.plot(x_labels[0:len(time_min)-12],plot_array, color='red', alpha=0.7)
plt.plot(x_labels[0:len(time_min)-12], X_test[:len(time_min)-12,3]*5, color='black', alpha=0.8)
plt.plot(x_labels[0:len(time_min)-12], fan_pred, color='green')

with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Actual Temperature', 'Simulated Temperature', 'Actual Fan/Wet Wall State', 'Simulated Fan/Wet Wall State', 'Solar Radiation (W/m^2)', 'Outside Temperature'])
    for i in range(len(fan_pred)):
        temp_array = scalerx.inverse_transform(X_test[i,:].reshape(1,-1))
        temp_array = temp_array.reshape(-1,1)
        temp_array = temp_array.flatten()
        writer.writerow([actual_res[i], plot_array[i],  X_test[i, 3], fan_pred[i], temp_array[0], temp_array[1]])

plt.xlabel("Time (5-minute Intervals)", fontsize=15)
plt.ylabel("Temperature (deg. C)", fontsize=15)
plt.ylim(0, 37)
plt.legend(['Actual Temperature', 'Time Ahead Prediction', 'Actual Fan State', 'Simulated Fan State'], loc='lower left')
plt.xticks(rotation=90)
plt.grid(False)                  
plt.show()


# Error plots based on the ahead interval used (i.e. 5-min, 10-min, 30-min etc.)
fig, ax = plt.subplots()
bp = ax.boxplot(error_array, showfliers=False, medianprops={'color': 'red'})
ax.set_xticklabels(['5 min', '10 min', '15 min', '20 min', '25 min', '30 min', '35 min', '40 min', '45 min', '50 min', '55 min', '60 min'])
ax.set_ylabel('Temperature')
ax.set_xlabel('Time Step Ahead')
ax.set_title('5-min Ahead Predictions')
with open('error_boxplot.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['5 min', '10 min', '15 min', '20 min', '25 min', '30 min', '35 min', '40 min', '45 min', '50 min', '55 min', '60 min'])
    for i in range(error_array.shape[0]):
        writer.writerow([error_array[i,0], error_array[i,1], error_array[i,2], error_array[i,3], error_array[i,4], error_array[i,5], error_array[i,6], error_array[i,7], error_array[i,8], error_array[i,9], error_array[i,10], error_array[i,11]])


plt.show()
