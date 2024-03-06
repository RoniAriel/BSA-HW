
import Functions
import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd
file_path1 = "/Users/Roni_Ariel/PycharmProjects/pythonProject53/rawSpikes1.mat"
file_path2= "/Users/Roni_Ariel/PycharmProjects/pythonProject53/rawSpikes2.mat"
r = 94 #spikes per second
Fs = 1e3 #Sampling frequency
totalTime = 30 #seconds
dt = 0.001 #seconds
binSize = 0.03 #seconds
rawSpikes1_train = Functions.readPoiSpikes(file_path1, Fs)
rawSpikes2_train = Functions.readPoiSpikes(file_path2, Fs)
generated_spike_train = Functions.generatePoiSpikes(r, dt, totalTime)
cv_rawSpikes1_train= Functions.calcCV(rawSpikes1_train)
cv_rawSpikes2_train= Functions.calcCV(rawSpikes2_train)
cv_generated_spike_train= Functions.calcCV(generated_spike_train)
FF_rawSpikes1_train= Functions.calcFF(rawSpikes1_train)
FF_rawSpikes2_train= Functions.calcFF(rawSpikes2_train)
FF_generated_spike_train= Functions.calcFF(generated_spike_train)
rawSpikes1_Rate= Functions.calcRate(rawSpikes1_train, 1000, dt)
rawSpikes2_Rate = Functions.calcRate(rawSpikes2_train, 0, dt)
generated_spike_Rate = Functions.calcRate(generated_spike_train, 0, dt)


plt.figure(figsize=(10, 6))
time_points = np.arange(len(rawSpikes2_Rate)) * 1000 * dt
plt.plot(time_points, rawSpikes2_Rate, label='Generated Spike Train', color='g')
plt.xlabel('Time (s)')
plt.ylabel('Rate of Fire (spikes/s)')
plt.title('Rate of Fire Over Time')
plt.grid(True)
plt.legend()
plt.show()


  # Save to excel for pdf output
# rawSpikes1_train = rawSpikes1_train[:10000]  # slice the binned array due to space problem
# matrix_spike_train =np.reshape(rawSpikes1_train, (100, 100)) #convert it to matrix  due to space problem
# rawSpikes2_train = rawSpikes2_train[:10000]
# matrix_spike_train2 = np.reshape(rawSpikes2_train, (100, 100))
# generated_spike_train = generated_spike_train[:10000]
# matrix_spike_train3 = np.reshape(generated_spike_train, (100, 100))
# df1 = pd.DataFrame(matrix_spike_train)
# df2 = pd.DataFrame(matrix_spike_train2)
# df3 = pd.DataFrame(matrix_spike_train3)
# # # Create a Pandas Excel writer using XlsxWriter as the engine
# excel_writer = pd.ExcelWriter("output1.xlsx", engine='xlsxwriter')
#
# # # Write each DataFrame to a separate sheet in the Excel file
# df1.to_excel(excel_writer, sheet_name='Sheet1', index=False)
# df2.to_excel(excel_writer, sheet_name='Sheet2', index=False)
# df3.to_excel(excel_writer,sheet_name='Sheet3', index=False)
# excel_writer._save()