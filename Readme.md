
# BADMINTON STROKE CLASSIFICATION
### AN END TO END MACHINE LEARNING PROJECT

This project is on classifying the type of badminton stroke player played using the data collected from a small wrist device. This device gives us accelorometer and gyroscope data and is attached to the player's wrist. This project was part of Data Analytics course we did at IISc. And it involved everything from data collection, cleaning up till the final ML model evaluation.

This jupyter notebook explains "what we did?" and "how we did it?" (the code).

## Defining the problem

To start off, we first need to define and properly formulate the problem we want to solve. Since, this was a course project we could have chosen any problem. We finalized to work on classifying the kind of stroke a player is playing based on the sensor's data we get from her smartwatch. This was an extension of Human Activity Recognition (HAR) project, which using similar data, classifies the activities as running, walking, etc. In our case,  we decided to have the following badminton strokes as classes:
* Backhand Overarm (bo)
* Backhand Underarm (bu)
* Forehand Overarm (fo)
* Forehand Underarm (fu)
* Forehand Smash (fs)

The 2 letter in parenthesis are abbreviations of the strokes, which we'll be using throughout this notebook.

## Data
As per our knowledge, there was no publicly available data for this problem. There have been some research papers, but we couldn't get our hands on the data. So we decided to collect it ourselves (Why not? Can give it a try).

### Setup for collecting data
We got hold on a device which can give us accelorometer and gyroscope readings over a period of time at a descent frequency. The device was powered by a small cell and connected via bluetooth low energy (BLE) to a rasperry pi. So we collected our data on rasperry pi in csv formats.

### Collecting data
Thanks to our 5 volunteers, we collected the data for the 5 mentioned strokes by making them repeatedly play the same shot. Although this might be very different than a real game, but we tried to make them play as they are playing a real game, (i.e. starting from center position, going towards the shuttle cock and returning back to resting position). The device had it's own limitation (coverage, power, etc) which made it challenging. Going through all of this we finally got our data.

## Visualizing the data


```python
# Some imports
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks  #For finding peaks (we'll see later)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pickle
```

The data is saved in the data folder. We have 25 different csv files, for 5 players, 5 classes each. The naming convention is playerid_strokename. e.g. p2_bu.csv file contains data of player id 2 for "backhand underarm".

The columns in each file are: timestamp, accelorometer x-axis (ax), y-axis (ay), z-axis (az), gyroscope x-axis (gx), y-axis (gy), z-axis (gz). We also added magnitudes of accelerometer and gyroscope.


```python
# Columns in data files
cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']  # Accelerometer and gyroscope in 3 axis
addedCols = ['acc_mag', 'gyro_mag']          # Magnitude of acc and gyro
```


```python
# Data reading
data = {}
persons = ["p1", "p2", "p3", "p4", "p5"]
shots = ["bo", "bu", "fo", "fs", "fu"]
for person in persons:
    for shot in shots:
        data[person + "_" + shot] = pd.read_csv("data/" + str(person) + "_" + str(shot) + ".csv")
        data[person + "_" + shot]['acc_mag'] = np.linalg.norm(data[person + "_" + shot][cols[:3]].values,axis=1)
        data[person + "_" + shot]['gyro_mag'] = np.linalg.norm(data[person + "_" + shot][cols[3:]].values,axis=1)
```

Let's plot one of the csv, say p5_bo.


```python
plotName = "p5_bo"

# Set figure width as per the number of readings
plt.figure(figsize=(len(data[plotName])/10, 8), dpi=150)

# Plot for each accelerometer column
for col in cols[0:3]:
    plt.plot(data[plotName][col], label=col)

plt.xlim(xmin=0)
plt.xticks(np.arange(0, len(data[plotName])+1, 5.0))
plt.title(plotName + "_acc")
plt.legend()
plt.show()
```


![png](output_9_0.png)


## Data cleaning

The peaks in the above plot are the shots, and in between we have some rest times. The shots are not at uniform intervals, sometimes the rest period is very long (as in between 160 and 200 in above plot). So we need to extract out shots from this continous plot and remove all these players' resting periods. 

The approach we took was to take a window around the peaks in the signal. Many design decisions were to be made, like, should the window by dynamic or fixed? If fixed, window size? peaks in which signal, accelorometer or gyroscope, x or y or z? one signal or many signals? how to consider some spike as peak? any thresholding? any noise reduction algorithm?

We did some EDA by plotting all the data and chose a window size of 13. This was our choice by our observations. And we chose the ay-signal to consider for extracting peaks and window, with different threshold values for each class of stroke.


```python
# Size of the window, it is common for all shots
windowSize = 13
# shots = ["bo", "bu", "fo", "fs", "fu"] for reference
# Name of the sensor to be used for thresholding and the threshold value, the order is as per the shots list order
sensorToThreshold = ['ay', 'ay', 'ay', 'ay', 'ay']
threshold = [1.25, 1.25, 1.5, 1.5, 1.5]

```


```python
# DATA PLOTTING - This will save all the plots in plots folder
for p_name in persons:
    for s_name in shots:
        s_index = shots.index(s_name)  # Shot index of that shot
        plotName = p_name + "_" + s_name
        
        # Set figure width as per the number of readings
        plt.figure(figsize=(len(data[plotName])/10, 5), dpi=100)
        
        # Plot peaks with threashold of that shot and give the name of sensor used for thresholding
        peaks, _ = find_peaks(data[plotName][sensorToThreshold[s_index]], height=threshold[s_index])
        #plt.plot(peaks, data[plotName][sensorToThreshold[s_index]][peaks], "x", c='red')
        
        # Plot for each accelerometer column
        for col in cols[0:3]:
            plt.plot(data[plotName][col], label=col)

        # drawing windows around each peak
        for peak in peaks:
            plt.axvspan(int(peak - windowSize/2), int(peak + windowSize/2), alpha =.1, facecolor='g',edgecolor='black')
        
        plt.xlim(xmin=0)
        plt.xticks(np.arange(0, len(data[plotName])+1, 5.0))
        plt.title(plotName + "_acc")
        plt.legend()
        plt.savefig("plots/" + plotName + "_acc")
        #plt.show()
        plt.close();
        
        # Plot for each gyrometer column (not drawing peaks in these)
        
        # Set figure width as per the number of readings
        plt.figure(figsize=(len(data[plotName])/10, 5), dpi=100)
        
        for col in cols[3:]:
            plt.plot(data[plotName][col], label=col)

        # drawing windows around each peak
        for peak in peaks:
            plt.axvspan(int(peak - windowSize/2), int(peak + windowSize/2), alpha =.1, facecolor='g',edgecolor='black')
        
        plt.xlim(xmin=0)
        plt.xticks(np.arange(0, len(data[plotName])+1, 5.0))
        plt.title(plotName + "_gyro")
        plt.legend()
        plt.savefig("plots/" + plotName + "_gyro")
        #plt.show()
        plt.close();
                  
        # Magnitude - Acc
        # Set figure width as per the number of readings
        plt.figure(figsize=(len(data[plotName])/10, 5), dpi=100)

        plt.plot(data[plotName][addedCols[0]], label="Magnitude Accelorometer")

        # drawing windows around each peak
        for peak in peaks:
            plt.axvspan(int(peak - windowSize/2), int(peak + windowSize/2), alpha =.1, facecolor='g',edgecolor='black')
        
        plt.xlim(xmin=0)
        plt.xticks(np.arange(0, len(data[plotName])+1, 5.0))
        plt.title(plotName + "_mag_acc")
        plt.legend()
        plt.savefig("plots/" + plotName + "_mag_acc")
        #plt.show()
        plt.close();
        
        # Magnitude - Gyro
        # Set figure width as per the number of readings
        plt.figure(figsize=(len(data[plotName])/10, 5), dpi=100)

        plt.plot(data[plotName][addedCols[1]], label="Magnitude Gyroscope")

        # drawing windows around each peak
        for peak in peaks:
            plt.axvspan(int(peak - windowSize/2), int(peak + windowSize/2), alpha =.1, facecolor='g',edgecolor='black')
        
        plt.xlim(xmin=0)
        plt.xticks(np.arange(0, len(data[plotName])+1, 5.0))
        plt.title(plotName + "_mag_gyro")
        plt.legend()
        plt.savefig("plots/" + plotName + "_mag_gyro")
        #plt.show()
        plt.close();
        
```

Now this is what our extracted shots look like. 

Accelorometer:
<img src="plots/p5_bu_acc.png">

Gyroscope:
<img src="plots/p5_bu_gyro.png">

There are some overlaps, either because of shots being very close to each other or because of double peaks showing up in some shots. And some shots are missed because of low threshold value. But this is what we found to give reasonable balance between extracting good shots and not the noisy ones.

We'll save the start and end frame for each of these extracted shots in an X_y dataframe, along with the true labels. Later on, we'll augment this dataframe with hand engineered features. We are also saving person id although it's not used in classification.


```python
# Final data frame with features described in doc along with shot name
X_y = pd.DataFrame(columns=['StartFrame', 'EndFrame', 'PersonID', 'ShotName'])

# Creating final data frame and adding end and begin frame of window
for p_name in persons:
    for s_name in shots:
        s_index = shots.index(s_name)  # Shot index of that shot
        plotName = p_name + "_" + s_name
        # Find peaks for window
        #TODO: we can look at peak_features(second returned value) for more data features
        timeSeries = data[plotName][sensorToThreshold[s_index]]
        peaks, _ = find_peaks(timeSeries, height=threshold[s_index])
        for peak in peaks:
            if(peak < windowSize/2 or peak > len(timeSeries)-windowSize/2):
                #print(peak)
                continue
            d = {'StartFrame': int(peak - windowSize/2),
                 'EndFrame': int(peak + windowSize/2), 
                 'PersonID': p_name, 
                 'ShotName': s_name}
            #print(d)
            X_y = X_y.append(d, ignore_index=True)
            
```


```python
X_y
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StartFrame</th>
      <th>EndFrame</th>
      <th>PersonID</th>
      <th>ShotName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>17</td>
      <td>p1</td>
      <td>bo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>20</td>
      <td>p1</td>
      <td>bo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>35</td>
      <td>p1</td>
      <td>bo</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>73</td>
      <td>p1</td>
      <td>bo</td>
    </tr>
    <tr>
      <th>4</th>
      <td>75</td>
      <td>88</td>
      <td>p1</td>
      <td>bo</td>
    </tr>
    <tr>
      <th>5</th>
      <td>77</td>
      <td>90</td>
      <td>p1</td>
      <td>bo</td>
    </tr>
    <tr>
      <th>6</th>
      <td>94</td>
      <td>107</td>
      <td>p1</td>
      <td>bo</td>
    </tr>
    <tr>
      <th>7</th>
      <td>143</td>
      <td>156</td>
      <td>p1</td>
      <td>bo</td>
    </tr>
    <tr>
      <th>8</th>
      <td>162</td>
      <td>175</td>
      <td>p1</td>
      <td>bo</td>
    </tr>
    <tr>
      <th>9</th>
      <td>179</td>
      <td>192</td>
      <td>p1</td>
      <td>bo</td>
    </tr>
    <tr>
      <th>10</th>
      <td>181</td>
      <td>194</td>
      <td>p1</td>
      <td>bo</td>
    </tr>
    <tr>
      <th>11</th>
      <td>198</td>
      <td>211</td>
      <td>p1</td>
      <td>bo</td>
    </tr>
    <tr>
      <th>12</th>
      <td>220</td>
      <td>233</td>
      <td>p1</td>
      <td>bo</td>
    </tr>
    <tr>
      <th>13</th>
      <td>224</td>
      <td>237</td>
      <td>p1</td>
      <td>bo</td>
    </tr>
    <tr>
      <th>14</th>
      <td>6</td>
      <td>19</td>
      <td>p1</td>
      <td>bu</td>
    </tr>
    <tr>
      <th>15</th>
      <td>27</td>
      <td>40</td>
      <td>p1</td>
      <td>bu</td>
    </tr>
    <tr>
      <th>16</th>
      <td>41</td>
      <td>54</td>
      <td>p1</td>
      <td>bu</td>
    </tr>
    <tr>
      <th>17</th>
      <td>94</td>
      <td>107</td>
      <td>p1</td>
      <td>bu</td>
    </tr>
    <tr>
      <th>18</th>
      <td>109</td>
      <td>122</td>
      <td>p1</td>
      <td>bu</td>
    </tr>
    <tr>
      <th>19</th>
      <td>127</td>
      <td>140</td>
      <td>p1</td>
      <td>bu</td>
    </tr>
    <tr>
      <th>20</th>
      <td>143</td>
      <td>156</td>
      <td>p1</td>
      <td>bu</td>
    </tr>
    <tr>
      <th>21</th>
      <td>158</td>
      <td>171</td>
      <td>p1</td>
      <td>bu</td>
    </tr>
    <tr>
      <th>22</th>
      <td>175</td>
      <td>188</td>
      <td>p1</td>
      <td>bu</td>
    </tr>
    <tr>
      <th>23</th>
      <td>193</td>
      <td>206</td>
      <td>p1</td>
      <td>bu</td>
    </tr>
    <tr>
      <th>24</th>
      <td>215</td>
      <td>228</td>
      <td>p1</td>
      <td>bu</td>
    </tr>
    <tr>
      <th>25</th>
      <td>232</td>
      <td>245</td>
      <td>p1</td>
      <td>bu</td>
    </tr>
    <tr>
      <th>26</th>
      <td>257</td>
      <td>270</td>
      <td>p1</td>
      <td>bu</td>
    </tr>
    <tr>
      <th>27</th>
      <td>259</td>
      <td>272</td>
      <td>p1</td>
      <td>bu</td>
    </tr>
    <tr>
      <th>28</th>
      <td>272</td>
      <td>285</td>
      <td>p1</td>
      <td>bu</td>
    </tr>
    <tr>
      <th>29</th>
      <td>277</td>
      <td>290</td>
      <td>p1</td>
      <td>bu</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>678</th>
      <td>114</td>
      <td>127</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>679</th>
      <td>125</td>
      <td>138</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>680</th>
      <td>141</td>
      <td>154</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>681</th>
      <td>157</td>
      <td>170</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>682</th>
      <td>168</td>
      <td>181</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>683</th>
      <td>190</td>
      <td>203</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>684</th>
      <td>218</td>
      <td>231</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>685</th>
      <td>237</td>
      <td>250</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>686</th>
      <td>249</td>
      <td>262</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>687</th>
      <td>277</td>
      <td>290</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>688</th>
      <td>294</td>
      <td>307</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>689</th>
      <td>313</td>
      <td>326</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>690</th>
      <td>321</td>
      <td>334</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>691</th>
      <td>340</td>
      <td>353</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>692</th>
      <td>372</td>
      <td>385</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>693</th>
      <td>382</td>
      <td>395</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>694</th>
      <td>398</td>
      <td>411</td>
      <td>p5</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>695</th>
      <td>3</td>
      <td>16</td>
      <td>p5</td>
      <td>fu</td>
    </tr>
    <tr>
      <th>696</th>
      <td>18</td>
      <td>31</td>
      <td>p5</td>
      <td>fu</td>
    </tr>
    <tr>
      <th>697</th>
      <td>31</td>
      <td>44</td>
      <td>p5</td>
      <td>fu</td>
    </tr>
    <tr>
      <th>698</th>
      <td>45</td>
      <td>58</td>
      <td>p5</td>
      <td>fu</td>
    </tr>
    <tr>
      <th>699</th>
      <td>69</td>
      <td>82</td>
      <td>p5</td>
      <td>fu</td>
    </tr>
    <tr>
      <th>700</th>
      <td>81</td>
      <td>94</td>
      <td>p5</td>
      <td>fu</td>
    </tr>
    <tr>
      <th>701</th>
      <td>94</td>
      <td>107</td>
      <td>p5</td>
      <td>fu</td>
    </tr>
    <tr>
      <th>702</th>
      <td>119</td>
      <td>132</td>
      <td>p5</td>
      <td>fu</td>
    </tr>
    <tr>
      <th>703</th>
      <td>132</td>
      <td>145</td>
      <td>p5</td>
      <td>fu</td>
    </tr>
    <tr>
      <th>704</th>
      <td>149</td>
      <td>162</td>
      <td>p5</td>
      <td>fu</td>
    </tr>
    <tr>
      <th>705</th>
      <td>179</td>
      <td>192</td>
      <td>p5</td>
      <td>fu</td>
    </tr>
    <tr>
      <th>706</th>
      <td>192</td>
      <td>205</td>
      <td>p5</td>
      <td>fu</td>
    </tr>
    <tr>
      <th>707</th>
      <td>241</td>
      <td>254</td>
      <td>p5</td>
      <td>fu</td>
    </tr>
  </tbody>
</table>
<p>708 rows × 4 columns</p>
</div>



We have extracted the shots out of the continous data we had. Total data: 708 (Not much, but let's see). Now let's see how much imbalance is there in our data. We'll handle that (if any). And then move on to adding features.


```python
sns.set_style('whitegrid')
plt.figure(figsize=(16,8))
plt.title('Data provided by each user', fontsize=20)
sns.countplot(x='PersonID', hue='ShotName', data = X_y)
plt.savefig("plots/data_count")

plt.show()
```


![png](output_17_0.png)



```python
shotFullName = {
    'fo': "Overhead Forehand",
    'bo': "Overhead Backhand",
    'fs': "Forehand Smash",
    'fu': "Underarm Forehand Stroke",
    'bu': "Underarm Backhand Stroke",
}
plt.title('No of Datapoints per Stroke', fontsize=15)
sns.countplot([shotFullName[i] for i in X_y.ShotName]).set_xticklabels([shotFullName[i] for i in shots], 
                                                                       rotation=30, horizontalalignment='right')

plt.tight_layout()
plt.savefig("plots/stroke_count")
plt.show()
```


![png](output_18_0.png)


There is a lot of difference in the amount of data we collected from each person (We didn't realized it during collection). But there isn't much imbalace between the classes (there is, but not much), so we'll work with this only.

## Adding Features for classical ML models

We will be using this X_y dataframe to store all our features for all the samples. And then use it for training classical machine learning models.

After some searching we found following features that we can add to such a time series data.
* Range
* Minimum
* Maximum
* Average
* Absolute Average
* Kurtosis f 
* Kurtosis p 
* Skewness statistic and p value
* Entropy
* Standard Deviation
* Angle betweenness
* Inter quartile range
* maxmin relative position ( Max position - min position , to see if max comes before min or vice-versa)


```python
# List of features we'll add
features = []
```


```python
# Some helper functions

# Add feature which depends only on one sensor, like range
def add_feature(fname, sensor):
    v = [fname(data[str(row['PersonID']) + "_" + str(row['ShotName'])][int(row['StartFrame']):int(row['EndFrame'])],
              sensor)
            for index, row in X_y.iterrows()]
    X_y[fname.__name__ + str(sensor)] = v
    if(fname.__name__ + str(sensor) not in features):
        features.append(fname.__name__ + str(sensor))
    print("Added feature " + fname.__name__ + str(sensor) + " for " + str(len(v)) + " rows.")
    
# Add feature which depends on more than one sensors, like magnitude
def add_feature_mult_sensor(fname, sensors):
    v = [fname(data[str(row['PersonID']) + "_" + str(row['ShotName'])][int(row['StartFrame']):int(row['EndFrame'])],
              sensors)
             for index, row in X_y.iterrows()]
    
    name = "_".join(sensors)
    X_y[fname.__name__ + name] = v
    if(fname.__name__ + name not in features):
        features.append(fname.__name__ + name)
    print("Added feature " + fname.__name__ + name + " for " + str(len(v)) + " rows.")
```


```python
# Range 
def range_(df, sensor):
    return np.max(df[sensor]) - np.min(df[sensor])
for sensor in cols + addedCols:
    add_feature(range_, sensor)
```

    Added feature range_ax for 708 rows.
    Added feature range_ay for 708 rows.
    Added feature range_az for 708 rows.
    Added feature range_gx for 708 rows.
    Added feature range_gy for 708 rows.
    Added feature range_gz for 708 rows.
    Added feature range_acc_mag for 708 rows.
    Added feature range_gyro_mag for 708 rows.



```python
# Minimum
def min_(df, sensor):
    return np.min(df[sensor])
for sensor in cols + addedCols:
    add_feature(min_, sensor)
```

    Added feature min_ax for 708 rows.
    Added feature min_ay for 708 rows.
    Added feature min_az for 708 rows.
    Added feature min_gx for 708 rows.
    Added feature min_gy for 708 rows.
    Added feature min_gz for 708 rows.
    Added feature min_acc_mag for 708 rows.
    Added feature min_gyro_mag for 708 rows.



```python
# Maximum
def max_(df, sensor):
    return np.max(df[sensor])
for sensor in cols + addedCols:
    add_feature(max_, sensor)
```

    Added feature max_ax for 708 rows.
    Added feature max_ay for 708 rows.
    Added feature max_az for 708 rows.
    Added feature max_gx for 708 rows.
    Added feature max_gy for 708 rows.
    Added feature max_gz for 708 rows.
    Added feature max_acc_mag for 708 rows.
    Added feature max_gyro_mag for 708 rows.



```python
# Average
def avg_(df, sensor):
    return np.mean(df[sensor])
for sensor in cols + addedCols:
    add_feature(avg_, sensor)
```

    Added feature avg_ax for 708 rows.
    Added feature avg_ay for 708 rows.
    Added feature avg_az for 708 rows.
    Added feature avg_gx for 708 rows.
    Added feature avg_gy for 708 rows.
    Added feature avg_gz for 708 rows.
    Added feature avg_acc_mag for 708 rows.
    Added feature avg_gyro_mag for 708 rows.



```python
# Absolute Average
def absavg_(df, sensor):
    return np.mean(np.absolute(df[sensor]))
for sensor in cols + addedCols:
    add_feature(absavg_, sensor)
```

    Added feature absavg_ax for 708 rows.
    Added feature absavg_ay for 708 rows.
    Added feature absavg_az for 708 rows.
    Added feature absavg_gx for 708 rows.
    Added feature absavg_gy for 708 rows.
    Added feature absavg_gz for 708 rows.
    Added feature absavg_acc_mag for 708 rows.
    Added feature absavg_gyro_mag for 708 rows.



```python
def kurtosis_f_(df , sensor):
    from scipy.stats import kurtosis 
    val = kurtosis(df[sensor],fisher = True)
    return val
for sensor in cols + addedCols:
    add_feature(kurtosis_f_, sensor)
```

    Added feature kurtosis_f_ax for 708 rows.
    Added feature kurtosis_f_ay for 708 rows.
    Added feature kurtosis_f_az for 708 rows.
    Added feature kurtosis_f_gx for 708 rows.
    Added feature kurtosis_f_gy for 708 rows.
    Added feature kurtosis_f_gz for 708 rows.
    Added feature kurtosis_f_acc_mag for 708 rows.
    Added feature kurtosis_f_gyro_mag for 708 rows.



```python
def kurtosis_p_(df , sensor):
    from scipy.stats import kurtosis 
    val = kurtosis(df[sensor],fisher = False)
    return val
for sensor in cols + addedCols:
    add_feature(kurtosis_p_, sensor)
```

    Added feature kurtosis_p_ax for 708 rows.
    Added feature kurtosis_p_ay for 708 rows.
    Added feature kurtosis_p_az for 708 rows.
    Added feature kurtosis_p_gx for 708 rows.
    Added feature kurtosis_p_gy for 708 rows.
    Added feature kurtosis_p_gz for 708 rows.
    Added feature kurtosis_p_acc_mag for 708 rows.
    Added feature kurtosis_p_gyro_mag for 708 rows.



```python
#skewness
def skewness_statistic_(df, sensor):
    if(len(df) == 0):
        print(df)
    from scipy.stats import skewtest 
    statistic, pvalue = skewtest(df[sensor], nan_policy='propagate')
    return statistic
for sensor in cols + addedCols:
    add_feature(skewness_statistic_, sensor)
```

    Added feature skewness_statistic_ax for 708 rows.
    Added feature skewness_statistic_ay for 708 rows.
    Added feature skewness_statistic_az for 708 rows.
    Added feature skewness_statistic_gx for 708 rows.
    Added feature skewness_statistic_gy for 708 rows.
    Added feature skewness_statistic_gz for 708 rows.
    Added feature skewness_statistic_acc_mag for 708 rows.
    Added feature skewness_statistic_gyro_mag for 708 rows.



```python
def skewness_pvalue_(df, sensor):
    from scipy.stats import skewtest 
    statistic, pvalue = skewtest(df[sensor])
    return pvalue
for sensor in cols + addedCols:
    add_feature(skewness_pvalue_, sensor)
```

    Added feature skewness_pvalue_ax for 708 rows.
    Added feature skewness_pvalue_ay for 708 rows.
    Added feature skewness_pvalue_az for 708 rows.
    Added feature skewness_pvalue_gx for 708 rows.
    Added feature skewness_pvalue_gy for 708 rows.
    Added feature skewness_pvalue_gz for 708 rows.
    Added feature skewness_pvalue_acc_mag for 708 rows.
    Added feature skewness_pvalue_gyro_mag for 708 rows.



```python
#entropy 
def entropy_(df, sensor):
    from scipy.stats import entropy
    ent = entropy(df[sensor])
    return ent
for sensor in addedCols:
    add_feature(entropy_, sensor)
```

    Added feature entropy_acc_mag for 708 rows.
    Added feature entropy_gyro_mag for 708 rows.



```python
# Standard Deviation
def std_(df, sensor):
    return np.std(df[sensor])
for sensor in cols + addedCols:
    add_feature(std_, sensor)
```

    Added feature std_ax for 708 rows.
    Added feature std_ay for 708 rows.
    Added feature std_az for 708 rows.
    Added feature std_gx for 708 rows.
    Added feature std_gy for 708 rows.
    Added feature std_gz for 708 rows.
    Added feature std_acc_mag for 708 rows.
    Added feature std_gyro_mag for 708 rows.



```python
#angle between two vectors
def anglebetween_(df, sensors):
    v1 = sensors[0]
    v2 = sensors[1]
    v1_u = df[v1] / np.linalg.norm(df[v1])
    v2_u = df[v2] / np.linalg.norm(df[v2])
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
add_feature_mult_sensor(anglebetween_, ["ax", "ay"])
add_feature_mult_sensor(anglebetween_, ["ay", "az"])
add_feature_mult_sensor(anglebetween_, ["ax", "az"])
add_feature_mult_sensor(anglebetween_, ["gx", "gy"])
add_feature_mult_sensor(anglebetween_, ["gy", "gz"])
add_feature_mult_sensor(anglebetween_, ["gx", "gz"])
```

    Added feature anglebetween_ax_ay for 708 rows.
    Added feature anglebetween_ay_az for 708 rows.
    Added feature anglebetween_ax_az for 708 rows.
    Added feature anglebetween_gx_gy for 708 rows.
    Added feature anglebetween_gy_gz for 708 rows.
    Added feature anglebetween_gx_gz for 708 rows.



```python
#inter quartile range
def iqr_(df, sensor):
    from scipy import stats
    return stats.iqr(df[sensor])
for sensor in cols + addedCols:
    add_feature(iqr_, sensor)
```

    Added feature iqr_ax for 708 rows.
    Added feature iqr_ay for 708 rows.
    Added feature iqr_az for 708 rows.
    Added feature iqr_gx for 708 rows.
    Added feature iqr_gy for 708 rows.
    Added feature iqr_gz for 708 rows.
    Added feature iqr_acc_mag for 708 rows.
    Added feature iqr_gyro_mag for 708 rows.



```python
# Max position - min position (relative difference)
def maxmin_relative_pos_(df, sensor):
    return np.argmax(np.array(df[sensor])) - np.argmin(np.array(df[sensor]))
for sensor in cols + addedCols:
    add_feature(maxmin_relative_pos_, sensor)
```

    Added feature maxmin_relative_pos_ax for 708 rows.
    Added feature maxmin_relative_pos_ay for 708 rows.
    Added feature maxmin_relative_pos_az for 708 rows.
    Added feature maxmin_relative_pos_gx for 708 rows.
    Added feature maxmin_relative_pos_gy for 708 rows.
    Added feature maxmin_relative_pos_gz for 708 rows.
    Added feature maxmin_relative_pos_acc_mag for 708 rows.
    Added feature maxmin_relative_pos_gyro_mag for 708 rows.


### Saving the processed data

Our final X_y, with all the hand-engineered features, is ready. Now we can do some "Machine Learning" on it. But before that, it's better to save this X_y, so that we need not do any preprocessing and we can do all ML model testing on a different notebook. (For case of this article, we'll continue in same notebook).


```python
X_y
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StartFrame</th>
      <th>EndFrame</th>
      <th>PersonID</th>
      <th>ShotName</th>
      <th>range_ax</th>
      <th>range_ay</th>
      <th>range_az</th>
      <th>range_gx</th>
      <th>range_gy</th>
      <th>range_gz</th>
      <th>range_acc_mag</th>
      <th>range_gyro_mag</th>
      <th>min_ax</th>
      <th>min_ay</th>
      <th>min_az</th>
      <th>min_gx</th>
      <th>min_gy</th>
      <th>min_gz</th>
      <th>min_acc_mag</th>
      <th>min_gyro_mag</th>
      <th>max_ax</th>
      <th>max_ay</th>
      <th>max_az</th>
      <th>max_gx</th>
      <th>max_gy</th>
      <th>max_gz</th>
      <th>max_acc_mag</th>
      <th>max_gyro_mag</th>
      <th>avg_ax</th>
      <th>avg_ay</th>
      <th>avg_az</th>
      <th>avg_gx</th>
      <th>avg_gy</th>
      <th>avg_gz</th>
      <th>avg_acc_mag</th>
      <th>avg_gyro_mag</th>
      <th>absavg_ax</th>
      <th>absavg_ay</th>
      <th>absavg_az</th>
      <th>absavg_gx</th>
      <th>absavg_gy</th>
      <th>absavg_gz</th>
      <th>absavg_acc_mag</th>
      <th>absavg_gyro_mag</th>
      <th>kurtosis_f_ax</th>
      <th>kurtosis_f_ay</th>
      <th>kurtosis_f_az</th>
      <th>kurtosis_f_gx</th>
      <th>kurtosis_f_gy</th>
      <th>kurtosis_f_gz</th>
      <th>kurtosis_f_acc_mag</th>
      <th>kurtosis_f_gyro_mag</th>
      <th>kurtosis_p_ax</th>
      <th>kurtosis_p_ay</th>
      <th>kurtosis_p_az</th>
      <th>kurtosis_p_gx</th>
      <th>kurtosis_p_gy</th>
      <th>kurtosis_p_gz</th>
      <th>kurtosis_p_acc_mag</th>
      <th>kurtosis_p_gyro_mag</th>
      <th>skewness_statistic_ax</th>
      <th>skewness_statistic_ay</th>
      <th>skewness_statistic_az</th>
      <th>skewness_statistic_gx</th>
      <th>skewness_statistic_gy</th>
      <th>skewness_statistic_gz</th>
      <th>skewness_statistic_acc_mag</th>
      <th>skewness_statistic_gyro_mag</th>
      <th>skewness_pvalue_ax</th>
      <th>skewness_pvalue_ay</th>
      <th>skewness_pvalue_az</th>
      <th>skewness_pvalue_gx</th>
      <th>skewness_pvalue_gy</th>
      <th>skewness_pvalue_gz</th>
      <th>skewness_pvalue_acc_mag</th>
      <th>skewness_pvalue_gyro_mag</th>
      <th>entropy_acc_mag</th>
      <th>entropy_gyro_mag</th>
      <th>std_ax</th>
      <th>std_ay</th>
      <th>std_az</th>
      <th>std_gx</th>
      <th>std_gy</th>
      <th>std_gz</th>
      <th>std_acc_mag</th>
      <th>std_gyro_mag</th>
      <th>anglebetween_ax_ay</th>
      <th>anglebetween_ay_az</th>
      <th>anglebetween_ax_az</th>
      <th>anglebetween_gx_gy</th>
      <th>anglebetween_gy_gz</th>
      <th>anglebetween_gx_gz</th>
      <th>iqr_ax</th>
      <th>iqr_ay</th>
      <th>iqr_az</th>
      <th>iqr_gx</th>
      <th>iqr_gy</th>
      <th>iqr_gz</th>
      <th>iqr_acc_mag</th>
      <th>iqr_gyro_mag</th>
      <th>maxmin_relative_pos_ax</th>
      <th>maxmin_relative_pos_ay</th>
      <th>maxmin_relative_pos_az</th>
      <th>maxmin_relative_pos_gx</th>
      <th>maxmin_relative_pos_gy</th>
      <th>maxmin_relative_pos_gz</th>
      <th>maxmin_relative_pos_acc_mag</th>
      <th>maxmin_relative_pos_gyro_mag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>17</td>
      <td>p1</td>
      <td>bo</td>
      <td>3.05341</td>
      <td>1.84393</td>
      <td>1.12860</td>
      <td>227.69928</td>
      <td>465.68298</td>
      <td>439.82696</td>
      <td>2.399066</td>
      <td>388.228389</td>
      <td>-1.36011</td>
      <td>0.15601</td>
      <td>-1.11713</td>
      <td>-42.58728</td>
      <td>-250.00000</td>
      <td>-189.83459</td>
      <td>0.449619</td>
      <td>10.848844</td>
      <td>1.69330</td>
      <td>1.99994</td>
      <td>0.01147</td>
      <td>185.11200</td>
      <td>215.68298</td>
      <td>249.99237</td>
      <td>2.848685</td>
      <td>399.077233</td>
      <td>-0.327224</td>
      <td>0.929121</td>
      <td>-0.313877</td>
      <td>25.042607</td>
      <td>12.265720</td>
      <td>-8.008517</td>
      <td>1.209606</td>
      <td>101.207901</td>
      <td>0.587732</td>
      <td>0.929121</td>
      <td>0.315642</td>
      <td>38.159298</td>
      <td>60.093806</td>
      <td>64.064612</td>
      <td>1.209606</td>
      <td>101.207901</td>
      <td>4.332231</td>
      <td>1.428159</td>
      <td>4.290885</td>
      <td>2.603525</td>
      <td>2.390531</td>
      <td>1.736930</td>
      <td>4.257268</td>
      <td>1.196086</td>
      <td>7.332231</td>
      <td>4.428159</td>
      <td>7.290885</td>
      <td>5.603525</td>
      <td>5.390531</td>
      <td>4.736930</td>
      <td>7.257268</td>
      <td>4.196086</td>
      <td>3.099352</td>
      <td>1.401094</td>
      <td>-3.202052</td>
      <td>2.811551</td>
      <td>-1.303761</td>
      <td>1.307331</td>
      <td>3.281096</td>
      <td>2.701275</td>
      <td>0.001939</td>
      <td>0.161186</td>
      <td>0.001365</td>
      <td>0.004930</td>
      <td>0.192315</td>
      <td>0.191100</td>
      <td>0.001034</td>
      <td>0.006907</td>
      <td>2.482802</td>
      <td>2.050504</td>
      <td>0.658365</td>
      <td>0.419744</td>
      <td>0.260681</td>
      <td>55.497505</td>
      <td>98.367075</td>
      <td>98.854521</td>
      <td>0.532256</td>
      <td>114.572893</td>
      <td>1.705647</td>
      <td>2.592944</td>
      <td>1.704230</td>
      <td>2.057122</td>
      <td>2.872174</td>
      <td>1.025739</td>
      <td>0.15014</td>
      <td>0.09363</td>
      <td>0.07550</td>
      <td>52.20032</td>
      <td>40.33660</td>
      <td>57.17468</td>
      <td>0.246593</td>
      <td>63.851815</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>20</td>
      <td>p1</td>
      <td>bo</td>
      <td>3.05341</td>
      <td>1.84393</td>
      <td>1.12860</td>
      <td>227.69928</td>
      <td>465.68298</td>
      <td>439.82696</td>
      <td>2.399066</td>
      <td>388.228389</td>
      <td>-1.36011</td>
      <td>0.15601</td>
      <td>-1.11713</td>
      <td>-42.58728</td>
      <td>-250.00000</td>
      <td>-189.83459</td>
      <td>0.449619</td>
      <td>10.848844</td>
      <td>1.69330</td>
      <td>1.99994</td>
      <td>0.01147</td>
      <td>185.11200</td>
      <td>215.68298</td>
      <td>249.99237</td>
      <td>2.848685</td>
      <td>399.077233</td>
      <td>-0.358685</td>
      <td>0.925064</td>
      <td>-0.312102</td>
      <td>22.215623</td>
      <td>2.295274</td>
      <td>-9.633578</td>
      <td>1.219778</td>
      <td>114.757904</td>
      <td>0.619192</td>
      <td>0.925064</td>
      <td>0.313867</td>
      <td>40.818435</td>
      <td>71.522052</td>
      <td>68.692135</td>
      <td>1.219778</td>
      <td>114.757904</td>
      <td>4.516519</td>
      <td>1.448142</td>
      <td>3.658435</td>
      <td>2.278342</td>
      <td>0.921873</td>
      <td>1.503207</td>
      <td>4.240144</td>
      <td>0.966073</td>
      <td>7.516519</td>
      <td>4.448142</td>
      <td>6.658435</td>
      <td>5.278342</td>
      <td>3.921873</td>
      <td>4.503207</td>
      <td>7.240144</td>
      <td>3.966073</td>
      <td>3.227788</td>
      <td>1.449650</td>
      <td>-3.019751</td>
      <td>2.709864</td>
      <td>-0.929834</td>
      <td>1.324887</td>
      <td>3.250108</td>
      <td>2.483311</td>
      <td>0.001248</td>
      <td>0.147156</td>
      <td>0.002530</td>
      <td>0.006731</td>
      <td>0.352457</td>
      <td>0.185209</td>
      <td>0.001154</td>
      <td>0.013017</td>
      <td>2.484444</td>
      <td>2.164318</td>
      <td>0.662445</td>
      <td>0.419967</td>
      <td>0.267336</td>
      <td>57.339286</td>
      <td>107.273813</td>
      <td>100.409502</td>
      <td>0.529737</td>
      <td>110.904302</td>
      <td>1.738168</td>
      <td>2.572913</td>
      <td>1.665239</td>
      <td>1.961745</td>
      <td>2.788544</td>
      <td>1.052583</td>
      <td>0.27136</td>
      <td>0.09363</td>
      <td>0.21216</td>
      <td>64.89563</td>
      <td>64.77356</td>
      <td>85.98327</td>
      <td>0.246593</td>
      <td>117.644558</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>35</td>
      <td>p1</td>
      <td>bo</td>
      <td>2.59113</td>
      <td>1.79791</td>
      <td>2.55182</td>
      <td>341.14075</td>
      <td>314.94904</td>
      <td>470.93964</td>
      <td>2.820702</td>
      <td>359.692988</td>
      <td>-2.00000</td>
      <td>0.20203</td>
      <td>-0.55188</td>
      <td>-190.50598</td>
      <td>-64.95667</td>
      <td>-220.94727</td>
      <td>0.643330</td>
      <td>41.909924</td>
      <td>0.59113</td>
      <td>1.99994</td>
      <td>1.99994</td>
      <td>150.63477</td>
      <td>249.99237</td>
      <td>249.99237</td>
      <td>3.464032</td>
      <td>401.602912</td>
      <td>-0.589476</td>
      <td>0.843896</td>
      <td>0.092327</td>
      <td>-4.143347</td>
      <td>38.427499</td>
      <td>-11.661236</td>
      <td>1.238610</td>
      <td>123.438871</td>
      <td>0.680419</td>
      <td>0.843896</td>
      <td>0.390347</td>
      <td>48.787044</td>
      <td>74.330258</td>
      <td>65.545888</td>
      <td>1.238610</td>
      <td>123.438871</td>
      <td>1.143829</td>
      <td>2.090847</td>
      <td>3.104243</td>
      <td>1.552827</td>
      <td>0.082220</td>
      <td>2.011180</td>
      <td>5.666403</td>
      <td>1.822596</td>
      <td>4.143829</td>
      <td>5.090847</td>
      <td>6.104243</td>
      <td>4.552827</td>
      <td>3.082220</td>
      <td>5.011180</td>
      <td>8.666403</td>
      <td>4.822596</td>
      <td>-1.024435</td>
      <td>2.097186</td>
      <td>3.136364</td>
      <td>-0.940783</td>
      <td>1.774096</td>
      <td>1.156026</td>
      <td>3.860729</td>
      <td>2.951899</td>
      <td>0.305630</td>
      <td>0.035977</td>
      <td>0.001711</td>
      <td>0.346816</td>
      <td>0.076047</td>
      <td>0.247670</td>
      <td>0.000113</td>
      <td>0.003158</td>
      <td>2.449221</td>
      <td>2.297862</td>
      <td>0.582813</td>
      <td>0.419766</td>
      <td>0.647254</td>
      <td>74.576184</td>
      <td>91.462439</td>
      <td>100.737093</td>
      <td>0.684709</td>
      <td>102.311370</td>
      <td>2.619621</td>
      <td>1.037181</td>
      <td>2.228971</td>
      <td>1.983909</td>
      <td>1.475343</td>
      <td>2.013400</td>
      <td>0.39808</td>
      <td>0.30738</td>
      <td>0.32989</td>
      <td>36.96441</td>
      <td>85.13642</td>
      <td>46.69189</td>
      <td>0.476475</td>
      <td>73.961360</td>
      <td>-1</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>-3</td>
      <td>2</td>
      <td>-3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>73</td>
      <td>p1</td>
      <td>bo</td>
      <td>1.42700</td>
      <td>1.93488</td>
      <td>0.81769</td>
      <td>181.92291</td>
      <td>474.31946</td>
      <td>475.95978</td>
      <td>1.866181</td>
      <td>349.843787</td>
      <td>-1.31549</td>
      <td>0.06506</td>
      <td>-0.80542</td>
      <td>-54.55017</td>
      <td>-250.00000</td>
      <td>-225.96741</td>
      <td>0.607624</td>
      <td>5.044586</td>
      <td>0.11151</td>
      <td>1.99994</td>
      <td>0.01227</td>
      <td>127.37274</td>
      <td>224.31946</td>
      <td>249.99237</td>
      <td>2.473805</td>
      <td>354.888372</td>
      <td>-0.576142</td>
      <td>0.907255</td>
      <td>-0.311669</td>
      <td>10.797940</td>
      <td>0.294612</td>
      <td>-7.702168</td>
      <td>1.188472</td>
      <td>115.416716</td>
      <td>0.593298</td>
      <td>0.907255</td>
      <td>0.313557</td>
      <td>34.821143</td>
      <td>64.475426</td>
      <td>68.515485</td>
      <td>1.188472</td>
      <td>115.416716</td>
      <td>-0.143819</td>
      <td>3.155069</td>
      <td>0.001100</td>
      <td>0.569585</td>
      <td>1.349121</td>
      <td>1.204342</td>
      <td>3.627210</td>
      <td>0.126530</td>
      <td>2.856181</td>
      <td>6.155069</td>
      <td>3.001100</td>
      <td>3.569585</td>
      <td>4.349121</td>
      <td>4.204342</td>
      <td>6.627210</td>
      <td>3.126530</td>
      <td>-0.694345</td>
      <td>1.587864</td>
      <td>-1.641120</td>
      <td>1.952565</td>
      <td>-0.720400</td>
      <td>1.121208</td>
      <td>3.191432</td>
      <td>1.971372</td>
      <td>0.487466</td>
      <td>0.112317</td>
      <td>0.100773</td>
      <td>0.050871</td>
      <td>0.471279</td>
      <td>0.262199</td>
      <td>0.001416</td>
      <td>0.048681</td>
      <td>2.509193</td>
      <td>2.143942</td>
      <td>0.374384</td>
      <td>0.394391</td>
      <td>0.218167</td>
      <td>47.383947</td>
      <td>104.871959</td>
      <td>107.420121</td>
      <td>0.427817</td>
      <td>107.876949</td>
      <td>2.698599</td>
      <td>2.506891</td>
      <td>0.736958</td>
      <td>1.305000</td>
      <td>2.572208</td>
      <td>1.627996</td>
      <td>0.30560</td>
      <td>0.03504</td>
      <td>0.07873</td>
      <td>44.90662</td>
      <td>42.21344</td>
      <td>60.92072</td>
      <td>0.158270</td>
      <td>115.045971</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>75</td>
      <td>88</td>
      <td>p1</td>
      <td>bo</td>
      <td>1.67822</td>
      <td>1.68512</td>
      <td>2.73737</td>
      <td>170.34912</td>
      <td>499.99237</td>
      <td>499.99237</td>
      <td>2.544345</td>
      <td>380.297801</td>
      <td>-2.00000</td>
      <td>0.31482</td>
      <td>-0.73743</td>
      <td>-20.05005</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>0.919687</td>
      <td>3.871427</td>
      <td>-0.32178</td>
      <td>1.99994</td>
      <td>1.99994</td>
      <td>150.29907</td>
      <td>249.99237</td>
      <td>249.99237</td>
      <td>3.464032</td>
      <td>384.169228</td>
      <td>-0.891581</td>
      <td>0.959407</td>
      <td>-0.043837</td>
      <td>29.310960</td>
      <td>-7.772006</td>
      <td>12.432979</td>
      <td>1.476049</td>
      <td>97.842099</td>
      <td>0.891581</td>
      <td>0.959407</td>
      <td>0.442040</td>
      <td>36.896925</td>
      <td>57.654162</td>
      <td>60.988792</td>
      <td>1.476049</td>
      <td>97.842099</td>
      <td>-1.267835</td>
      <td>0.690171</td>
      <td>4.344974</td>
      <td>0.737945</td>
      <td>2.347301</td>
      <td>2.147072</td>
      <td>1.109940</td>
      <td>0.733589</td>
      <td>1.732165</td>
      <td>3.690171</td>
      <td>7.344974</td>
      <td>3.737945</td>
      <td>5.347301</td>
      <td>5.147072</td>
      <td>4.109940</td>
      <td>3.733589</td>
      <td>-1.317498</td>
      <td>2.240916</td>
      <td>3.526968</td>
      <td>2.189892</td>
      <td>0.278057</td>
      <td>-0.490533</td>
      <td>2.679185</td>
      <td>2.582531</td>
      <td>0.187672</td>
      <td>0.025032</td>
      <td>0.000420</td>
      <td>0.028532</td>
      <td>0.780968</td>
      <td>0.623757</td>
      <td>0.007380</td>
      <td>0.009808</td>
      <td>2.451057</td>
      <td>1.900606</td>
      <td>0.621917</td>
      <td>0.481607</td>
      <td>0.657929</td>
      <td>47.483096</td>
      <td>103.431016</td>
      <td>104.600460</td>
      <td>0.768229</td>
      <td>124.076246</td>
      <td>2.553833</td>
      <td>1.261289</td>
      <td>1.802243</td>
      <td>2.020222</td>
      <td>2.708513</td>
      <td>1.017772</td>
      <td>1.29065</td>
      <td>0.28472</td>
      <td>0.20545</td>
      <td>44.05212</td>
      <td>29.65546</td>
      <td>51.49841</td>
      <td>0.766604</td>
      <td>98.432232</td>
      <td>-7</td>
      <td>-3</td>
      <td>1</td>
      <td>3</td>
      <td>-1</td>
      <td>1</td>
      <td>-3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>77</td>
      <td>90</td>
      <td>p1</td>
      <td>bo</td>
      <td>1.67822</td>
      <td>1.68512</td>
      <td>2.73737</td>
      <td>170.34912</td>
      <td>499.99237</td>
      <td>499.99237</td>
      <td>2.544345</td>
      <td>380.297801</td>
      <td>-2.00000</td>
      <td>0.31482</td>
      <td>-0.73743</td>
      <td>-20.05005</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>0.919687</td>
      <td>3.871427</td>
      <td>-0.32178</td>
      <td>1.99994</td>
      <td>1.99994</td>
      <td>150.29907</td>
      <td>249.99237</td>
      <td>249.99237</td>
      <td>3.464032</td>
      <td>384.169228</td>
      <td>-0.898568</td>
      <td>0.983432</td>
      <td>-0.033104</td>
      <td>32.079844</td>
      <td>-8.632954</td>
      <td>18.804111</td>
      <td>1.496871</td>
      <td>103.660160</td>
      <td>0.898568</td>
      <td>0.983432</td>
      <td>0.431307</td>
      <td>36.835890</td>
      <td>61.641986</td>
      <td>67.249591</td>
      <td>1.496871</td>
      <td>103.660160</td>
      <td>-1.260085</td>
      <td>0.395367</td>
      <td>4.440146</td>
      <td>0.872848</td>
      <td>2.117778</td>
      <td>2.089812</td>
      <td>1.146730</td>
      <td>0.705884</td>
      <td>1.739915</td>
      <td>3.395367</td>
      <td>7.440146</td>
      <td>3.872848</td>
      <td>5.117778</td>
      <td>5.089812</td>
      <td>4.146730</td>
      <td>3.705884</td>
      <td>-1.329097</td>
      <td>1.989036</td>
      <td>3.548791</td>
      <td>2.236668</td>
      <td>0.316184</td>
      <td>-0.808471</td>
      <td>2.680918</td>
      <td>2.523354</td>
      <td>0.183816</td>
      <td>0.046697</td>
      <td>0.000387</td>
      <td>0.025308</td>
      <td>0.751862</td>
      <td>0.418819</td>
      <td>0.007342</td>
      <td>0.011624</td>
      <td>2.456716</td>
      <td>1.975242</td>
      <td>0.616361</td>
      <td>0.483733</td>
      <td>0.652414</td>
      <td>45.892042</td>
      <td>104.622579</td>
      <td>105.331553</td>
      <td>0.758008</td>
      <td>121.895486</td>
      <td>2.545061</td>
      <td>1.256510</td>
      <td>1.808812</td>
      <td>2.021724</td>
      <td>2.689395</td>
      <td>0.989975</td>
      <td>1.23761</td>
      <td>0.34674</td>
      <td>0.20471</td>
      <td>43.99108</td>
      <td>43.71643</td>
      <td>74.51630</td>
      <td>0.766358</td>
      <td>98.432232</td>
      <td>-7</td>
      <td>-3</td>
      <td>1</td>
      <td>3</td>
      <td>-1</td>
      <td>1</td>
      <td>-3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>94</td>
      <td>107</td>
      <td>p1</td>
      <td>bo</td>
      <td>3.10632</td>
      <td>1.23432</td>
      <td>0.92858</td>
      <td>354.01153</td>
      <td>493.66760</td>
      <td>437.27112</td>
      <td>1.291482</td>
      <td>312.450390</td>
      <td>-2.00000</td>
      <td>0.17230</td>
      <td>-0.76031</td>
      <td>-241.38641</td>
      <td>-243.67523</td>
      <td>-202.15607</td>
      <td>0.908957</td>
      <td>9.297542</td>
      <td>1.10632</td>
      <td>1.40662</td>
      <td>0.16827</td>
      <td>112.62512</td>
      <td>249.99237</td>
      <td>235.11505</td>
      <td>2.200438</td>
      <td>321.747932</td>
      <td>-0.526227</td>
      <td>0.833115</td>
      <td>-0.282884</td>
      <td>1.199575</td>
      <td>15.411377</td>
      <td>3.484873</td>
      <td>1.242495</td>
      <td>155.998273</td>
      <td>0.699604</td>
      <td>0.833115</td>
      <td>0.308772</td>
      <td>62.606811</td>
      <td>107.174214</td>
      <td>67.273653</td>
      <td>1.242495</td>
      <td>155.998273</td>
      <td>1.555072</td>
      <td>-0.603796</td>
      <td>1.280847</td>
      <td>1.627893</td>
      <td>-0.611660</td>
      <td>1.042730</td>
      <td>2.723135</td>
      <td>-1.682720</td>
      <td>4.555072</td>
      <td>2.396204</td>
      <td>4.280847</td>
      <td>4.627893</td>
      <td>2.388340</td>
      <td>4.042730</td>
      <td>5.723135</td>
      <td>1.317280</td>
      <td>0.529287</td>
      <td>-0.523077</td>
      <td>-0.195113</td>
      <td>-2.243886</td>
      <td>0.399532</td>
      <td>0.646515</td>
      <td>2.908711</td>
      <td>0.125787</td>
      <td>0.596606</td>
      <td>0.600920</td>
      <td>0.845304</td>
      <td>0.024840</td>
      <td>0.689501</td>
      <td>0.517946</td>
      <td>0.003629</td>
      <td>0.899900</td>
      <td>2.533484</td>
      <td>2.256316</td>
      <td>0.671632</td>
      <td>0.329076</td>
      <td>0.204705</td>
      <td>89.512982</td>
      <td>139.393102</td>
      <td>98.219873</td>
      <td>0.329408</td>
      <td>114.042918</td>
      <td>2.190027</td>
      <td>2.423324</td>
      <td>0.569889</td>
      <td>2.106033</td>
      <td>2.377283</td>
      <td>1.298514</td>
      <td>0.46142</td>
      <td>0.50659</td>
      <td>0.09326</td>
      <td>88.89007</td>
      <td>182.25098</td>
      <td>77.53754</td>
      <td>0.296997</td>
      <td>228.726362</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>-4</td>
      <td>-4</td>
      <td>3</td>
      <td>-7</td>
      <td>-8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>143</td>
      <td>156</td>
      <td>p1</td>
      <td>bo</td>
      <td>1.67023</td>
      <td>1.79779</td>
      <td>2.48444</td>
      <td>221.41266</td>
      <td>477.60010</td>
      <td>499.99237</td>
      <td>2.480478</td>
      <td>389.998399</td>
      <td>-2.00000</td>
      <td>0.20215</td>
      <td>-0.48450</td>
      <td>-42.26685</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>0.983554</td>
      <td>6.346661</td>
      <td>-0.32977</td>
      <td>1.99994</td>
      <td>1.99994</td>
      <td>179.14581</td>
      <td>227.60010</td>
      <td>249.99237</td>
      <td>3.464032</td>
      <td>396.345060</td>
      <td>-0.656800</td>
      <td>0.961736</td>
      <td>-0.093036</td>
      <td>11.417681</td>
      <td>11.800912</td>
      <td>-5.396917</td>
      <td>1.304580</td>
      <td>113.814360</td>
      <td>0.656800</td>
      <td>0.961736</td>
      <td>0.400719</td>
      <td>37.712096</td>
      <td>75.215268</td>
      <td>68.376983</td>
      <td>1.304580</td>
      <td>113.814360</td>
      <td>1.648364</td>
      <td>3.663604</td>
      <td>7.393257</td>
      <td>3.200357</td>
      <td>0.748508</td>
      <td>1.627525</td>
      <td>6.593097</td>
      <td>0.377946</td>
      <td>4.648364</td>
      <td>6.663604</td>
      <td>10.393257</td>
      <td>6.200357</td>
      <td>3.748508</td>
      <td>4.627525</td>
      <td>9.593097</td>
      <td>3.377946</td>
      <td>-2.863239</td>
      <td>1.982122</td>
      <td>4.317681</td>
      <td>3.064810</td>
      <td>-0.828019</td>
      <td>0.478798</td>
      <td>4.158060</td>
      <td>2.309240</td>
      <td>0.004193</td>
      <td>0.047466</td>
      <td>0.000016</td>
      <td>0.002178</td>
      <td>0.407660</td>
      <td>0.632082</td>
      <td>0.000032</td>
      <td>0.020930</td>
      <td>2.474353</td>
      <td>2.070636</td>
      <td>0.496363</td>
      <td>0.362505</td>
      <td>0.614279</td>
      <td>56.490811</td>
      <td>111.721304</td>
      <td>107.960845</td>
      <td>0.646664</td>
      <td>121.134010</td>
      <td>2.577763</td>
      <td>1.418781</td>
      <td>1.908538</td>
      <td>2.345039</td>
      <td>2.805693</td>
      <td>0.720330</td>
      <td>0.48822</td>
      <td>0.04675</td>
      <td>0.17327</td>
      <td>53.80249</td>
      <td>50.39216</td>
      <td>34.99604</td>
      <td>0.117790</td>
      <td>130.106230</td>
      <td>5</td>
      <td>1</td>
      <td>-1</td>
      <td>2</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>162</td>
      <td>175</td>
      <td>p1</td>
      <td>bo</td>
      <td>2.15625</td>
      <td>1.83765</td>
      <td>2.09332</td>
      <td>345.12329</td>
      <td>499.99237</td>
      <td>391.51001</td>
      <td>2.397498</td>
      <td>429.639514</td>
      <td>-2.00000</td>
      <td>0.16229</td>
      <td>-0.85620</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>-141.51764</td>
      <td>0.689608</td>
      <td>3.364378</td>
      <td>0.15625</td>
      <td>1.99994</td>
      <td>1.23712</td>
      <td>95.12329</td>
      <td>249.99237</td>
      <td>249.99237</td>
      <td>3.087106</td>
      <td>433.003892</td>
      <td>-0.466810</td>
      <td>0.866972</td>
      <td>-0.199154</td>
      <td>-15.955412</td>
      <td>21.338242</td>
      <td>7.785503</td>
      <td>1.145207</td>
      <td>130.041764</td>
      <td>0.490848</td>
      <td>0.866972</td>
      <td>0.389480</td>
      <td>44.542166</td>
      <td>92.764634</td>
      <td>70.852428</td>
      <td>1.145207</td>
      <td>130.041764</td>
      <td>3.947451</td>
      <td>2.479860</td>
      <td>4.450730</td>
      <td>3.721980</td>
      <td>-0.035514</td>
      <td>0.404974</td>
      <td>6.931047</td>
      <td>0.193224</td>
      <td>6.947451</td>
      <td>5.479860</td>
      <td>7.450730</td>
      <td>6.721980</td>
      <td>2.964486</td>
      <td>3.404974</td>
      <td>9.931047</td>
      <td>3.193224</td>
      <td>-3.317391</td>
      <td>1.869342</td>
      <td>3.269808</td>
      <td>-3.014066</td>
      <td>-0.640014</td>
      <td>1.443177</td>
      <td>4.197251</td>
      <td>1.729616</td>
      <td>0.000909</td>
      <td>0.061575</td>
      <td>0.001076</td>
      <td>0.002578</td>
      <td>0.522163</td>
      <td>0.148971</td>
      <td>0.000027</td>
      <td>0.083699</td>
      <td>2.471646</td>
      <td>2.077734</td>
      <td>0.502397</td>
      <td>0.409488</td>
      <td>0.463576</td>
      <td>77.788037</td>
      <td>125.124325</td>
      <td>100.893260</td>
      <td>0.576803</td>
      <td>125.483763</td>
      <td>2.527545</td>
      <td>1.559490</td>
      <td>1.785308</td>
      <td>2.567806</td>
      <td>1.750129</td>
      <td>1.923466</td>
      <td>0.04278</td>
      <td>0.15271</td>
      <td>0.18634</td>
      <td>37.26196</td>
      <td>143.34869</td>
      <td>79.12445</td>
      <td>0.073978</td>
      <td>170.240947</td>
      <td>-1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>179</td>
      <td>192</td>
      <td>p1</td>
      <td>bo</td>
      <td>3.56635</td>
      <td>1.29492</td>
      <td>0.78571</td>
      <td>372.31445</td>
      <td>453.79639</td>
      <td>499.99237</td>
      <td>1.934282</td>
      <td>350.944507</td>
      <td>-2.00000</td>
      <td>0.70502</td>
      <td>-0.89270</td>
      <td>-250.00000</td>
      <td>-203.80402</td>
      <td>-250.00000</td>
      <td>1.031637</td>
      <td>2.945232</td>
      <td>1.56635</td>
      <td>1.99994</td>
      <td>-0.10699</td>
      <td>122.31445</td>
      <td>249.99237</td>
      <td>249.99237</td>
      <td>2.965919</td>
      <td>353.889739</td>
      <td>-0.483965</td>
      <td>1.095345</td>
      <td>-0.340632</td>
      <td>-15.943675</td>
      <td>1.118586</td>
      <td>7.897597</td>
      <td>1.416175</td>
      <td>123.565683</td>
      <td>0.724942</td>
      <td>1.095345</td>
      <td>0.340632</td>
      <td>44.575030</td>
      <td>71.773823</td>
      <td>67.641625</td>
      <td>1.416175</td>
      <td>123.565683</td>
      <td>1.824771</td>
      <td>2.933261</td>
      <td>1.984071</td>
      <td>3.287867</td>
      <td>0.426581</td>
      <td>1.058607</td>
      <td>1.618089</td>
      <td>-1.411902</td>
      <td>4.824771</td>
      <td>5.933261</td>
      <td>4.984071</td>
      <td>6.287867</td>
      <td>3.426581</td>
      <td>4.058607</td>
      <td>4.618089</td>
      <td>1.588098</td>
      <td>0.813838</td>
      <td>2.993985</td>
      <td>-2.867360</td>
      <td>-2.654889</td>
      <td>0.857129</td>
      <td>-0.071120</td>
      <td>2.850421</td>
      <td>1.123050</td>
      <td>0.415738</td>
      <td>0.002754</td>
      <td>0.004139</td>
      <td>0.007933</td>
      <td>0.391374</td>
      <td>0.943303</td>
      <td>0.004366</td>
      <td>0.261416</td>
      <td>2.495832</td>
      <td>1.971833</td>
      <td>0.802286</td>
      <td>0.309171</td>
      <td>0.202179</td>
      <td>80.276682</td>
      <td>113.293759</td>
      <td>113.308796</td>
      <td>0.569744</td>
      <td>131.026908</td>
      <td>2.119431</td>
      <td>2.774556</td>
      <td>1.076470</td>
      <td>2.013046</td>
      <td>2.234159</td>
      <td>1.823712</td>
      <td>0.12195</td>
      <td>0.19434</td>
      <td>0.06109</td>
      <td>36.30828</td>
      <td>31.43311</td>
      <td>35.43090</td>
      <td>0.292378</td>
      <td>250.743424</td>
      <td>-1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>-3</td>
      <td>2</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>10</th>
      <td>181</td>
      <td>194</td>
      <td>p1</td>
      <td>bo</td>
      <td>3.56635</td>
      <td>1.29492</td>
      <td>0.94415</td>
      <td>372.31445</td>
      <td>453.79639</td>
      <td>499.99237</td>
      <td>2.010966</td>
      <td>345.795432</td>
      <td>-2.00000</td>
      <td>0.70502</td>
      <td>-0.89270</td>
      <td>-250.00000</td>
      <td>-203.80402</td>
      <td>-250.00000</td>
      <td>0.954952</td>
      <td>8.094307</td>
      <td>1.56635</td>
      <td>1.99994</td>
      <td>0.05145</td>
      <td>122.31445</td>
      <td>249.99237</td>
      <td>249.99237</td>
      <td>2.965919</td>
      <td>353.889739</td>
      <td>-0.470951</td>
      <td>1.087688</td>
      <td>-0.323219</td>
      <td>-15.745309</td>
      <td>0.764113</td>
      <td>6.171592</td>
      <td>1.406763</td>
      <td>129.865801</td>
      <td>0.711928</td>
      <td>1.087688</td>
      <td>0.331135</td>
      <td>46.916668</td>
      <td>77.717708</td>
      <td>68.720891</td>
      <td>1.406763</td>
      <td>129.865801</td>
      <td>1.729265</td>
      <td>2.603602</td>
      <td>0.898657</td>
      <td>3.216404</td>
      <td>0.304534</td>
      <td>1.032180</td>
      <td>1.538454</td>
      <td>-1.382180</td>
      <td>4.729265</td>
      <td>5.603602</td>
      <td>3.898657</td>
      <td>6.216404</td>
      <td>3.304534</td>
      <td>4.032180</td>
      <td>4.538454</td>
      <td>1.617820</td>
      <td>0.719170</td>
      <td>2.823384</td>
      <td>-1.836477</td>
      <td>-2.636197</td>
      <td>0.848982</td>
      <td>0.016422</td>
      <td>2.805518</td>
      <td>1.155694</td>
      <td>0.472036</td>
      <td>0.004752</td>
      <td>0.066287</td>
      <td>0.008384</td>
      <td>0.395892</td>
      <td>0.986897</td>
      <td>0.005024</td>
      <td>0.247806</td>
      <td>2.493110</td>
      <td>2.088702</td>
      <td>0.805256</td>
      <td>0.317095</td>
      <td>0.229966</td>
      <td>80.561859</td>
      <td>114.393538</td>
      <td>113.488718</td>
      <td>0.576384</td>
      <td>126.004940</td>
      <td>2.107430</td>
      <td>2.663364</td>
      <td>1.092721</td>
      <td>2.018825</td>
      <td>2.223631</td>
      <td>1.824295</td>
      <td>0.13086</td>
      <td>0.19080</td>
      <td>0.13196</td>
      <td>38.43689</td>
      <td>56.48804</td>
      <td>41.80145</td>
      <td>0.292378</td>
      <td>226.301875</td>
      <td>-1</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>-3</td>
      <td>2</td>
      <td>-7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>198</td>
      <td>211</td>
      <td>p1</td>
      <td>bo</td>
      <td>2.51343</td>
      <td>1.97119</td>
      <td>1.67291</td>
      <td>304.14581</td>
      <td>468.03284</td>
      <td>384.50623</td>
      <td>1.690211</td>
      <td>391.224523</td>
      <td>-1.63672</td>
      <td>-0.09686</td>
      <td>-1.49200</td>
      <td>-240.49377</td>
      <td>-218.04047</td>
      <td>-250.00000</td>
      <td>0.860827</td>
      <td>4.914335</td>
      <td>0.87671</td>
      <td>1.87433</td>
      <td>0.18091</td>
      <td>63.65204</td>
      <td>249.99237</td>
      <td>134.50623</td>
      <td>2.551038</td>
      <td>396.138858</td>
      <td>-0.428498</td>
      <td>0.804941</td>
      <td>-0.332857</td>
      <td>-9.170531</td>
      <td>34.309975</td>
      <td>-17.510633</td>
      <td>1.218461</td>
      <td>133.943226</td>
      <td>0.674414</td>
      <td>0.819842</td>
      <td>0.360689</td>
      <td>47.677846</td>
      <td>92.343844</td>
      <td>73.083144</td>
      <td>1.218461</td>
      <td>133.943226</td>
      <td>0.313165</td>
      <td>0.937511</td>
      <td>4.204529</td>
      <td>3.719118</td>
      <td>-0.351555</td>
      <td>0.152233</td>
      <td>3.299575</td>
      <td>-0.624820</td>
      <td>3.313165</td>
      <td>3.937511</td>
      <td>7.204529</td>
      <td>6.719118</td>
      <td>2.648445</td>
      <td>3.152233</td>
      <td>6.299575</td>
      <td>2.375180</td>
      <td>0.990458</td>
      <td>0.595832</td>
      <td>-3.261396</td>
      <td>-3.326627</td>
      <td>0.327112</td>
      <td>-1.431505</td>
      <td>3.360394</td>
      <td>1.526698</td>
      <td>0.321950</td>
      <td>0.551287</td>
      <td>0.001109</td>
      <td>0.000879</td>
      <td>0.743583</td>
      <td>0.152286</td>
      <td>0.000778</td>
      <td>0.126836</td>
      <td>2.508546</td>
      <td>2.102299</td>
      <td>0.632798</td>
      <td>0.447991</td>
      <td>0.376821</td>
      <td>76.151350</td>
      <td>127.122972</td>
      <td>104.123765</td>
      <td>0.448120</td>
      <td>128.172321</td>
      <td>1.743707</td>
      <td>2.420271</td>
      <td>1.364796</td>
      <td>2.316862</td>
      <td>2.779894</td>
      <td>0.915307</td>
      <td>0.45111</td>
      <td>0.35480</td>
      <td>0.12891</td>
      <td>51.63574</td>
      <td>152.19116</td>
      <td>95.42083</td>
      <td>0.151963</td>
      <td>164.004672</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-3</td>
      <td>7</td>
      <td>-2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>12</th>
      <td>220</td>
      <td>233</td>
      <td>p1</td>
      <td>bo</td>
      <td>1.25909</td>
      <td>2.03558</td>
      <td>2.27307</td>
      <td>285.00366</td>
      <td>499.99237</td>
      <td>478.96576</td>
      <td>2.225653</td>
      <td>368.202819</td>
      <td>-1.47284</td>
      <td>-0.03564</td>
      <td>-0.87085</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>0.422914</td>
      <td>3.502292</td>
      <td>-0.21375</td>
      <td>1.99994</td>
      <td>1.40222</td>
      <td>35.00366</td>
      <td>249.99237</td>
      <td>228.96576</td>
      <td>2.648567</td>
      <td>371.705111</td>
      <td>-0.587655</td>
      <td>0.867821</td>
      <td>-0.162278</td>
      <td>-62.545776</td>
      <td>24.615948</td>
      <td>-4.169758</td>
      <td>1.261102</td>
      <td>143.617799</td>
      <td>0.587655</td>
      <td>0.874008</td>
      <td>0.408277</td>
      <td>69.711539</td>
      <td>86.473905</td>
      <td>70.722726</td>
      <td>1.261102</td>
      <td>143.617799</td>
      <td>-0.078696</td>
      <td>-0.436852</td>
      <td>3.678786</td>
      <td>-0.253034</td>
      <td>0.373417</td>
      <td>0.827555</td>
      <td>1.196396</td>
      <td>-1.494619</td>
      <td>2.921304</td>
      <td>2.563148</td>
      <td>6.678786</td>
      <td>2.746966</td>
      <td>3.373417</td>
      <td>3.827555</td>
      <td>4.196396</td>
      <td>1.505381</td>
      <td>-2.117945</td>
      <td>0.398716</td>
      <td>3.066346</td>
      <td>-1.978430</td>
      <td>-0.284171</td>
      <td>-0.482675</td>
      <td>2.024749</td>
      <td>0.902144</td>
      <td>0.034180</td>
      <td>0.690103</td>
      <td>0.002167</td>
      <td>0.047880</td>
      <td>0.776279</td>
      <td>0.629326</td>
      <td>0.042893</td>
      <td>0.366981</td>
      <td>2.481170</td>
      <td>2.029689</td>
      <td>0.392569</td>
      <td>0.571005</td>
      <td>0.519502</td>
      <td>93.462874</td>
      <td>124.196074</td>
      <td>109.579307</td>
      <td>0.533335</td>
      <td>141.692737</td>
      <td>2.225393</td>
      <td>1.573346</td>
      <td>1.407194</td>
      <td>1.514596</td>
      <td>2.341863</td>
      <td>1.327714</td>
      <td>0.39386</td>
      <td>0.53326</td>
      <td>0.22229</td>
      <td>92.03338</td>
      <td>67.97790</td>
      <td>41.88538</td>
      <td>0.455386</td>
      <td>257.628909</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>-2</td>
      <td>3</td>
      <td>-3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>13</th>
      <td>224</td>
      <td>237</td>
      <td>p1</td>
      <td>bo</td>
      <td>1.30457</td>
      <td>2.03558</td>
      <td>2.27307</td>
      <td>287.28485</td>
      <td>499.99237</td>
      <td>478.96576</td>
      <td>2.225653</td>
      <td>350.502525</td>
      <td>-1.47284</td>
      <td>-0.03564</td>
      <td>-0.87085</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>0.422914</td>
      <td>21.202586</td>
      <td>-0.16827</td>
      <td>1.99994</td>
      <td>1.40222</td>
      <td>37.28485</td>
      <td>249.99237</td>
      <td>228.96576</td>
      <td>2.648567</td>
      <td>371.705111</td>
      <td>-0.582041</td>
      <td>0.882992</td>
      <td>-0.135673</td>
      <td>-61.276950</td>
      <td>19.867531</td>
      <td>-2.816420</td>
      <td>1.271103</td>
      <td>152.384039</td>
      <td>0.582041</td>
      <td>0.889179</td>
      <td>0.381672</td>
      <td>73.386559</td>
      <td>91.546278</td>
      <td>75.431823</td>
      <td>1.271103</td>
      <td>152.384039</td>
      <td>-0.135896</td>
      <td>-0.552694</td>
      <td>3.298558</td>
      <td>-0.310121</td>
      <td>0.182221</td>
      <td>0.748211</td>
      <td>1.145751</td>
      <td>-1.458327</td>
      <td>2.864104</td>
      <td>2.447306</td>
      <td>6.298558</td>
      <td>2.689879</td>
      <td>3.182221</td>
      <td>3.748211</td>
      <td>4.145751</td>
      <td>1.541673</td>
      <td>-2.018564</td>
      <td>0.256044</td>
      <td>2.860551</td>
      <td>-1.872662</td>
      <td>-0.088620</td>
      <td>-0.539041</td>
      <td>1.940487</td>
      <td>0.965120</td>
      <td>0.043533</td>
      <td>0.797917</td>
      <td>0.004229</td>
      <td>0.061115</td>
      <td>0.929384</td>
      <td>0.589858</td>
      <td>0.052321</td>
      <td>0.334485</td>
      <td>2.482104</td>
      <td>2.166237</td>
      <td>0.398969</td>
      <td>0.576920</td>
      <td>0.519906</td>
      <td>95.041905</td>
      <td>125.975853</td>
      <td>110.257302</td>
      <td>0.532415</td>
      <td>134.149812</td>
      <td>2.215267</td>
      <td>1.530043</td>
      <td>1.424759</td>
      <td>1.525988</td>
      <td>2.340089</td>
      <td>1.324315</td>
      <td>0.38458</td>
      <td>0.76318</td>
      <td>0.35120</td>
      <td>94.29931</td>
      <td>96.86279</td>
      <td>57.02209</td>
      <td>0.462991</td>
      <td>242.629763</td>
      <td>10</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>-2</td>
      <td>3</td>
      <td>-3</td>
      <td>-7</td>
    </tr>
    <tr>
      <th>14</th>
      <td>6</td>
      <td>19</td>
      <td>p1</td>
      <td>bu</td>
      <td>2.83838</td>
      <td>1.62488</td>
      <td>0.72870</td>
      <td>129.74548</td>
      <td>374.09973</td>
      <td>289.65759</td>
      <td>1.975030</td>
      <td>277.323635</td>
      <td>-0.83844</td>
      <td>0.37506</td>
      <td>-0.84973</td>
      <td>-69.89288</td>
      <td>-250.00000</td>
      <td>-117.43927</td>
      <td>0.890068</td>
      <td>5.296810</td>
      <td>1.99994</td>
      <td>1.99994</td>
      <td>-0.12103</td>
      <td>59.85260</td>
      <td>124.09973</td>
      <td>172.21832</td>
      <td>2.865099</td>
      <td>282.620445</td>
      <td>-0.190688</td>
      <td>0.940078</td>
      <td>-0.412940</td>
      <td>-13.863196</td>
      <td>-10.161181</td>
      <td>-10.879518</td>
      <td>1.202327</td>
      <td>85.711041</td>
      <td>0.500371</td>
      <td>0.940078</td>
      <td>0.412940</td>
      <td>23.748544</td>
      <td>67.183273</td>
      <td>38.456844</td>
      <td>1.202327</td>
      <td>85.711041</td>
      <td>5.895584</td>
      <td>4.240339</td>
      <td>-0.188728</td>
      <td>1.121040</td>
      <td>0.665812</td>
      <td>3.729177</td>
      <td>6.186358</td>
      <td>0.163970</td>
      <td>8.895584</td>
      <td>7.240339</td>
      <td>2.811272</td>
      <td>4.121040</td>
      <td>3.665812</td>
      <td>6.729177</td>
      <td>9.186358</td>
      <td>3.163970</td>
      <td>3.884461</td>
      <td>3.012904</td>
      <td>-1.537687</td>
      <td>1.049760</td>
      <td>-2.074572</td>
      <td>2.723486</td>
      <td>4.079975</td>
      <td>2.185225</td>
      <td>0.000103</td>
      <td>0.002588</td>
      <td>0.124125</td>
      <td>0.293828</td>
      <td>0.038026</td>
      <td>0.006460</td>
      <td>0.000045</td>
      <td>0.028872</td>
      <td>2.497571</td>
      <td>2.103990</td>
      <td>0.670342</td>
      <td>0.347401</td>
      <td>0.196285</td>
      <td>29.573978</td>
      <td>99.465316</td>
      <td>61.569303</td>
      <td>0.503600</td>
      <td>87.327931</td>
      <td>1.544614</td>
      <td>2.549666</td>
      <td>1.296389</td>
      <td>2.171408</td>
      <td>1.664455</td>
      <td>1.867364</td>
      <td>0.12634</td>
      <td>0.12054</td>
      <td>0.21649</td>
      <td>29.60205</td>
      <td>23.21624</td>
      <td>38.56659</td>
      <td>0.074686</td>
      <td>111.952505</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>27</td>
      <td>40</td>
      <td>p1</td>
      <td>bu</td>
      <td>1.19024</td>
      <td>1.02942</td>
      <td>1.03778</td>
      <td>109.33686</td>
      <td>255.51605</td>
      <td>247.05506</td>
      <td>1.179858</td>
      <td>184.951966</td>
      <td>-1.33630</td>
      <td>0.42450</td>
      <td>-1.05432</td>
      <td>-95.92438</td>
      <td>-158.35571</td>
      <td>-147.97211</td>
      <td>0.839582</td>
      <td>19.228326</td>
      <td>-0.14606</td>
      <td>1.45392</td>
      <td>-0.01654</td>
      <td>13.41248</td>
      <td>97.16034</td>
      <td>99.08295</td>
      <td>2.019441</td>
      <td>204.180292</td>
      <td>-0.493324</td>
      <td>0.865549</td>
      <td>-0.340352</td>
      <td>-21.241408</td>
      <td>11.749855</td>
      <td>-17.678482</td>
      <td>1.112244</td>
      <td>91.201049</td>
      <td>0.493324</td>
      <td>0.865549</td>
      <td>0.340352</td>
      <td>29.963567</td>
      <td>58.066148</td>
      <td>52.362883</td>
      <td>1.112244</td>
      <td>91.201049</td>
      <td>1.264483</td>
      <td>0.180113</td>
      <td>3.615829</td>
      <td>-0.781972</td>
      <td>0.353090</td>
      <td>-0.421954</td>
      <td>2.955697</td>
      <td>-0.734706</td>
      <td>4.264483</td>
      <td>3.180113</td>
      <td>6.615829</td>
      <td>2.218028</td>
      <td>3.353090</td>
      <td>2.578046</td>
      <td>5.955697</td>
      <td>2.265294</td>
      <td>-2.373976</td>
      <td>0.423154</td>
      <td>-3.138118</td>
      <td>-1.622602</td>
      <td>-1.895390</td>
      <td>-0.227950</td>
      <td>3.163111</td>
      <td>0.826671</td>
      <td>0.017598</td>
      <td>0.672183</td>
      <td>0.001700</td>
      <td>0.104675</td>
      <td>0.058041</td>
      <td>0.819685</td>
      <td>0.001561</td>
      <td>0.408424</td>
      <td>2.530882</td>
      <td>2.366122</td>
      <td>0.317567</td>
      <td>0.260116</td>
      <td>0.237693</td>
      <td>37.921420</td>
      <td>70.914630</td>
      <td>64.070250</td>
      <td>0.310328</td>
      <td>56.178907</td>
      <td>2.460652</td>
      <td>2.525231</td>
      <td>0.516732</td>
      <td>1.544973</td>
      <td>2.293368</td>
      <td>1.165482</td>
      <td>0.24158</td>
      <td>0.20850</td>
      <td>0.10907</td>
      <td>45.81451</td>
      <td>79.84924</td>
      <td>82.60346</td>
      <td>0.201497</td>
      <td>81.354033</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>-2</td>
      <td>-8</td>
    </tr>
    <tr>
      <th>16</th>
      <td>41</td>
      <td>54</td>
      <td>p1</td>
      <td>bu</td>
      <td>0.98639</td>
      <td>1.44342</td>
      <td>0.88568</td>
      <td>220.39032</td>
      <td>467.04865</td>
      <td>472.92328</td>
      <td>1.478319</td>
      <td>395.180834</td>
      <td>-1.23682</td>
      <td>0.31818</td>
      <td>-0.95514</td>
      <td>-193.55011</td>
      <td>-250.00000</td>
      <td>-222.93091</td>
      <td>0.864521</td>
      <td>7.879744</td>
      <td>-0.25043</td>
      <td>1.76160</td>
      <td>-0.06946</td>
      <td>26.84021</td>
      <td>217.04865</td>
      <td>249.99237</td>
      <td>2.342840</td>
      <td>403.060579</td>
      <td>-0.502747</td>
      <td>0.861550</td>
      <td>-0.383718</td>
      <td>-27.024488</td>
      <td>9.036725</td>
      <td>-17.564627</td>
      <td>1.136771</td>
      <td>104.256071</td>
      <td>0.502747</td>
      <td>0.861550</td>
      <td>0.383718</td>
      <td>32.327505</td>
      <td>63.766478</td>
      <td>69.186284</td>
      <td>1.136771</td>
      <td>104.256071</td>
      <td>1.354099</td>
      <td>1.028044</td>
      <td>-0.128428</td>
      <td>4.538247</td>
      <td>1.982310</td>
      <td>2.214743</td>
      <td>6.971485</td>
      <td>1.448061</td>
      <td>4.354099</td>
      <td>4.028044</td>
      <td>2.871572</td>
      <td>7.538247</td>
      <td>4.982310</td>
      <td>5.214743</td>
      <td>9.971485</td>
      <td>4.448061</td>
      <td>-2.559645</td>
      <td>1.222117</td>
      <td>-1.833782</td>
      <td>-3.582987</td>
      <td>-1.117853</td>
      <td>1.458864</td>
      <td>4.202143</td>
      <td>2.811519</td>
      <td>0.010478</td>
      <td>0.221663</td>
      <td>0.066686</td>
      <td>0.000340</td>
      <td>0.263630</td>
      <td>0.144602</td>
      <td>0.000026</td>
      <td>0.004931</td>
      <td>2.525213</td>
      <td>2.073890</td>
      <td>0.273154</td>
      <td>0.356648</td>
      <td>0.276137</td>
      <td>53.184480</td>
      <td>100.115731</td>
      <td>100.630076</td>
      <td>0.357878</td>
      <td>115.018842</td>
      <td>2.646187</td>
      <td>2.439647</td>
      <td>0.613590</td>
      <td>1.050661</td>
      <td>2.597176</td>
      <td>2.001150</td>
      <td>0.26136</td>
      <td>0.23462</td>
      <td>0.32617</td>
      <td>27.19879</td>
      <td>51.20086</td>
      <td>78.41492</td>
      <td>0.081260</td>
      <td>58.714430</td>
      <td>-2</td>
      <td>2</td>
      <td>7</td>
      <td>3</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>17</th>
      <td>94</td>
      <td>107</td>
      <td>p1</td>
      <td>bu</td>
      <td>1.01313</td>
      <td>1.24310</td>
      <td>0.84406</td>
      <td>133.22449</td>
      <td>499.99237</td>
      <td>499.99237</td>
      <td>0.909534</td>
      <td>350.703583</td>
      <td>-1.22327</td>
      <td>0.20081</td>
      <td>-0.72101</td>
      <td>-98.38867</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>0.684635</td>
      <td>3.318649</td>
      <td>-0.21014</td>
      <td>1.44391</td>
      <td>0.12305</td>
      <td>34.83582</td>
      <td>249.99237</td>
      <td>249.99237</td>
      <td>1.594169</td>
      <td>354.022232</td>
      <td>-0.472050</td>
      <td>0.804818</td>
      <td>-0.279255</td>
      <td>-11.753377</td>
      <td>7.370582</td>
      <td>-19.425025</td>
      <td>1.050239</td>
      <td>107.830667</td>
      <td>0.472050</td>
      <td>0.804818</td>
      <td>0.299772</td>
      <td>28.068545</td>
      <td>62.979478</td>
      <td>71.315472</td>
      <td>1.050239</td>
      <td>107.830667</td>
      <td>2.405747</td>
      <td>0.144638</td>
      <td>-0.279560</td>
      <td>0.107820</td>
      <td>2.192151</td>
      <td>1.910901</td>
      <td>0.351962</td>
      <td>0.744253</td>
      <td>5.405747</td>
      <td>3.144638</td>
      <td>2.720440</td>
      <td>3.107820</td>
      <td>5.192151</td>
      <td>4.910901</td>
      <td>3.351962</td>
      <td>3.744253</td>
      <td>-3.068572</td>
      <td>-0.645289</td>
      <td>-0.277007</td>
      <td>-1.638343</td>
      <td>-0.317496</td>
      <td>0.858976</td>
      <td>1.738480</td>
      <td>2.534900</td>
      <td>0.002151</td>
      <td>0.518740</td>
      <td>0.781775</td>
      <td>0.101350</td>
      <td>0.750867</td>
      <td>0.390354</td>
      <td>0.082126</td>
      <td>0.011248</td>
      <td>2.537687</td>
      <td>2.106316</td>
      <td>0.265641</td>
      <td>0.316477</td>
      <td>0.216581</td>
      <td>37.207783</td>
      <td>104.089150</td>
      <td>106.567139</td>
      <td>0.251349</td>
      <td>111.883295</td>
      <td>2.497930</td>
      <td>2.279947</td>
      <td>0.930601</td>
      <td>1.918005</td>
      <td>2.630402</td>
      <td>1.204576</td>
      <td>0.11548</td>
      <td>0.31512</td>
      <td>0.22486</td>
      <td>30.13611</td>
      <td>55.85480</td>
      <td>54.43573</td>
      <td>0.097955</td>
      <td>84.760137</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>18</th>
      <td>109</td>
      <td>122</td>
      <td>p1</td>
      <td>bu</td>
      <td>1.67248</td>
      <td>0.92810</td>
      <td>1.86017</td>
      <td>191.45966</td>
      <td>456.97021</td>
      <td>196.78497</td>
      <td>1.913505</td>
      <td>304.530904</td>
      <td>-0.73773</td>
      <td>0.34357</td>
      <td>-2.00000</td>
      <td>-153.87726</td>
      <td>-250.00000</td>
      <td>-95.63446</td>
      <td>0.634220</td>
      <td>5.968108</td>
      <td>0.93475</td>
      <td>1.27167</td>
      <td>-0.13983</td>
      <td>37.58240</td>
      <td>206.97021</td>
      <td>101.15051</td>
      <td>2.547725</td>
      <td>310.499013</td>
      <td>-0.232285</td>
      <td>0.861681</td>
      <td>-0.508981</td>
      <td>-24.596581</td>
      <td>10.732798</td>
      <td>-3.730187</td>
      <td>1.148257</td>
      <td>104.963170</td>
      <td>0.395248</td>
      <td>0.861681</td>
      <td>0.508981</td>
      <td>38.235592</td>
      <td>83.949162</td>
      <td>39.454533</td>
      <td>1.148257</td>
      <td>104.963170</td>
      <td>3.216258</td>
      <td>-0.081504</td>
      <td>3.271959</td>
      <td>0.473398</td>
      <td>0.244027</td>
      <td>-0.251205</td>
      <td>4.021182</td>
      <td>-0.255123</td>
      <td>6.216258</td>
      <td>2.918496</td>
      <td>6.271959</td>
      <td>3.473398</td>
      <td>3.244027</td>
      <td>2.748795</td>
      <td>7.021182</td>
      <td>2.744877</td>
      <td>3.008195</td>
      <td>-1.189444</td>
      <td>-3.440649</td>
      <td>-2.453688</td>
      <td>-0.978899</td>
      <td>0.272329</td>
      <td>3.404867</td>
      <td>1.749462</td>
      <td>0.002628</td>
      <td>0.234265</td>
      <td>0.000580</td>
      <td>0.014140</td>
      <td>0.327630</td>
      <td>0.785369</td>
      <td>0.000662</td>
      <td>0.080211</td>
      <td>2.499180</td>
      <td>2.179919</td>
      <td>0.394673</td>
      <td>0.246391</td>
      <td>0.504926</td>
      <td>60.036972</td>
      <td>113.706237</td>
      <td>49.814382</td>
      <td>0.456554</td>
      <td>93.444374</td>
      <td>1.932775</td>
      <td>2.399414</td>
      <td>1.576775</td>
      <td>1.667307</td>
      <td>2.247407</td>
      <td>1.928509</td>
      <td>0.23565</td>
      <td>0.19995</td>
      <td>0.10772</td>
      <td>20.62225</td>
      <td>81.84814</td>
      <td>67.84058</td>
      <td>0.218534</td>
      <td>115.708763</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>-1</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>19</th>
      <td>127</td>
      <td>140</td>
      <td>p1</td>
      <td>bu</td>
      <td>1.36139</td>
      <td>1.66010</td>
      <td>1.20764</td>
      <td>257.65991</td>
      <td>351.69983</td>
      <td>499.99237</td>
      <td>1.678036</td>
      <td>410.128796</td>
      <td>-1.47400</td>
      <td>0.33984</td>
      <td>-0.98706</td>
      <td>-220.25299</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>0.937523</td>
      <td>6.413597</td>
      <td>-0.11261</td>
      <td>1.99994</td>
      <td>0.22058</td>
      <td>37.40692</td>
      <td>101.69983</td>
      <td>249.99237</td>
      <td>2.615559</td>
      <td>416.542392</td>
      <td>-0.539870</td>
      <td>0.919575</td>
      <td>-0.333758</td>
      <td>-28.169485</td>
      <td>17.275883</td>
      <td>-19.951454</td>
      <td>1.181555</td>
      <td>108.169343</td>
      <td>0.539870</td>
      <td>0.919575</td>
      <td>0.377046</td>
      <td>39.289621</td>
      <td>67.174471</td>
      <td>63.624455</td>
      <td>1.181555</td>
      <td>108.169343</td>
      <td>2.333900</td>
      <td>4.595520</td>
      <td>0.068517</td>
      <td>3.835385</td>
      <td>3.031313</td>
      <td>2.857057</td>
      <td>7.239838</td>
      <td>2.107182</td>
      <td>5.333900</td>
      <td>7.595520</td>
      <td>3.068517</td>
      <td>6.835385</td>
      <td>6.031313</td>
      <td>5.857057</td>
      <td>10.239838</td>
      <td>5.107182</td>
      <td>-2.759364</td>
      <td>3.132534</td>
      <td>-0.723453</td>
      <td>-3.321813</td>
      <td>-3.018246</td>
      <td>1.034702</td>
      <td>4.288650</td>
      <td>2.882213</td>
      <td>0.005791</td>
      <td>0.001733</td>
      <td>0.469402</td>
      <td>0.000894</td>
      <td>0.002542</td>
      <td>0.300808</td>
      <td>0.000018</td>
      <td>0.003949</td>
      <td>2.515331</td>
      <td>2.126621</td>
      <td>0.329638</td>
      <td>0.349742</td>
      <td>0.308313</td>
      <td>63.011914</td>
      <td>90.575870</td>
      <td>101.627926</td>
      <td>0.422419</td>
      <td>110.869990</td>
      <td>2.813538</td>
      <td>2.371099</td>
      <td>0.803824</td>
      <td>1.020586</td>
      <td>2.552127</td>
      <td>2.237704</td>
      <td>0.24652</td>
      <td>0.10486</td>
      <td>0.22857</td>
      <td>22.11762</td>
      <td>75.17243</td>
      <td>49.33167</td>
      <td>0.115055</td>
      <td>80.228567</td>
      <td>-2</td>
      <td>2</td>
      <td>5</td>
      <td>-1</td>
      <td>-2</td>
      <td>1</td>
      <td>-1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>20</th>
      <td>143</td>
      <td>156</td>
      <td>p1</td>
      <td>bu</td>
      <td>1.79211</td>
      <td>1.15448</td>
      <td>1.57971</td>
      <td>272.41516</td>
      <td>380.34058</td>
      <td>309.41009</td>
      <td>1.330794</td>
      <td>384.469683</td>
      <td>-2.00000</td>
      <td>0.31085</td>
      <td>-0.87402</td>
      <td>-107.68890</td>
      <td>-250.00000</td>
      <td>-59.41772</td>
      <td>0.906118</td>
      <td>5.569964</td>
      <td>-0.20789</td>
      <td>1.46533</td>
      <td>0.70569</td>
      <td>164.72626</td>
      <td>130.34058</td>
      <td>249.99237</td>
      <td>2.236912</td>
      <td>390.039646</td>
      <td>-0.549137</td>
      <td>0.895558</td>
      <td>-0.255569</td>
      <td>3.606357</td>
      <td>-19.166799</td>
      <td>22.578313</td>
      <td>1.194670</td>
      <td>117.960999</td>
      <td>0.549137</td>
      <td>0.895558</td>
      <td>0.364137</td>
      <td>39.110037</td>
      <td>80.734841</td>
      <td>67.138672</td>
      <td>1.194670</td>
      <td>117.960999</td>
      <td>6.015779</td>
      <td>1.234769</td>
      <td>2.307069</td>
      <td>1.783079</td>
      <td>-0.052325</td>
      <td>0.417921</td>
      <td>3.259983</td>
      <td>0.753772</td>
      <td>9.015779</td>
      <td>4.234769</td>
      <td>5.307069</td>
      <td>4.783079</td>
      <td>2.947675</td>
      <td>3.417921</td>
      <td>6.259983</td>
      <td>3.753772</td>
      <td>-3.973622</td>
      <td>-0.304361</td>
      <td>2.025383</td>
      <td>1.859710</td>
      <td>-1.726743</td>
      <td>2.238305</td>
      <td>3.295675</td>
      <td>2.378608</td>
      <td>0.000071</td>
      <td>0.760853</td>
      <td>0.042828</td>
      <td>0.062927</td>
      <td>0.084214</td>
      <td>0.025201</td>
      <td>0.000982</td>
      <td>0.017378</td>
      <td>2.527773</td>
      <td>2.157112</td>
      <td>0.441717</td>
      <td>0.254454</td>
      <td>0.348594</td>
      <td>60.877990</td>
      <td>114.492504</td>
      <td>93.702146</td>
      <td>0.350761</td>
      <td>112.116503</td>
      <td>2.382216</td>
      <td>2.165170</td>
      <td>1.497764</td>
      <td>1.618500</td>
      <td>2.759396</td>
      <td>1.377127</td>
      <td>0.22541</td>
      <td>0.13385</td>
      <td>0.17994</td>
      <td>31.63147</td>
      <td>110.42786</td>
      <td>114.28833</td>
      <td>0.184357</td>
      <td>88.064667</td>
      <td>-1</td>
      <td>2</td>
      <td>1</td>
      <td>-1</td>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>21</th>
      <td>158</td>
      <td>171</td>
      <td>p1</td>
      <td>bu</td>
      <td>2.23883</td>
      <td>1.72992</td>
      <td>1.80194</td>
      <td>249.34387</td>
      <td>499.99237</td>
      <td>354.29382</td>
      <td>2.938068</td>
      <td>367.820792</td>
      <td>-0.51331</td>
      <td>0.27002</td>
      <td>-2.00000</td>
      <td>-123.39020</td>
      <td>-250.00000</td>
      <td>-104.30145</td>
      <td>0.375115</td>
      <td>4.298193</td>
      <td>1.72552</td>
      <td>1.99994</td>
      <td>-0.19806</td>
      <td>125.95367</td>
      <td>249.99237</td>
      <td>249.99237</td>
      <td>3.313183</td>
      <td>372.118985</td>
      <td>-0.146438</td>
      <td>0.943748</td>
      <td>-0.477267</td>
      <td>6.429232</td>
      <td>-14.166612</td>
      <td>23.701595</td>
      <td>1.180865</td>
      <td>138.488358</td>
      <td>0.436946</td>
      <td>0.943748</td>
      <td>0.477267</td>
      <td>46.455383</td>
      <td>105.986962</td>
      <td>63.572812</td>
      <td>1.180865</td>
      <td>138.488358</td>
      <td>6.108327</td>
      <td>3.252661</td>
      <td>6.063222</td>
      <td>-0.265768</td>
      <td>-0.540705</td>
      <td>0.792420</td>
      <td>6.361118</td>
      <td>-0.858851</td>
      <td>9.108327</td>
      <td>6.252661</td>
      <td>9.063222</td>
      <td>2.734232</td>
      <td>2.459295</td>
      <td>3.792420</td>
      <td>9.361118</td>
      <td>2.141149</td>
      <td>4.032136</td>
      <td>2.323963</td>
      <td>-4.017812</td>
      <td>0.407239</td>
      <td>-0.403617</td>
      <td>1.996202</td>
      <td>3.949701</td>
      <td>1.205421</td>
      <td>0.000055</td>
      <td>0.020127</td>
      <td>0.000059</td>
      <td>0.683832</td>
      <td>0.686494</td>
      <td>0.045912</td>
      <td>0.000078</td>
      <td>0.228041</td>
      <td>2.450373</td>
      <td>2.197500</td>
      <td>0.568082</td>
      <td>0.366265</td>
      <td>0.462724</td>
      <td>66.880586</td>
      <td>139.153807</td>
      <td>89.316770</td>
      <td>0.645313</td>
      <td>115.926910</td>
      <td>1.547310</td>
      <td>2.608756</td>
      <td>2.014400</td>
      <td>1.411347</td>
      <td>1.281856</td>
      <td>0.496909</td>
      <td>0.08582</td>
      <td>0.18164</td>
      <td>0.19433</td>
      <td>31.75354</td>
      <td>216.20178</td>
      <td>109.47419</td>
      <td>0.126908</td>
      <td>177.156019</td>
      <td>2</td>
      <td>-1</td>
      <td>5</td>
      <td>-1</td>
      <td>-1</td>
      <td>-3</td>
      <td>-1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>22</th>
      <td>175</td>
      <td>188</td>
      <td>p1</td>
      <td>bu</td>
      <td>0.31970</td>
      <td>1.77484</td>
      <td>0.63306</td>
      <td>439.31580</td>
      <td>385.76508</td>
      <td>366.48560</td>
      <td>1.621982</td>
      <td>427.562760</td>
      <td>-0.59009</td>
      <td>0.22510</td>
      <td>-0.59845</td>
      <td>-189.32343</td>
      <td>-250.00000</td>
      <td>-116.49323</td>
      <td>0.403293</td>
      <td>5.441132</td>
      <td>-0.27039</td>
      <td>1.99994</td>
      <td>0.03461</td>
      <td>249.99237</td>
      <td>135.76508</td>
      <td>249.99237</td>
      <td>2.025275</td>
      <td>433.003892</td>
      <td>-0.375344</td>
      <td>1.013278</td>
      <td>-0.302945</td>
      <td>-5.486708</td>
      <td>-20.020116</td>
      <td>19.396268</td>
      <td>1.149608</td>
      <td>115.305042</td>
      <td>0.375344</td>
      <td>1.013278</td>
      <td>0.308270</td>
      <td>62.308092</td>
      <td>63.756502</td>
      <td>66.447332</td>
      <td>1.149608</td>
      <td>115.305042</td>
      <td>0.383264</td>
      <td>0.697401</td>
      <td>-0.525659</td>
      <td>1.703731</td>
      <td>0.775724</td>
      <td>0.803316</td>
      <td>0.485607</td>
      <td>1.209209</td>
      <td>3.383264</td>
      <td>3.697401</td>
      <td>2.474341</td>
      <td>4.703731</td>
      <td>3.775724</td>
      <td>3.803316</td>
      <td>3.485607</td>
      <td>4.209209</td>
      <td>-1.977265</td>
      <td>1.109891</td>
      <td>0.239995</td>
      <td>1.456001</td>
      <td>-1.718338</td>
      <td>1.893231</td>
      <td>0.904217</td>
      <td>2.539108</td>
      <td>0.048012</td>
      <td>0.267046</td>
      <td>0.810334</td>
      <td>0.145392</td>
      <td>0.085735</td>
      <td>0.058327</td>
      <td>0.365880</td>
      <td>0.011114</td>
      <td>2.506439</td>
      <td>2.091412</td>
      <td>0.087919</td>
      <td>0.419282</td>
      <td>0.176068</td>
      <td>97.583724</td>
      <td>95.646970</td>
      <td>91.588627</td>
      <td>0.390282</td>
      <td>120.711533</td>
      <td>2.671239</td>
      <td>2.535780</td>
      <td>0.541273</td>
      <td>2.028181</td>
      <td>2.737275</td>
      <td>1.023089</td>
      <td>0.07324</td>
      <td>0.17908</td>
      <td>0.16443</td>
      <td>69.79370</td>
      <td>40.39001</td>
      <td>82.30591</td>
      <td>0.193781</td>
      <td>133.343073</td>
      <td>-3</td>
      <td>1</td>
      <td>3</td>
      <td>-1</td>
      <td>-2</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>23</th>
      <td>193</td>
      <td>206</td>
      <td>p1</td>
      <td>bu</td>
      <td>1.77783</td>
      <td>1.62512</td>
      <td>1.80914</td>
      <td>187.81280</td>
      <td>351.94397</td>
      <td>433.63190</td>
      <td>2.361311</td>
      <td>279.357497</td>
      <td>-2.00000</td>
      <td>0.37482</td>
      <td>-0.65283</td>
      <td>-151.28326</td>
      <td>-175.06409</td>
      <td>-183.63953</td>
      <td>0.694309</td>
      <td>3.476104</td>
      <td>-0.22217</td>
      <td>1.99994</td>
      <td>1.15631</td>
      <td>36.52954</td>
      <td>176.87988</td>
      <td>249.99237</td>
      <td>3.055620</td>
      <td>282.833601</td>
      <td>-0.655658</td>
      <td>0.882240</td>
      <td>-0.220628</td>
      <td>-27.384244</td>
      <td>-4.278329</td>
      <td>0.009976</td>
      <td>1.248883</td>
      <td>118.645910</td>
      <td>0.655658</td>
      <td>0.882240</td>
      <td>0.398522</td>
      <td>38.619407</td>
      <td>69.894643</td>
      <td>72.655898</td>
      <td>1.248883</td>
      <td>118.645910</td>
      <td>0.305469</td>
      <td>3.834116</td>
      <td>5.533806</td>
      <td>0.065606</td>
      <td>-0.311130</td>
      <td>0.489955</td>
      <td>4.202873</td>
      <td>-1.481763</td>
      <td>3.305469</td>
      <td>6.834116</td>
      <td>8.533806</td>
      <td>3.065606</td>
      <td>2.688870</td>
      <td>3.489955</td>
      <td>7.202873</td>
      <td>1.518237</td>
      <td>-2.438301</td>
      <td>3.034644</td>
      <td>3.753537</td>
      <td>-1.849514</td>
      <td>0.441670</td>
      <td>1.056420</td>
      <td>3.549183</td>
      <td>0.619309</td>
      <td>0.014756</td>
      <td>0.002408</td>
      <td>0.000174</td>
      <td>0.064384</td>
      <td>0.658728</td>
      <td>0.290776</td>
      <td>0.000386</td>
      <td>0.535713</td>
      <td>2.477697</td>
      <td>2.152701</td>
      <td>0.570925</td>
      <td>0.369241</td>
      <td>0.426245</td>
      <td>52.136859</td>
      <td>96.220153</td>
      <td>107.262139</td>
      <td>0.584107</td>
      <td>100.861714</td>
      <td>2.525793</td>
      <td>1.717504</td>
      <td>1.676154</td>
      <td>2.140975</td>
      <td>2.558219</td>
      <td>0.935597</td>
      <td>0.23627</td>
      <td>0.19476</td>
      <td>0.15539</td>
      <td>59.14307</td>
      <td>59.33380</td>
      <td>65.74249</td>
      <td>0.135363</td>
      <td>193.664508</td>
      <td>3</td>
      <td>1</td>
      <td>-2</td>
      <td>2</td>
      <td>-3</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>24</th>
      <td>215</td>
      <td>228</td>
      <td>p1</td>
      <td>bu</td>
      <td>0.57874</td>
      <td>1.45703</td>
      <td>0.95129</td>
      <td>133.63647</td>
      <td>386.41357</td>
      <td>366.82892</td>
      <td>1.346827</td>
      <td>357.711884</td>
      <td>-0.68610</td>
      <td>0.37848</td>
      <td>-1.08191</td>
      <td>-67.35229</td>
      <td>-250.00000</td>
      <td>-116.83655</td>
      <td>0.679564</td>
      <td>2.194376</td>
      <td>-0.10736</td>
      <td>1.83551</td>
      <td>-0.13062</td>
      <td>66.28418</td>
      <td>136.41357</td>
      <td>249.99237</td>
      <td>2.026391</td>
      <td>359.906260</td>
      <td>-0.412565</td>
      <td>0.944298</td>
      <td>-0.416166</td>
      <td>-0.225360</td>
      <td>-22.365864</td>
      <td>2.085172</td>
      <td>1.153756</td>
      <td>104.390302</td>
      <td>0.412565</td>
      <td>0.944298</td>
      <td>0.416166</td>
      <td>27.502206</td>
      <td>75.083805</td>
      <td>49.861615</td>
      <td>1.153756</td>
      <td>104.390302</td>
      <td>0.715153</td>
      <td>2.131568</td>
      <td>2.274484</td>
      <td>-0.529862</td>
      <td>0.249922</td>
      <td>3.428587</td>
      <td>3.922429</td>
      <td>0.913743</td>
      <td>3.715153</td>
      <td>5.131568</td>
      <td>5.274484</td>
      <td>2.470138</td>
      <td>3.249922</td>
      <td>6.428587</td>
      <td>6.922429</td>
      <td>3.913743</td>
      <td>0.394340</td>
      <td>1.902652</td>
      <td>-2.736148</td>
      <td>0.082269</td>
      <td>-1.798824</td>
      <td>2.918947</td>
      <td>2.938728</td>
      <td>2.304779</td>
      <td>0.693330</td>
      <td>0.057086</td>
      <td>0.006216</td>
      <td>0.934433</td>
      <td>0.072047</td>
      <td>0.003512</td>
      <td>0.003296</td>
      <td>0.021179</td>
      <td>2.536356</td>
      <td>2.134695</td>
      <td>0.133150</td>
      <td>0.326705</td>
      <td>0.235337</td>
      <td>36.028174</td>
      <td>110.445815</td>
      <td>83.360912</td>
      <td>0.289480</td>
      <td>100.262876</td>
      <td>2.845714</td>
      <td>2.477341</td>
      <td>0.689828</td>
      <td>1.703270</td>
      <td>2.175225</td>
      <td>2.023713</td>
      <td>0.09937</td>
      <td>0.11493</td>
      <td>0.20269</td>
      <td>49.16382</td>
      <td>50.66681</td>
      <td>27.95410</td>
      <td>0.187772</td>
      <td>102.774229</td>
      <td>-1</td>
      <td>2</td>
      <td>5</td>
      <td>-1</td>
      <td>6</td>
      <td>-1</td>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>25</th>
      <td>232</td>
      <td>245</td>
      <td>p1</td>
      <td>bu</td>
      <td>2.82489</td>
      <td>1.04681</td>
      <td>1.85742</td>
      <td>166.09955</td>
      <td>499.99237</td>
      <td>333.86230</td>
      <td>2.244672</td>
      <td>354.618668</td>
      <td>-0.82495</td>
      <td>0.40918</td>
      <td>-2.00000</td>
      <td>-115.07416</td>
      <td>-250.00000</td>
      <td>-83.86993</td>
      <td>0.936470</td>
      <td>5.766383</td>
      <td>1.99994</td>
      <td>1.45599</td>
      <td>-0.14258</td>
      <td>51.02539</td>
      <td>249.99237</td>
      <td>249.99237</td>
      <td>3.181142</td>
      <td>360.385051</td>
      <td>-0.104482</td>
      <td>0.934396</td>
      <td>-0.511918</td>
      <td>-19.757785</td>
      <td>1.035837</td>
      <td>8.699272</td>
      <td>1.232884</td>
      <td>103.254619</td>
      <td>0.458402</td>
      <td>0.934396</td>
      <td>0.511918</td>
      <td>28.607883</td>
      <td>64.754192</td>
      <td>61.505244</td>
      <td>1.232884</td>
      <td>103.254619</td>
      <td>5.428680</td>
      <td>0.627955</td>
      <td>5.108362</td>
      <td>0.729538</td>
      <td>1.312481</td>
      <td>1.648406</td>
      <td>7.304737</td>
      <td>0.328488</td>
      <td>8.428680</td>
      <td>3.627955</td>
      <td>8.108362</td>
      <td>3.729538</td>
      <td>4.312481</td>
      <td>4.648406</td>
      <td>10.304737</td>
      <td>3.328488</td>
      <td>3.775883</td>
      <td>0.181020</td>
      <td>-3.794225</td>
      <td>-1.476673</td>
      <td>-0.395534</td>
      <td>2.539067</td>
      <td>4.305961</td>
      <td>2.150185</td>
      <td>0.000159</td>
      <td>0.856352</td>
      <td>0.000148</td>
      <td>0.139763</td>
      <td>0.692449</td>
      <td>0.011115</td>
      <td>0.000017</td>
      <td>0.031541</td>
      <td>2.485587</td>
      <td>2.066455</td>
      <td>0.653056</td>
      <td>0.242327</td>
      <td>0.465817</td>
      <td>39.292025</td>
      <td>110.917338</td>
      <td>88.411539</td>
      <td>0.572940</td>
      <td>107.092126</td>
      <td>1.542359</td>
      <td>2.511136</td>
      <td>2.076979</td>
      <td>1.869341</td>
      <td>1.328623</td>
      <td>1.610408</td>
      <td>0.14234</td>
      <td>0.05145</td>
      <td>0.13342</td>
      <td>43.85376</td>
      <td>12.37488</td>
      <td>71.78498</td>
      <td>0.210171</td>
      <td>109.304010</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>-2</td>
      <td>2</td>
      <td>-1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>26</th>
      <td>257</td>
      <td>270</td>
      <td>p1</td>
      <td>bu</td>
      <td>1.31628</td>
      <td>1.48029</td>
      <td>1.84235</td>
      <td>282.96661</td>
      <td>499.99237</td>
      <td>378.65448</td>
      <td>2.210545</td>
      <td>400.496100</td>
      <td>-1.20306</td>
      <td>0.51965</td>
      <td>-2.00000</td>
      <td>-88.21869</td>
      <td>-250.00000</td>
      <td>-128.66211</td>
      <td>0.834041</td>
      <td>3.136310</td>
      <td>0.11322</td>
      <td>1.99994</td>
      <td>-0.15765</td>
      <td>194.74792</td>
      <td>249.99237</td>
      <td>249.99237</td>
      <td>3.044586</td>
      <td>403.632410</td>
      <td>-0.489623</td>
      <td>0.975229</td>
      <td>-0.535977</td>
      <td>-4.815909</td>
      <td>16.652619</td>
      <td>6.009616</td>
      <td>1.270897</td>
      <td>110.391488</td>
      <td>0.507042</td>
      <td>0.975229</td>
      <td>0.535977</td>
      <td>35.137469</td>
      <td>71.661728</td>
      <td>64.818162</td>
      <td>1.270897</td>
      <td>110.391488</td>
      <td>-0.068062</td>
      <td>2.065421</td>
      <td>3.636229</td>
      <td>4.844037</td>
      <td>1.288745</td>
      <td>0.973489</td>
      <td>3.388989</td>
      <td>0.602103</td>
      <td>2.931938</td>
      <td>5.065421</td>
      <td>6.636229</td>
      <td>7.844037</td>
      <td>4.288745</td>
      <td>3.973489</td>
      <td>6.388989</td>
      <td>3.602103</td>
      <td>-1.342457</td>
      <td>2.607500</td>
      <td>-3.384992</td>
      <td>3.425153</td>
      <td>-0.536302</td>
      <td>2.114303</td>
      <td>3.478801</td>
      <td>2.283436</td>
      <td>0.179448</td>
      <td>0.009121</td>
      <td>0.000712</td>
      <td>0.000614</td>
      <td>0.591750</td>
      <td>0.034489</td>
      <td>0.000504</td>
      <td>0.022405</td>
      <td>2.477440</td>
      <td>2.050364</td>
      <td>0.350506</td>
      <td>0.370552</td>
      <td>0.483278</td>
      <td>63.428714</td>
      <td>111.012435</td>
      <td>96.913327</td>
      <td>0.597173</td>
      <td>117.853694</td>
      <td>2.704550</td>
      <td>2.639353</td>
      <td>0.558700</td>
      <td>0.998742</td>
      <td>1.407963</td>
      <td>0.568384</td>
      <td>0.26709</td>
      <td>0.12177</td>
      <td>0.32349</td>
      <td>19.81353</td>
      <td>54.00848</td>
      <td>41.00037</td>
      <td>0.111128</td>
      <td>151.575050</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>-3</td>
      <td>-2</td>
      <td>-1</td>
      <td>-4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>27</th>
      <td>259</td>
      <td>272</td>
      <td>p1</td>
      <td>bu</td>
      <td>1.31628</td>
      <td>1.48029</td>
      <td>1.94812</td>
      <td>282.96661</td>
      <td>499.99237</td>
      <td>378.65448</td>
      <td>2.210545</td>
      <td>400.496100</td>
      <td>-1.20306</td>
      <td>0.51965</td>
      <td>-2.00000</td>
      <td>-88.21869</td>
      <td>-250.00000</td>
      <td>-128.66211</td>
      <td>0.834041</td>
      <td>3.136310</td>
      <td>0.11322</td>
      <td>1.99994</td>
      <td>-0.05188</td>
      <td>194.74792</td>
      <td>249.99237</td>
      <td>249.99237</td>
      <td>3.044586</td>
      <td>403.632410</td>
      <td>-0.490149</td>
      <td>0.969477</td>
      <td>-0.523127</td>
      <td>-3.308811</td>
      <td>14.986477</td>
      <td>4.581745</td>
      <td>1.267000</td>
      <td>113.370896</td>
      <td>0.507568</td>
      <td>0.969477</td>
      <td>0.523127</td>
      <td>38.766714</td>
      <td>71.304320</td>
      <td>66.246033</td>
      <td>1.267000</td>
      <td>113.370896</td>
      <td>-0.155340</td>
      <td>1.829198</td>
      <td>3.348583</td>
      <td>4.160170</td>
      <td>1.239291</td>
      <td>0.954881</td>
      <td>3.343725</td>
      <td>0.680561</td>
      <td>2.844660</td>
      <td>4.829198</td>
      <td>6.348583</td>
      <td>7.160170</td>
      <td>4.239291</td>
      <td>3.954881</td>
      <td>6.343725</td>
      <td>3.680561</td>
      <td>-1.257560</td>
      <td>2.541515</td>
      <td>-3.255189</td>
      <td>3.236905</td>
      <td>-0.448493</td>
      <td>2.138603</td>
      <td>3.461172</td>
      <td>2.313564</td>
      <td>0.208551</td>
      <td>0.011037</td>
      <td>0.001133</td>
      <td>0.001208</td>
      <td>0.653798</td>
      <td>0.032468</td>
      <td>0.000538</td>
      <td>0.020692</td>
      <td>2.476079</td>
      <td>2.094717</td>
      <td>0.353744</td>
      <td>0.376955</td>
      <td>0.493858</td>
      <td>64.499050</td>
      <td>111.236422</td>
      <td>97.482735</td>
      <td>0.599692</td>
      <td>115.931565</td>
      <td>2.685107</td>
      <td>2.606304</td>
      <td>0.567996</td>
      <td>0.987467</td>
      <td>1.399036</td>
      <td>0.586806</td>
      <td>0.34015</td>
      <td>0.30566</td>
      <td>0.32953</td>
      <td>19.81353</td>
      <td>54.82482</td>
      <td>55.64118</td>
      <td>0.111128</td>
      <td>141.900320</td>
      <td>-1</td>
      <td>-1</td>
      <td>7</td>
      <td>-3</td>
      <td>-2</td>
      <td>-1</td>
      <td>-4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>28</th>
      <td>272</td>
      <td>285</td>
      <td>p1</td>
      <td>bu</td>
      <td>2.83063</td>
      <td>1.87348</td>
      <td>1.46557</td>
      <td>179.11530</td>
      <td>409.20258</td>
      <td>278.47290</td>
      <td>2.200381</td>
      <td>300.839130</td>
      <td>-0.83069</td>
      <td>0.12646</td>
      <td>-0.76898</td>
      <td>-138.00812</td>
      <td>-250.00000</td>
      <td>-161.94153</td>
      <td>0.712480</td>
      <td>27.446264</td>
      <td>1.99994</td>
      <td>1.99994</td>
      <td>0.69659</td>
      <td>41.10718</td>
      <td>159.20258</td>
      <td>116.53137</td>
      <td>2.912861</td>
      <td>328.285395</td>
      <td>-0.261841</td>
      <td>0.994164</td>
      <td>-0.248395</td>
      <td>-23.867682</td>
      <td>-36.149832</td>
      <td>-7.778462</td>
      <td>1.311581</td>
      <td>116.815329</td>
      <td>0.595835</td>
      <td>0.994164</td>
      <td>0.404514</td>
      <td>34.654471</td>
      <td>91.552148</td>
      <td>46.295165</td>
      <td>1.311581</td>
      <td>116.815329</td>
      <td>6.060024</td>
      <td>-0.754908</td>
      <td>0.400095</td>
      <td>1.225477</td>
      <td>-0.757504</td>
      <td>0.761712</td>
      <td>2.486950</td>
      <td>0.304292</td>
      <td>9.060024</td>
      <td>2.245092</td>
      <td>3.400095</td>
      <td>4.225477</td>
      <td>2.242496</td>
      <td>3.761712</td>
      <td>5.486950</td>
      <td>3.304292</td>
      <td>4.015009</td>
      <td>0.571838</td>
      <td>1.755966</td>
      <td>-2.070762</td>
      <td>-0.740259</td>
      <td>-0.941339</td>
      <td>2.984658</td>
      <td>1.999082</td>
      <td>0.000059</td>
      <td>0.567432</td>
      <td>0.079094</td>
      <td>0.038381</td>
      <td>0.459143</td>
      <td>0.346531</td>
      <td>0.002839</td>
      <td>0.045599</td>
      <td>2.487543</td>
      <td>2.309233</td>
      <td>0.687752</td>
      <td>0.537952</td>
      <td>0.390837</td>
      <td>43.623624</td>
      <td>114.969389</td>
      <td>64.829164</td>
      <td>0.559933</td>
      <td>87.263747</td>
      <td>1.624909</td>
      <td>1.771501</td>
      <td>0.616228</td>
      <td>1.230850</td>
      <td>1.650519</td>
      <td>0.959352</td>
      <td>0.15319</td>
      <td>0.80114</td>
      <td>0.41992</td>
      <td>28.19824</td>
      <td>162.82654</td>
      <td>47.78290</td>
      <td>0.439705</td>
      <td>108.855717</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>-3</td>
      <td>1</td>
      <td>-3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>29</th>
      <td>277</td>
      <td>290</td>
      <td>p1</td>
      <td>bu</td>
      <td>2.62232</td>
      <td>1.87348</td>
      <td>1.46557</td>
      <td>199.33319</td>
      <td>316.26129</td>
      <td>278.47290</td>
      <td>2.200381</td>
      <td>296.928051</td>
      <td>-0.62238</td>
      <td>0.12646</td>
      <td>-0.76898</td>
      <td>-138.00812</td>
      <td>-250.00000</td>
      <td>-161.94153</td>
      <td>0.712480</td>
      <td>31.357344</td>
      <td>1.99994</td>
      <td>1.99994</td>
      <td>0.69659</td>
      <td>61.32507</td>
      <td>66.26129</td>
      <td>116.53137</td>
      <td>2.912861</td>
      <td>328.285395</td>
      <td>-0.198242</td>
      <td>1.047738</td>
      <td>-0.236953</td>
      <td>-15.374405</td>
      <td>-47.748859</td>
      <td>-1.372705</td>
      <td>1.324408</td>
      <td>112.077155</td>
      <td>0.532236</td>
      <td>1.047738</td>
      <td>0.393072</td>
      <td>35.940904</td>
      <td>78.961298</td>
      <td>48.442547</td>
      <td>1.324408</td>
      <td>112.077155</td>
      <td>6.418382</td>
      <td>-0.642111</td>
      <td>0.165540</td>
      <td>0.657328</td>
      <td>-0.746401</td>
      <td>0.766772</td>
      <td>2.471995</td>
      <td>1.186003</td>
      <td>9.418382</td>
      <td>2.357889</td>
      <td>3.165540</td>
      <td>3.657328</td>
      <td>2.253599</td>
      <td>3.766772</td>
      <td>5.471995</td>
      <td>4.186003</td>
      <td>4.112072</td>
      <td>0.073646</td>
      <td>1.571890</td>
      <td>-1.661528</td>
      <td>-1.460373</td>
      <td>-1.145848</td>
      <td>2.946551</td>
      <td>2.611293</td>
      <td>0.000039</td>
      <td>0.941292</td>
      <td>0.115976</td>
      <td>0.096607</td>
      <td>0.144188</td>
      <td>0.251858</td>
      <td>0.003213</td>
      <td>0.009020</td>
      <td>2.489543</td>
      <td>2.335725</td>
      <td>0.661498</td>
      <td>0.526120</td>
      <td>0.394735</td>
      <td>49.541823</td>
      <td>100.926192</td>
      <td>65.669662</td>
      <td>0.556188</td>
      <td>83.125895</td>
      <td>1.592607</td>
      <td>1.767815</td>
      <td>0.646096</td>
      <td>1.180018</td>
      <td>1.423547</td>
      <td>0.806164</td>
      <td>0.18774</td>
      <td>0.54315</td>
      <td>0.41034</td>
      <td>39.89410</td>
      <td>145.43152</td>
      <td>68.44330</td>
      <td>0.495061</td>
      <td>63.098288</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>5</td>
      <td>1</td>
      <td>-3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>678</th>
      <td>114</td>
      <td>127</td>
      <td>p5</td>
      <td>fs</td>
      <td>1.57013</td>
      <td>2.61096</td>
      <td>2.64441</td>
      <td>284.26361</td>
      <td>365.71503</td>
      <td>499.99237</td>
      <td>2.894207</td>
      <td>328.264541</td>
      <td>-2.00000</td>
      <td>-0.61102</td>
      <td>-2.00000</td>
      <td>-181.41174</td>
      <td>-115.72266</td>
      <td>-250.00000</td>
      <td>0.569860</td>
      <td>7.355576</td>
      <td>-0.42987</td>
      <td>1.99994</td>
      <td>0.64441</td>
      <td>102.85187</td>
      <td>249.99237</td>
      <td>249.99237</td>
      <td>3.464067</td>
      <td>335.620117</td>
      <td>-0.957304</td>
      <td>0.581502</td>
      <td>-0.158582</td>
      <td>-15.894962</td>
      <td>31.961881</td>
      <td>-15.125568</td>
      <td>1.263470</td>
      <td>118.394751</td>
      <td>0.957304</td>
      <td>0.675505</td>
      <td>0.323425</td>
      <td>39.672852</td>
      <td>68.041876</td>
      <td>61.141382</td>
      <td>1.263470</td>
      <td>118.394751</td>
      <td>1.301469</td>
      <td>1.978425</td>
      <td>4.913504</td>
      <td>2.610011</td>
      <td>0.299516</td>
      <td>2.389842</td>
      <td>4.699013</td>
      <td>-1.029934</td>
      <td>4.301469</td>
      <td>4.978425</td>
      <td>7.913504</td>
      <td>5.610011</td>
      <td>3.299516</td>
      <td>5.389842</td>
      <td>7.699013</td>
      <td>1.970066</td>
      <td>-1.989507</td>
      <td>0.997107</td>
      <td>-3.492407</td>
      <td>-1.804048</td>
      <td>1.853850</td>
      <td>0.686930</td>
      <td>3.571204</td>
      <td>1.257432</td>
      <td>0.046645</td>
      <td>0.318713</td>
      <td>0.000479</td>
      <td>0.071224</td>
      <td>0.063761</td>
      <td>0.492127</td>
      <td>0.000355</td>
      <td>0.208597</td>
      <td>2.444443</td>
      <td>2.143415</td>
      <td>0.395260</td>
      <td>0.555151</td>
      <td>0.583655</td>
      <td>60.189378</td>
      <td>99.280161</td>
      <td>103.453341</td>
      <td>0.698887</td>
      <td>108.016586</td>
      <td>2.490142</td>
      <td>2.440977</td>
      <td>1.104466</td>
      <td>1.908562</td>
      <td>1.929555</td>
      <td>1.767362</td>
      <td>0.34368</td>
      <td>0.29181</td>
      <td>0.25757</td>
      <td>27.40478</td>
      <td>52.26899</td>
      <td>65.17029</td>
      <td>0.309480</td>
      <td>183.649596</td>
      <td>-7</td>
      <td>1</td>
      <td>-1</td>
      <td>-3</td>
      <td>-5</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>679</th>
      <td>125</td>
      <td>138</td>
      <td>p5</td>
      <td>fs</td>
      <td>1.44537</td>
      <td>2.77808</td>
      <td>0.75696</td>
      <td>339.14184</td>
      <td>399.77264</td>
      <td>499.99237</td>
      <td>2.184340</td>
      <td>342.582939</td>
      <td>-2.00000</td>
      <td>-0.77814</td>
      <td>-0.57916</td>
      <td>-126.22070</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>0.644823</td>
      <td>15.734065</td>
      <td>-0.55463</td>
      <td>1.99994</td>
      <td>0.17780</td>
      <td>212.92114</td>
      <td>149.77264</td>
      <td>249.99237</td>
      <td>2.829163</td>
      <td>358.317004</td>
      <td>-0.934392</td>
      <td>0.477468</td>
      <td>-0.136426</td>
      <td>10.587252</td>
      <td>-1.541725</td>
      <td>-8.979798</td>
      <td>1.170818</td>
      <td>108.236438</td>
      <td>0.934392</td>
      <td>0.597182</td>
      <td>0.208440</td>
      <td>43.481681</td>
      <td>61.897278</td>
      <td>59.700012</td>
      <td>1.170818</td>
      <td>108.236438</td>
      <td>2.400012</td>
      <td>3.191735</td>
      <td>-1.189200</td>
      <td>2.886885</td>
      <td>1.647381</td>
      <td>2.173737</td>
      <td>5.693542</td>
      <td>0.013984</td>
      <td>5.400012</td>
      <td>6.191735</td>
      <td>1.810800</td>
      <td>5.886885</td>
      <td>4.647381</td>
      <td>5.173737</td>
      <td>8.693542</td>
      <td>3.013984</td>
      <td>-2.797105</td>
      <td>1.282067</td>
      <td>-1.013829</td>
      <td>2.116892</td>
      <td>-1.734738</td>
      <td>0.268131</td>
      <td>3.850042</td>
      <td>2.155215</td>
      <td>0.005156</td>
      <td>0.199819</td>
      <td>0.310665</td>
      <td>0.034269</td>
      <td>0.082787</td>
      <td>0.788598</td>
      <td>0.000118</td>
      <td>0.031145</td>
      <td>2.489919</td>
      <td>2.067995</td>
      <td>0.374014</td>
      <td>0.559054</td>
      <td>0.243242</td>
      <td>71.842345</td>
      <td>93.320367</td>
      <td>104.563114</td>
      <td>0.510350</td>
      <td>115.253786</td>
      <td>2.501228</td>
      <td>1.718333</td>
      <td>1.133634</td>
      <td>1.946826</td>
      <td>2.153114</td>
      <td>2.024308</td>
      <td>0.29559</td>
      <td>0.20019</td>
      <td>0.40179</td>
      <td>35.65216</td>
      <td>57.35016</td>
      <td>30.92194</td>
      <td>0.134868</td>
      <td>132.666498</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>-2</td>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>680</th>
      <td>141</td>
      <td>154</td>
      <td>p5</td>
      <td>fs</td>
      <td>1.55133</td>
      <td>2.78113</td>
      <td>1.30109</td>
      <td>342.94891</td>
      <td>317.76428</td>
      <td>457.74841</td>
      <td>2.302838</td>
      <td>424.256858</td>
      <td>-2.00000</td>
      <td>-0.78119</td>
      <td>-0.80701</td>
      <td>-250.00000</td>
      <td>-78.21655</td>
      <td>-207.75604</td>
      <td>0.638424</td>
      <td>2.802037</td>
      <td>-0.44867</td>
      <td>1.99994</td>
      <td>0.49408</td>
      <td>92.94891</td>
      <td>239.54773</td>
      <td>249.99237</td>
      <td>2.941263</td>
      <td>427.058895</td>
      <td>-0.940828</td>
      <td>0.486582</td>
      <td>0.034305</td>
      <td>6.870563</td>
      <td>31.170772</td>
      <td>-3.457876</td>
      <td>1.193242</td>
      <td>107.879467</td>
      <td>0.940828</td>
      <td>0.606765</td>
      <td>0.221392</td>
      <td>46.231197</td>
      <td>61.816289</td>
      <td>61.730018</td>
      <td>1.193242</td>
      <td>107.879467</td>
      <td>0.821506</td>
      <td>2.032202</td>
      <td>1.908171</td>
      <td>4.885281</td>
      <td>0.350205</td>
      <td>1.711746</td>
      <td>2.798864</td>
      <td>1.967653</td>
      <td>3.821506</td>
      <td>5.032202</td>
      <td>4.908171</td>
      <td>7.885281</td>
      <td>3.350205</td>
      <td>4.711746</td>
      <td>5.798864</td>
      <td>4.967653</td>
      <td>-2.524690</td>
      <td>0.905919</td>
      <td>-2.367005</td>
      <td>-3.513404</td>
      <td>1.964274</td>
      <td>1.012795</td>
      <td>3.290119</td>
      <td>2.714427</td>
      <td>0.011580</td>
      <td>0.364979</td>
      <td>0.017933</td>
      <td>0.000442</td>
      <td>0.049498</td>
      <td>0.311158</td>
      <td>0.001001</td>
      <td>0.006639</td>
      <td>2.460998</td>
      <td>2.062362</td>
      <td>0.478212</td>
      <td>0.589825</td>
      <td>0.307487</td>
      <td>81.283323</td>
      <td>85.037534</td>
      <td>99.752700</td>
      <td>0.608604</td>
      <td>114.811852</td>
      <td>2.185201</td>
      <td>1.966432</td>
      <td>1.455748</td>
      <td>2.239368</td>
      <td>1.224410</td>
      <td>2.341282</td>
      <td>0.29230</td>
      <td>0.42388</td>
      <td>0.22986</td>
      <td>32.68432</td>
      <td>108.54340</td>
      <td>49.04938</td>
      <td>0.050695</td>
      <td>113.244244</td>
      <td>-3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>681</th>
      <td>157</td>
      <td>170</td>
      <td>p5</td>
      <td>fs</td>
      <td>2.12524</td>
      <td>2.82440</td>
      <td>1.56378</td>
      <td>178.46679</td>
      <td>405.08271</td>
      <td>348.06824</td>
      <td>2.263621</td>
      <td>337.994150</td>
      <td>-2.00000</td>
      <td>-0.82446</td>
      <td>-1.27417</td>
      <td>-51.38397</td>
      <td>-201.29395</td>
      <td>-250.00000</td>
      <td>0.838518</td>
      <td>7.214914</td>
      <td>0.12524</td>
      <td>1.99994</td>
      <td>0.28961</td>
      <td>127.08282</td>
      <td>203.78876</td>
      <td>98.06824</td>
      <td>3.102139</td>
      <td>345.209063</td>
      <td>-0.890208</td>
      <td>0.420454</td>
      <td>-0.006958</td>
      <td>18.474285</td>
      <td>-2.305837</td>
      <td>-43.870778</td>
      <td>1.173553</td>
      <td>98.225230</td>
      <td>0.909475</td>
      <td>0.557942</td>
      <td>0.238461</td>
      <td>35.310598</td>
      <td>57.770362</td>
      <td>60.648405</td>
      <td>1.173553</td>
      <td>98.225230</td>
      <td>2.068577</td>
      <td>1.644497</td>
      <td>5.618899</td>
      <td>0.481868</td>
      <td>1.245703</td>
      <td>0.711208</td>
      <td>6.917279</td>
      <td>0.498725</td>
      <td>5.068577</td>
      <td>4.644497</td>
      <td>8.618899</td>
      <td>3.481868</td>
      <td>4.245703</td>
      <td>3.711208</td>
      <td>9.917279</td>
      <td>3.498725</td>
      <td>-0.467987</td>
      <td>1.350522</td>
      <td>-3.893920</td>
      <td>1.510347</td>
      <td>0.250360</td>
      <td>-1.730696</td>
      <td>4.209719</td>
      <td>2.137525</td>
      <td>0.639794</td>
      <td>0.176849</td>
      <td>0.000099</td>
      <td>0.130955</td>
      <td>0.802309</td>
      <td>0.083506</td>
      <td>0.000026</td>
      <td>0.032555</td>
      <td>2.477152</td>
      <td>2.102311</td>
      <td>0.446199</td>
      <td>0.620146</td>
      <td>0.390288</td>
      <td>44.653747</td>
      <td>89.209108</td>
      <td>84.469378</td>
      <td>0.572776</td>
      <td>98.541176</td>
      <td>2.454402</td>
      <td>2.217448</td>
      <td>1.252678</td>
      <td>1.986948</td>
      <td>1.106185</td>
      <td>2.554358</td>
      <td>0.24688</td>
      <td>0.28992</td>
      <td>0.15308</td>
      <td>45.88318</td>
      <td>22.12525</td>
      <td>79.97894</td>
      <td>0.291334</td>
      <td>111.596742</td>
      <td>-1</td>
      <td>1</td>
      <td>2</td>
      <td>-1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>682</th>
      <td>168</td>
      <td>181</td>
      <td>p5</td>
      <td>fs</td>
      <td>3.34534</td>
      <td>1.85724</td>
      <td>2.55158</td>
      <td>283.64563</td>
      <td>302.45209</td>
      <td>366.87469</td>
      <td>2.478474</td>
      <td>259.905943</td>
      <td>-1.34540</td>
      <td>0.14270</td>
      <td>-0.72101</td>
      <td>-52.03247</td>
      <td>-217.51404</td>
      <td>-226.18103</td>
      <td>0.890577</td>
      <td>11.165010</td>
      <td>1.99994</td>
      <td>1.99994</td>
      <td>1.83057</td>
      <td>231.61316</td>
      <td>84.93805</td>
      <td>140.69366</td>
      <td>3.369051</td>
      <td>271.070952</td>
      <td>-0.685299</td>
      <td>0.517847</td>
      <td>0.176147</td>
      <td>24.915842</td>
      <td>-6.375827</td>
      <td>-10.564365</td>
      <td>1.241984</td>
      <td>88.390173</td>
      <td>0.992982</td>
      <td>0.517847</td>
      <td>0.391255</td>
      <td>50.108103</td>
      <td>39.260864</td>
      <td>39.053696</td>
      <td>1.241984</td>
      <td>88.390173</td>
      <td>7.274664</td>
      <td>4.492660</td>
      <td>2.819668</td>
      <td>1.976168</td>
      <td>4.568268</td>
      <td>3.683212</td>
      <td>6.470031</td>
      <td>-0.489862</td>
      <td>10.274664</td>
      <td>7.492660</td>
      <td>5.819668</td>
      <td>4.976168</td>
      <td>7.568268</td>
      <td>6.683212</td>
      <td>9.470031</td>
      <td>2.510138</td>
      <td>4.276669</td>
      <td>3.623241</td>
      <td>2.474617</td>
      <td>2.701727</td>
      <td>-3.346808</td>
      <td>-2.168101</td>
      <td>4.122382</td>
      <td>2.035784</td>
      <td>0.000019</td>
      <td>0.000291</td>
      <td>0.013338</td>
      <td>0.006898</td>
      <td>0.000817</td>
      <td>0.030151</td>
      <td>0.000037</td>
      <td>0.041772</td>
      <td>2.467556</td>
      <td>2.089018</td>
      <td>0.790726</td>
      <td>0.473813</td>
      <td>0.576505</td>
      <td>74.098162</td>
      <td>67.751819</td>
      <td>74.573055</td>
      <td>0.639070</td>
      <td>92.748156</td>
      <td>1.585722</td>
      <td>0.867214</td>
      <td>1.144132</td>
      <td>1.261241</td>
      <td>1.954883</td>
      <td>1.447667</td>
      <td>0.17718</td>
      <td>0.19520</td>
      <td>0.26916</td>
      <td>51.43738</td>
      <td>30.64727</td>
      <td>19.73725</td>
      <td>0.182370</td>
      <td>81.510902</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>683</th>
      <td>190</td>
      <td>203</td>
      <td>p5</td>
      <td>fs</td>
      <td>3.42352</td>
      <td>2.53851</td>
      <td>2.53980</td>
      <td>328.23181</td>
      <td>363.18207</td>
      <td>459.30481</td>
      <td>2.621138</td>
      <td>397.023079</td>
      <td>-1.42358</td>
      <td>-0.53857</td>
      <td>-0.53986</td>
      <td>-250.00000</td>
      <td>-187.34741</td>
      <td>-209.31244</td>
      <td>0.842860</td>
      <td>3.095950</td>
      <td>1.99994</td>
      <td>1.99994</td>
      <td>1.99994</td>
      <td>78.23181</td>
      <td>175.83466</td>
      <td>249.99237</td>
      <td>3.463998</td>
      <td>400.119029</td>
      <td>-0.680513</td>
      <td>0.390605</td>
      <td>0.313984</td>
      <td>-1.568720</td>
      <td>4.192059</td>
      <td>-13.521634</td>
      <td>1.229279</td>
      <td>89.181953</td>
      <td>0.988196</td>
      <td>0.524196</td>
      <td>0.397039</td>
      <td>48.520015</td>
      <td>41.729855</td>
      <td>53.007858</td>
      <td>1.229279</td>
      <td>89.181953</td>
      <td>6.978713</td>
      <td>2.634473</td>
      <td>4.896935</td>
      <td>4.491592</td>
      <td>2.125905</td>
      <td>3.392379</td>
      <td>6.455244</td>
      <td>1.859536</td>
      <td>9.978713</td>
      <td>5.634473</td>
      <td>7.896935</td>
      <td>7.491592</td>
      <td>5.125905</td>
      <td>6.392379</td>
      <td>9.455244</td>
      <td>4.859536</td>
      <td>4.187131</td>
      <td>2.205469</td>
      <td>3.303961</td>
      <td>-3.474587</td>
      <td>-0.495368</td>
      <td>1.837873</td>
      <td>4.120335</td>
      <td>2.937679</td>
      <td>0.000028</td>
      <td>0.027421</td>
      <td>0.000953</td>
      <td>0.000512</td>
      <td>0.620340</td>
      <td>0.066081</td>
      <td>0.000038</td>
      <td>0.003307</td>
      <td>2.456461</td>
      <td>1.905592</td>
      <td>0.795668</td>
      <td>0.570217</td>
      <td>0.538352</td>
      <td>79.498356</td>
      <td>76.382588</td>
      <td>93.320945</td>
      <td>0.671705</td>
      <td>114.509978</td>
      <td>1.373715</td>
      <td>0.504224</td>
      <td>1.265677</td>
      <td>0.844133</td>
      <td>2.855966</td>
      <td>2.487025</td>
      <td>0.08130</td>
      <td>0.18060</td>
      <td>0.13293</td>
      <td>52.39106</td>
      <td>13.58033</td>
      <td>30.24292</td>
      <td>0.162151</td>
      <td>78.276297</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>-1</td>
      <td>1</td>
      <td>-5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>684</th>
      <td>218</td>
      <td>231</td>
      <td>p5</td>
      <td>fs</td>
      <td>2.86652</td>
      <td>1.91925</td>
      <td>1.86780</td>
      <td>267.61627</td>
      <td>400.45166</td>
      <td>286.14044</td>
      <td>1.932268</td>
      <td>422.557628</td>
      <td>-1.75385</td>
      <td>0.08069</td>
      <td>-0.65619</td>
      <td>-17.62390</td>
      <td>-239.21967</td>
      <td>-250.00000</td>
      <td>0.657287</td>
      <td>4.317337</td>
      <td>1.11267</td>
      <td>1.99994</td>
      <td>1.21161</td>
      <td>249.99237</td>
      <td>161.23199</td>
      <td>36.14044</td>
      <td>2.589555</td>
      <td>426.874965</td>
      <td>-0.796828</td>
      <td>0.540931</td>
      <td>0.139363</td>
      <td>44.766353</td>
      <td>-8.067204</td>
      <td>-43.964678</td>
      <td>1.259611</td>
      <td>105.560954</td>
      <td>0.968008</td>
      <td>0.540931</td>
      <td>0.376142</td>
      <td>51.286550</td>
      <td>64.970750</td>
      <td>52.226726</td>
      <td>1.259611</td>
      <td>105.560954</td>
      <td>3.532514</td>
      <td>1.252902</td>
      <td>0.220994</td>
      <td>2.324300</td>
      <td>0.667185</td>
      <td>1.605213</td>
      <td>0.998324</td>
      <td>2.047023</td>
      <td>6.532514</td>
      <td>4.252902</td>
      <td>3.220994</td>
      <td>5.324300</td>
      <td>3.667185</td>
      <td>4.605213</td>
      <td>3.998324</td>
      <td>5.047023</td>
      <td>3.011317</td>
      <td>2.657609</td>
      <td>1.031193</td>
      <td>2.923905</td>
      <td>-1.051537</td>
      <td>-2.695319</td>
      <td>2.660677</td>
      <td>2.715056</td>
      <td>0.002601</td>
      <td>0.007870</td>
      <td>0.302450</td>
      <td>0.003457</td>
      <td>0.293012</td>
      <td>0.007032</td>
      <td>0.007798</td>
      <td>0.006626</td>
      <td>2.481743</td>
      <td>2.056041</td>
      <td>0.640134</td>
      <td>0.560715</td>
      <td>0.471932</td>
      <td>72.003077</td>
      <td>96.012285</td>
      <td>76.022111</td>
      <td>0.554288</td>
      <td>114.197406</td>
      <td>1.882842</td>
      <td>1.267603</td>
      <td>1.381731</td>
      <td>2.218545</td>
      <td>1.197172</td>
      <td>2.688755</td>
      <td>0.03168</td>
      <td>0.56104</td>
      <td>0.40973</td>
      <td>57.97577</td>
      <td>34.79004</td>
      <td>53.15399</td>
      <td>0.235702</td>
      <td>124.652290</td>
      <td>-1</td>
      <td>6</td>
      <td>-1</td>
      <td>6</td>
      <td>3</td>
      <td>4</td>
      <td>-2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>685</th>
      <td>237</td>
      <td>250</td>
      <td>p5</td>
      <td>fs</td>
      <td>2.57965</td>
      <td>2.02936</td>
      <td>0.96228</td>
      <td>286.49139</td>
      <td>282.68433</td>
      <td>311.64551</td>
      <td>2.047728</td>
      <td>425.299886</td>
      <td>-2.00000</td>
      <td>-0.02942</td>
      <td>-0.01514</td>
      <td>-36.49902</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>0.781501</td>
      <td>7.708410</td>
      <td>0.57965</td>
      <td>1.99994</td>
      <td>0.94714</td>
      <td>249.99237</td>
      <td>32.68433</td>
      <td>61.64551</td>
      <td>2.829230</td>
      <td>433.008297</td>
      <td>-0.902865</td>
      <td>0.575425</td>
      <td>0.231638</td>
      <td>30.048078</td>
      <td>-48.085138</td>
      <td>-27.459952</td>
      <td>1.252187</td>
      <td>91.514948</td>
      <td>0.992042</td>
      <td>0.579952</td>
      <td>0.233968</td>
      <td>46.035765</td>
      <td>55.387057</td>
      <td>44.182410</td>
      <td>1.252187</td>
      <td>91.514948</td>
      <td>1.880615</td>
      <td>2.264206</td>
      <td>2.270726</td>
      <td>4.472735</td>
      <td>1.088333</td>
      <td>3.915638</td>
      <td>3.726867</td>
      <td>3.741241</td>
      <td>4.880615</td>
      <td>5.264206</td>
      <td>5.270726</td>
      <td>7.472735</td>
      <td>4.088333</td>
      <td>6.915638</td>
      <td>6.726867</td>
      <td>6.741241</td>
      <td>1.446815</td>
      <td>2.698546</td>
      <td>2.813742</td>
      <td>3.452092</td>
      <td>-2.717659</td>
      <td>-3.296678</td>
      <td>3.467948</td>
      <td>3.449158</td>
      <td>0.147949</td>
      <td>0.006964</td>
      <td>0.004897</td>
      <td>0.000556</td>
      <td>0.006575</td>
      <td>0.000978</td>
      <td>0.000524</td>
      <td>0.000562</td>
      <td>2.494581</td>
      <td>2.016621</td>
      <td>0.565728</td>
      <td>0.505331</td>
      <td>0.252260</td>
      <td>70.466434</td>
      <td>82.400336</td>
      <td>72.927695</td>
      <td>0.520582</td>
      <td>112.551824</td>
      <td>2.323012</td>
      <td>1.060209</td>
      <td>2.191958</td>
      <td>2.307128</td>
      <td>1.028178</td>
      <td>2.584717</td>
      <td>0.17108</td>
      <td>0.34723</td>
      <td>0.23969</td>
      <td>40.48156</td>
      <td>39.54316</td>
      <td>35.98022</td>
      <td>0.202374</td>
      <td>53.237106</td>
      <td>-1</td>
      <td>-4</td>
      <td>3</td>
      <td>-1</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>686</th>
      <td>249</td>
      <td>262</td>
      <td>p5</td>
      <td>fs</td>
      <td>3.92285</td>
      <td>2.23322</td>
      <td>2.65473</td>
      <td>286.73553</td>
      <td>348.21320</td>
      <td>272.79663</td>
      <td>2.604047</td>
      <td>425.880395</td>
      <td>-1.92291</td>
      <td>-0.23328</td>
      <td>-0.65479</td>
      <td>-36.74316</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>0.859951</td>
      <td>7.127902</td>
      <td>1.99994</td>
      <td>1.99994</td>
      <td>1.99994</td>
      <td>249.99237</td>
      <td>98.21320</td>
      <td>22.79663</td>
      <td>3.463998</td>
      <td>433.008297</td>
      <td>-0.828872</td>
      <td>0.453142</td>
      <td>0.415152</td>
      <td>28.906602</td>
      <td>-27.915368</td>
      <td>-36.634006</td>
      <td>1.405428</td>
      <td>103.122632</td>
      <td>1.136555</td>
      <td>0.489032</td>
      <td>0.515889</td>
      <td>37.705055</td>
      <td>71.823122</td>
      <td>46.235305</td>
      <td>1.405428</td>
      <td>103.122632</td>
      <td>4.448183</td>
      <td>2.669016</td>
      <td>2.823730</td>
      <td>5.391793</td>
      <td>0.311317</td>
      <td>3.038082</td>
      <td>2.221641</td>
      <td>2.410248</td>
      <td>7.448183</td>
      <td>5.669016</td>
      <td>5.823730</td>
      <td>8.391793</td>
      <td>3.311317</td>
      <td>6.038082</td>
      <td>5.221641</td>
      <td>5.410248</td>
      <td>3.291036</td>
      <td>2.797828</td>
      <td>2.147678</td>
      <td>3.799835</td>
      <td>-2.152892</td>
      <td>-3.193056</td>
      <td>2.971455</td>
      <td>3.008750</td>
      <td>0.000998</td>
      <td>0.005145</td>
      <td>0.031739</td>
      <td>0.000145</td>
      <td>0.031327</td>
      <td>0.001408</td>
      <td>0.002964</td>
      <td>0.002623</td>
      <td>2.454893</td>
      <td>2.066378</td>
      <td>0.911893</td>
      <td>0.535728</td>
      <td>0.563999</td>
      <td>68.605612</td>
      <td>105.446392</td>
      <td>72.295340</td>
      <td>0.725354</td>
      <td>115.649990</td>
      <td>1.582511</td>
      <td>0.774741</td>
      <td>1.470192</td>
      <td>2.304002</td>
      <td>0.890428</td>
      <td>2.781817</td>
      <td>0.24219</td>
      <td>0.38794</td>
      <td>0.20557</td>
      <td>38.48267</td>
      <td>50.36927</td>
      <td>41.06903</td>
      <td>0.603526</td>
      <td>75.181127</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-5</td>
      <td>2</td>
      <td>-3</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>687</th>
      <td>277</td>
      <td>290</td>
      <td>p5</td>
      <td>fs</td>
      <td>2.48590</td>
      <td>2.53919</td>
      <td>1.42890</td>
      <td>133.96454</td>
      <td>412.17804</td>
      <td>420.43304</td>
      <td>1.654039</td>
      <td>355.495707</td>
      <td>-1.45721</td>
      <td>-0.53925</td>
      <td>-1.00092</td>
      <td>-58.12073</td>
      <td>-250.00000</td>
      <td>-170.44067</td>
      <td>0.807628</td>
      <td>3.835587</td>
      <td>1.02869</td>
      <td>1.99994</td>
      <td>0.42798</td>
      <td>75.84381</td>
      <td>162.17804</td>
      <td>249.99237</td>
      <td>2.461667</td>
      <td>359.331293</td>
      <td>-0.799340</td>
      <td>0.527222</td>
      <td>0.110370</td>
      <td>15.203622</td>
      <td>-12.725830</td>
      <td>5.896348</td>
      <td>1.244030</td>
      <td>84.738997</td>
      <td>0.957600</td>
      <td>0.610183</td>
      <td>0.311965</td>
      <td>27.986966</td>
      <td>47.704842</td>
      <td>54.398168</td>
      <td>1.244030</td>
      <td>84.738997</td>
      <td>5.097978</td>
      <td>0.834016</td>
      <td>3.161137</td>
      <td>-0.444896</td>
      <td>2.851249</td>
      <td>1.779628</td>
      <td>1.275359</td>
      <td>1.346180</td>
      <td>8.097978</td>
      <td>3.834016</td>
      <td>6.161137</td>
      <td>2.555104</td>
      <td>5.851249</td>
      <td>4.779628</td>
      <td>4.275359</td>
      <td>4.346180</td>
      <td>3.591272</td>
      <td>1.544806</td>
      <td>-3.139127</td>
      <td>0.208219</td>
      <td>-1.806321</td>
      <td>1.730623</td>
      <td>2.490754</td>
      <td>2.627174</td>
      <td>0.000329</td>
      <td>0.122393</td>
      <td>0.001695</td>
      <td>0.835058</td>
      <td>0.070868</td>
      <td>0.083519</td>
      <td>0.012747</td>
      <td>0.008610</td>
      <td>2.505039</td>
      <td>1.926938</td>
      <td>0.574329</td>
      <td>0.597366</td>
      <td>0.374254</td>
      <td>36.499854</td>
      <td>85.806543</td>
      <td>93.482238</td>
      <td>0.456335</td>
      <td>103.349394</td>
      <td>1.846592</td>
      <td>1.798899</td>
      <td>2.268386</td>
      <td>1.990716</td>
      <td>2.450142</td>
      <td>1.447930</td>
      <td>0.21186</td>
      <td>0.53936</td>
      <td>0.29761</td>
      <td>40.90881</td>
      <td>29.31976</td>
      <td>21.10290</td>
      <td>0.499653</td>
      <td>104.029109</td>
      <td>-1</td>
      <td>1</td>
      <td>-7</td>
      <td>-4</td>
      <td>-1</td>
      <td>1</td>
      <td>-3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>688</th>
      <td>294</td>
      <td>307</td>
      <td>p5</td>
      <td>fs</td>
      <td>1.50427</td>
      <td>2.21387</td>
      <td>2.28375</td>
      <td>249.55750</td>
      <td>248.90900</td>
      <td>280.96008</td>
      <td>2.111770</td>
      <td>357.296954</td>
      <td>-2.00000</td>
      <td>-0.44574</td>
      <td>-1.37280</td>
      <td>-107.91016</td>
      <td>-214.75983</td>
      <td>-250.00000</td>
      <td>0.890040</td>
      <td>1.431012</td>
      <td>-0.49573</td>
      <td>1.76813</td>
      <td>0.91095</td>
      <td>141.64734</td>
      <td>34.14917</td>
      <td>30.96008</td>
      <td>3.001810</td>
      <td>358.727966</td>
      <td>-0.995991</td>
      <td>0.322374</td>
      <td>0.165434</td>
      <td>21.319462</td>
      <td>-28.076761</td>
      <td>-32.402039</td>
      <td>1.202160</td>
      <td>75.525473</td>
      <td>0.995991</td>
      <td>0.424688</td>
      <td>0.376634</td>
      <td>39.426950</td>
      <td>37.071230</td>
      <td>43.227561</td>
      <td>1.202160</td>
      <td>75.525473</td>
      <td>4.320372</td>
      <td>3.422502</td>
      <td>4.179804</td>
      <td>1.257188</td>
      <td>4.773246</td>
      <td>4.027874</td>
      <td>7.465741</td>
      <td>4.674924</td>
      <td>7.320372</td>
      <td>6.422502</td>
      <td>7.179804</td>
      <td>4.257188</td>
      <td>7.773246</td>
      <td>7.027874</td>
      <td>10.465741</td>
      <td>7.674924</td>
      <td>-3.190563</td>
      <td>2.722872</td>
      <td>-3.078335</td>
      <td>-0.228807</td>
      <td>-3.576027</td>
      <td>-3.441247</td>
      <td>4.340613</td>
      <td>3.615165</td>
      <td>0.001420</td>
      <td>0.006472</td>
      <td>0.002082</td>
      <td>0.819019</td>
      <td>0.000349</td>
      <td>0.000579</td>
      <td>0.000014</td>
      <td>0.000300</td>
      <td>2.493692</td>
      <td>2.041197</td>
      <td>0.326510</td>
      <td>0.489880</td>
      <td>0.503222</td>
      <td>54.828037</td>
      <td>59.146562</td>
      <td>70.704583</td>
      <td>0.527188</td>
      <td>89.956581</td>
      <td>2.443802</td>
      <td>2.212450</td>
      <td>1.588390</td>
      <td>2.210033</td>
      <td>0.431367</td>
      <td>2.068568</td>
      <td>0.09876</td>
      <td>0.12536</td>
      <td>0.12812</td>
      <td>54.63410</td>
      <td>44.36493</td>
      <td>46.66900</td>
      <td>0.133559</td>
      <td>65.753144</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>2</td>
      <td>-3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>689</th>
      <td>313</td>
      <td>326</td>
      <td>p5</td>
      <td>fs</td>
      <td>3.99994</td>
      <td>2.25226</td>
      <td>2.24628</td>
      <td>422.33276</td>
      <td>399.30725</td>
      <td>499.99237</td>
      <td>2.829958</td>
      <td>422.306376</td>
      <td>-2.00000</td>
      <td>-0.25232</td>
      <td>-2.00000</td>
      <td>-250.00000</td>
      <td>-149.31488</td>
      <td>-250.00000</td>
      <td>0.634074</td>
      <td>10.697515</td>
      <td>1.99994</td>
      <td>1.99994</td>
      <td>0.24628</td>
      <td>172.33276</td>
      <td>249.99237</td>
      <td>249.99237</td>
      <td>3.464032</td>
      <td>433.003892</td>
      <td>-0.720055</td>
      <td>0.428204</td>
      <td>-0.070815</td>
      <td>-11.535645</td>
      <td>41.090157</td>
      <td>-3.863995</td>
      <td>1.228323</td>
      <td>110.980965</td>
      <td>1.027738</td>
      <td>0.467022</td>
      <td>0.285903</td>
      <td>49.594586</td>
      <td>67.143952</td>
      <td>64.547025</td>
      <td>1.228323</td>
      <td>110.980965</td>
      <td>5.425770</td>
      <td>4.321404</td>
      <td>6.976124</td>
      <td>2.540893</td>
      <td>0.421024</td>
      <td>1.594411</td>
      <td>4.618721</td>
      <td>0.487988</td>
      <td>8.425770</td>
      <td>7.321404</td>
      <td>9.976124</td>
      <td>5.540893</td>
      <td>3.421024</td>
      <td>4.594411</td>
      <td>7.618721</td>
      <td>3.487988</td>
      <td>3.551178</td>
      <td>3.360891</td>
      <td>-4.223095</td>
      <td>-1.542941</td>
      <td>1.422166</td>
      <td>0.049408</td>
      <td>3.722835</td>
      <td>2.445360</td>
      <td>0.000384</td>
      <td>0.000777</td>
      <td>0.000024</td>
      <td>0.122845</td>
      <td>0.154978</td>
      <td>0.960595</td>
      <td>0.000197</td>
      <td>0.014471</td>
      <td>2.438486</td>
      <td>1.888617</td>
      <td>0.850958</td>
      <td>0.508262</td>
      <td>0.572089</td>
      <td>88.446914</td>
      <td>103.810097</td>
      <td>107.913853</td>
      <td>0.712615</td>
      <td>140.585990</td>
      <td>1.432393</td>
      <td>2.434034</td>
      <td>2.193008</td>
      <td>1.796883</td>
      <td>1.701249</td>
      <td>2.509556</td>
      <td>0.09131</td>
      <td>0.18372</td>
      <td>0.25202</td>
      <td>19.99664</td>
      <td>53.31421</td>
      <td>22.39227</td>
      <td>0.150540</td>
      <td>150.468537</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>-1</td>
      <td>-4</td>
      <td>1</td>
      <td>-2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>690</th>
      <td>321</td>
      <td>334</td>
      <td>p5</td>
      <td>fs</td>
      <td>1.52545</td>
      <td>2.60059</td>
      <td>0.56653</td>
      <td>366.27960</td>
      <td>399.30725</td>
      <td>464.46228</td>
      <td>2.197516</td>
      <td>418.151513</td>
      <td>-2.00000</td>
      <td>-0.60065</td>
      <td>-0.23914</td>
      <td>-250.00000</td>
      <td>-149.31488</td>
      <td>-214.46991</td>
      <td>0.634074</td>
      <td>14.852379</td>
      <td>-0.47455</td>
      <td>1.99994</td>
      <td>0.32739</td>
      <td>116.27960</td>
      <td>249.99237</td>
      <td>249.99237</td>
      <td>2.831590</td>
      <td>433.003892</td>
      <td>-1.018743</td>
      <td>0.621446</td>
      <td>0.037484</td>
      <td>-9.218070</td>
      <td>4.710857</td>
      <td>-6.659288</td>
      <td>1.286631</td>
      <td>136.777836</td>
      <td>1.018743</td>
      <td>0.713854</td>
      <td>0.131264</td>
      <td>58.008635</td>
      <td>86.381765</td>
      <td>74.188820</td>
      <td>1.286631</td>
      <td>136.777836</td>
      <td>0.955928</td>
      <td>1.196869</td>
      <td>-0.828892</td>
      <td>1.667946</td>
      <td>-0.246189</td>
      <td>1.185705</td>
      <td>1.519183</td>
      <td>1.286147</td>
      <td>3.955928</td>
      <td>4.196869</td>
      <td>2.171108</td>
      <td>4.667946</td>
      <td>2.753811</td>
      <td>4.185705</td>
      <td>4.519183</td>
      <td>4.286147</td>
      <td>-2.560918</td>
      <td>0.939611</td>
      <td>0.206112</td>
      <td>-1.892243</td>
      <td>1.163875</td>
      <td>0.902612</td>
      <td>2.712460</td>
      <td>2.350978</td>
      <td>0.010440</td>
      <td>0.347417</td>
      <td>0.836703</td>
      <td>0.058459</td>
      <td>0.244475</td>
      <td>0.366732</td>
      <td>0.006679</td>
      <td>0.018724</td>
      <td>2.479125</td>
      <td>2.259725</td>
      <td>0.441975</td>
      <td>0.583303</td>
      <td>0.157094</td>
      <td>89.711010</td>
      <td>110.099970</td>
      <td>104.464114</td>
      <td>0.574700</td>
      <td>111.920305</td>
      <td>2.343951</td>
      <td>1.305527</td>
      <td>1.808029</td>
      <td>2.116285</td>
      <td>1.416357</td>
      <td>2.295076</td>
      <td>0.23498</td>
      <td>0.39838</td>
      <td>0.21759</td>
      <td>33.82111</td>
      <td>132.54547</td>
      <td>72.74628</td>
      <td>0.119832</td>
      <td>120.478664</td>
      <td>-5</td>
      <td>1</td>
      <td>4</td>
      <td>-1</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>691</th>
      <td>340</td>
      <td>353</td>
      <td>p5</td>
      <td>fs</td>
      <td>1.66425</td>
      <td>3.57446</td>
      <td>1.29052</td>
      <td>387.22229</td>
      <td>392.84515</td>
      <td>499.99237</td>
      <td>2.462572</td>
      <td>317.580849</td>
      <td>-2.00000</td>
      <td>-1.57452</td>
      <td>-1.06152</td>
      <td>-192.13867</td>
      <td>-175.14038</td>
      <td>-250.00000</td>
      <td>0.558452</td>
      <td>1.613717</td>
      <td>-0.33575</td>
      <td>1.99994</td>
      <td>0.22900</td>
      <td>195.08362</td>
      <td>217.70477</td>
      <td>249.99237</td>
      <td>3.021024</td>
      <td>319.194566</td>
      <td>-1.041458</td>
      <td>0.359459</td>
      <td>-0.060369</td>
      <td>3.242493</td>
      <td>-1.815211</td>
      <td>0.083336</td>
      <td>1.263940</td>
      <td>111.041332</td>
      <td>1.041458</td>
      <td>0.601693</td>
      <td>0.188608</td>
      <td>54.265536</td>
      <td>57.246869</td>
      <td>55.702796</td>
      <td>1.263940</td>
      <td>111.041332</td>
      <td>1.861378</td>
      <td>3.025460</td>
      <td>4.781647</td>
      <td>1.034625</td>
      <td>1.077960</td>
      <td>2.290568</td>
      <td>2.671172</td>
      <td>-1.150060</td>
      <td>4.861378</td>
      <td>6.025460</td>
      <td>7.781647</td>
      <td>4.034625</td>
      <td>4.077960</td>
      <td>5.290568</td>
      <td>5.671172</td>
      <td>1.849940</td>
      <td>-1.665514</td>
      <td>-1.085631</td>
      <td>-3.628847</td>
      <td>-0.048105</td>
      <td>1.277228</td>
      <td>-0.090855</td>
      <td>3.088669</td>
      <td>1.350886</td>
      <td>0.095810</td>
      <td>0.277642</td>
      <td>0.000285</td>
      <td>0.961633</td>
      <td>0.201522</td>
      <td>0.927608</td>
      <td>0.002011</td>
      <td>0.176732</td>
      <td>2.467634</td>
      <td>1.974600</td>
      <td>0.364262</td>
      <td>0.721873</td>
      <td>0.316807</td>
      <td>87.100129</td>
      <td>90.504704</td>
      <td>103.566210</td>
      <td>0.611637</td>
      <td>119.109748</td>
      <td>2.093414</td>
      <td>2.171277</td>
      <td>1.176699</td>
      <td>1.667538</td>
      <td>2.063712</td>
      <td>2.413969</td>
      <td>0.22784</td>
      <td>0.05267</td>
      <td>0.21679</td>
      <td>16.92200</td>
      <td>40.19165</td>
      <td>16.59393</td>
      <td>0.175514</td>
      <td>226.652612</td>
      <td>1</td>
      <td>1</td>
      <td>-7</td>
      <td>-2</td>
      <td>2</td>
      <td>1</td>
      <td>-1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>692</th>
      <td>372</td>
      <td>385</td>
      <td>p5</td>
      <td>fs</td>
      <td>1.68970</td>
      <td>3.17719</td>
      <td>0.95850</td>
      <td>300.63629</td>
      <td>339.33258</td>
      <td>499.99237</td>
      <td>2.443822</td>
      <td>354.503576</td>
      <td>-2.00000</td>
      <td>-1.17725</td>
      <td>-0.63947</td>
      <td>-102.89764</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>0.455951</td>
      <td>3.627900</td>
      <td>-0.31030</td>
      <td>1.99994</td>
      <td>0.31903</td>
      <td>197.73865</td>
      <td>89.33258</td>
      <td>249.99237</td>
      <td>2.899773</td>
      <td>358.131476</td>
      <td>-0.919995</td>
      <td>0.369182</td>
      <td>0.005324</td>
      <td>15.397879</td>
      <td>-20.719675</td>
      <td>-25.177002</td>
      <td>1.187254</td>
      <td>100.275438</td>
      <td>0.919995</td>
      <td>0.550298</td>
      <td>0.284875</td>
      <td>44.942416</td>
      <td>44.742878</td>
      <td>67.745502</td>
      <td>1.187254</td>
      <td>100.275438</td>
      <td>0.594703</td>
      <td>1.866701</td>
      <td>-0.682350</td>
      <td>1.676324</td>
      <td>3.209174</td>
      <td>2.102125</td>
      <td>3.586829</td>
      <td>0.308598</td>
      <td>3.594703</td>
      <td>4.866701</td>
      <td>2.317650</td>
      <td>4.676324</td>
      <td>6.209174</td>
      <td>5.102125</td>
      <td>6.586829</td>
      <td>3.308598</td>
      <td>-1.610759</td>
      <td>0.435963</td>
      <td>-1.563499</td>
      <td>1.906851</td>
      <td>-2.870361</td>
      <td>1.020416</td>
      <td>3.215952</td>
      <td>2.326411</td>
      <td>0.107232</td>
      <td>0.662864</td>
      <td>0.117935</td>
      <td>0.056540</td>
      <td>0.004100</td>
      <td>0.307531</td>
      <td>0.001300</td>
      <td>0.019997</td>
      <td>2.468973</td>
      <td>1.981724</td>
      <td>0.438493</td>
      <td>0.673281</td>
      <td>0.324443</td>
      <td>68.869285</td>
      <td>78.033959</td>
      <td>106.129066</td>
      <td>0.569201</td>
      <td>115.502265</td>
      <td>2.117046</td>
      <td>2.140102</td>
      <td>1.471385</td>
      <td>0.965879</td>
      <td>2.294610</td>
      <td>2.126323</td>
      <td>0.40076</td>
      <td>0.26709</td>
      <td>0.44250</td>
      <td>44.30389</td>
      <td>16.25824</td>
      <td>75.28686</td>
      <td>0.143270</td>
      <td>89.104255</td>
      <td>-3</td>
      <td>1</td>
      <td>-3</td>
      <td>-2</td>
      <td>-1</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>693</th>
      <td>382</td>
      <td>395</td>
      <td>p5</td>
      <td>fs</td>
      <td>2.67474</td>
      <td>2.30689</td>
      <td>1.10382</td>
      <td>311.27930</td>
      <td>371.65070</td>
      <td>486.47308</td>
      <td>2.119160</td>
      <td>423.319668</td>
      <td>-2.00000</td>
      <td>-0.30695</td>
      <td>-0.58319</td>
      <td>-61.28693</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>0.709378</td>
      <td>9.688629</td>
      <td>0.67474</td>
      <td>1.99994</td>
      <td>0.52063</td>
      <td>249.99237</td>
      <td>121.65070</td>
      <td>236.47308</td>
      <td>2.828538</td>
      <td>433.008297</td>
      <td>-0.853581</td>
      <td>0.506075</td>
      <td>-0.029594</td>
      <td>21.621118</td>
      <td>-42.981073</td>
      <td>-10.942312</td>
      <td>1.202506</td>
      <td>116.190181</td>
      <td>0.957387</td>
      <td>0.553298</td>
      <td>0.169495</td>
      <td>50.286515</td>
      <td>73.132441</td>
      <td>65.026505</td>
      <td>1.202506</td>
      <td>116.190181</td>
      <td>1.609860</td>
      <td>1.475017</td>
      <td>1.054715</td>
      <td>3.642578</td>
      <td>0.436607</td>
      <td>1.886343</td>
      <td>4.988097</td>
      <td>1.360611</td>
      <td>4.609860</td>
      <td>4.475017</td>
      <td>4.054715</td>
      <td>6.642578</td>
      <td>3.436607</td>
      <td>4.886343</td>
      <td>7.988097</td>
      <td>4.360611</td>
      <td>1.224129</td>
      <td>2.134606</td>
      <td>-0.206479</td>
      <td>3.100144</td>
      <td>-1.702498</td>
      <td>0.179599</td>
      <td>3.684193</td>
      <td>2.825756</td>
      <td>0.220903</td>
      <td>0.032793</td>
      <td>0.836417</td>
      <td>0.001934</td>
      <td>0.088662</td>
      <td>0.857467</td>
      <td>0.000229</td>
      <td>0.004717</td>
      <td>2.492373</td>
      <td>2.120290</td>
      <td>0.593201</td>
      <td>0.555815</td>
      <td>0.246675</td>
      <td>75.673842</td>
      <td>100.218926</td>
      <td>102.796258</td>
      <td>0.511110</td>
      <td>123.579350</td>
      <td>2.484760</td>
      <td>1.821043</td>
      <td>1.241502</td>
      <td>2.267372</td>
      <td>1.634783</td>
      <td>2.144819</td>
      <td>0.36694</td>
      <td>0.64550</td>
      <td>0.18201</td>
      <td>51.06354</td>
      <td>58.24280</td>
      <td>53.05481</td>
      <td>0.304779</td>
      <td>60.093052</td>
      <td>-1</td>
      <td>1</td>
      <td>6</td>
      <td>-3</td>
      <td>-1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>694</th>
      <td>398</td>
      <td>411</td>
      <td>p5</td>
      <td>fs</td>
      <td>2.41070</td>
      <td>2.44708</td>
      <td>2.29364</td>
      <td>384.07898</td>
      <td>328.44543</td>
      <td>362.99896</td>
      <td>2.321445</td>
      <td>420.701511</td>
      <td>-1.47955</td>
      <td>-0.44714</td>
      <td>-0.29370</td>
      <td>-235.81696</td>
      <td>-250.00000</td>
      <td>-113.00659</td>
      <td>0.656232</td>
      <td>4.275928</td>
      <td>0.93115</td>
      <td>1.99994</td>
      <td>1.99994</td>
      <td>148.26202</td>
      <td>78.44543</td>
      <td>249.99237</td>
      <td>2.977677</td>
      <td>424.977439</td>
      <td>-0.640593</td>
      <td>0.578833</td>
      <td>0.312336</td>
      <td>10.499808</td>
      <td>-38.690422</td>
      <td>-8.598915</td>
      <td>1.201351</td>
      <td>116.168368</td>
      <td>0.783847</td>
      <td>0.683692</td>
      <td>0.386170</td>
      <td>58.412406</td>
      <td>71.274977</td>
      <td>55.050190</td>
      <td>1.201351</td>
      <td>116.168368</td>
      <td>3.139596</td>
      <td>1.392368</td>
      <td>4.405653</td>
      <td>2.116181</td>
      <td>0.059317</td>
      <td>3.790638</td>
      <td>5.679127</td>
      <td>1.140841</td>
      <td>6.139596</td>
      <td>4.392368</td>
      <td>7.405653</td>
      <td>5.116181</td>
      <td>3.059317</td>
      <td>6.790638</td>
      <td>8.679127</td>
      <td>4.140841</td>
      <td>2.838856</td>
      <td>1.118818</td>
      <td>3.423269</td>
      <td>-2.103326</td>
      <td>-2.010585</td>
      <td>3.147756</td>
      <td>3.893642</td>
      <td>2.394373</td>
      <td>0.004528</td>
      <td>0.263218</td>
      <td>0.000619</td>
      <td>0.035437</td>
      <td>0.044369</td>
      <td>0.001645</td>
      <td>0.000099</td>
      <td>0.016649</td>
      <td>2.484456</td>
      <td>2.086692</td>
      <td>0.537978</td>
      <td>0.561795</td>
      <td>0.542300</td>
      <td>89.259133</td>
      <td>102.547212</td>
      <td>85.279220</td>
      <td>0.546662</td>
      <td>118.075040</td>
      <td>1.716718</td>
      <td>0.739842</td>
      <td>1.531820</td>
      <td>1.460703</td>
      <td>2.012088</td>
      <td>2.444296</td>
      <td>0.09253</td>
      <td>0.23932</td>
      <td>0.31403</td>
      <td>45.42541</td>
      <td>90.31677</td>
      <td>42.52625</td>
      <td>0.088176</td>
      <td>128.699486</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>-1</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>695</th>
      <td>3</td>
      <td>16</td>
      <td>p5</td>
      <td>fu</td>
      <td>1.47668</td>
      <td>1.54602</td>
      <td>2.09100</td>
      <td>429.12292</td>
      <td>260.87189</td>
      <td>281.96716</td>
      <td>2.349910</td>
      <td>358.554349</td>
      <td>-1.29559</td>
      <td>0.45392</td>
      <td>-0.09106</td>
      <td>-250.00000</td>
      <td>-218.93311</td>
      <td>-31.97479</td>
      <td>0.496615</td>
      <td>1.359791</td>
      <td>0.18109</td>
      <td>1.99994</td>
      <td>1.99994</td>
      <td>179.12292</td>
      <td>41.93878</td>
      <td>249.99237</td>
      <td>2.846526</td>
      <td>359.914140</td>
      <td>-0.606767</td>
      <td>0.790649</td>
      <td>0.254498</td>
      <td>0.460698</td>
      <td>-38.906392</td>
      <td>35.503095</td>
      <td>1.157973</td>
      <td>94.184554</td>
      <td>0.634627</td>
      <td>0.790649</td>
      <td>0.304912</td>
      <td>51.598768</td>
      <td>48.087487</td>
      <td>44.246380</td>
      <td>1.157973</td>
      <td>94.184554</td>
      <td>0.361205</td>
      <td>5.109279</td>
      <td>4.430519</td>
      <td>2.238215</td>
      <td>0.742394</td>
      <td>1.602324</td>
      <td>4.507227</td>
      <td>0.254747</td>
      <td>3.361205</td>
      <td>8.109279</td>
      <td>7.430519</td>
      <td>5.238215</td>
      <td>3.742394</td>
      <td>4.602324</td>
      <td>7.507227</td>
      <td>3.254747</td>
      <td>0.605863</td>
      <td>3.735648</td>
      <td>3.632582</td>
      <td>-1.648348</td>
      <td>-2.428580</td>
      <td>3.066772</td>
      <td>3.525468</td>
      <td>2.341386</td>
      <td>0.544605</td>
      <td>0.000187</td>
      <td>0.000281</td>
      <td>0.099281</td>
      <td>0.015158</td>
      <td>0.002164</td>
      <td>0.000423</td>
      <td>0.019212</td>
      <td>2.476302</td>
      <td>1.790188</td>
      <td>0.347682</td>
      <td>0.378487</td>
      <td>0.558544</td>
      <td>93.090539</td>
      <td>72.805517</td>
      <td>86.705895</td>
      <td>0.541541</td>
      <td>124.048633</td>
      <td>2.439581</td>
      <td>0.781517</td>
      <td>1.754523</td>
      <td>2.278393</td>
      <td>2.090093</td>
      <td>1.745760</td>
      <td>0.41290</td>
      <td>0.21038</td>
      <td>0.20081</td>
      <td>14.98413</td>
      <td>68.42041</td>
      <td>9.32312</td>
      <td>0.088175</td>
      <td>90.552758</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>696</th>
      <td>18</td>
      <td>31</td>
      <td>p5</td>
      <td>fu</td>
      <td>0.83752</td>
      <td>1.41364</td>
      <td>1.15888</td>
      <td>454.52881</td>
      <td>499.99237</td>
      <td>346.00067</td>
      <td>1.497424</td>
      <td>429.627375</td>
      <td>-1.09564</td>
      <td>0.58630</td>
      <td>-0.14954</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>-96.00830</td>
      <td>0.996360</td>
      <td>3.376517</td>
      <td>-0.25812</td>
      <td>1.99994</td>
      <td>1.00934</td>
      <td>204.52881</td>
      <td>249.99237</td>
      <td>249.99237</td>
      <td>2.493783</td>
      <td>433.003892</td>
      <td>-0.626882</td>
      <td>0.968229</td>
      <td>0.208074</td>
      <td>4.838796</td>
      <td>-31.731239</td>
      <td>26.545012</td>
      <td>1.212982</td>
      <td>111.884180</td>
      <td>0.626882</td>
      <td>0.968229</td>
      <td>0.234958</td>
      <td>49.131541</td>
      <td>77.917245</td>
      <td>54.566016</td>
      <td>1.212982</td>
      <td>111.884180</td>
      <td>-0.415063</td>
      <td>4.172078</td>
      <td>1.113699</td>
      <td>3.100853</td>
      <td>0.852833</td>
      <td>0.890855</td>
      <td>4.924122</td>
      <td>0.168921</td>
      <td>2.584937</td>
      <td>7.172078</td>
      <td>4.113699</td>
      <td>6.100853</td>
      <td>3.852833</td>
      <td>3.890855</td>
      <td>7.924122</td>
      <td>3.168921</td>
      <td>-1.158826</td>
      <td>3.440813</td>
      <td>2.449612</td>
      <td>-1.506678</td>
      <td>0.657711</td>
      <td>2.281571</td>
      <td>3.774639</td>
      <td>2.239957</td>
      <td>0.246527</td>
      <td>0.000580</td>
      <td>0.014301</td>
      <td>0.131893</td>
      <td>0.510724</td>
      <td>0.022515</td>
      <td>0.000160</td>
      <td>0.025094</td>
      <td>2.519646</td>
      <td>1.834403</td>
      <td>0.231693</td>
      <td>0.334239</td>
      <td>0.307701</td>
      <td>92.567166</td>
      <td>117.046241</td>
      <td>93.648551</td>
      <td>0.403108</td>
      <td>142.321492</td>
      <td>2.822352</td>
      <td>0.800700</td>
      <td>2.258248</td>
      <td>2.717333</td>
      <td>1.358881</td>
      <td>1.773272</td>
      <td>0.20435</td>
      <td>0.12494</td>
      <td>0.23621</td>
      <td>12.02393</td>
      <td>109.37500</td>
      <td>30.67779</td>
      <td>0.174117</td>
      <td>135.935660</td>
      <td>-1</td>
      <td>-1</td>
      <td>-3</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>697</th>
      <td>31</td>
      <td>44</td>
      <td>p5</td>
      <td>fu</td>
      <td>2.16339</td>
      <td>1.33002</td>
      <td>0.98913</td>
      <td>459.74731</td>
      <td>388.37433</td>
      <td>279.32739</td>
      <td>1.316516</td>
      <td>391.048565</td>
      <td>-1.73987</td>
      <td>0.66992</td>
      <td>-0.10577</td>
      <td>-250.00000</td>
      <td>-195.91522</td>
      <td>-29.33502</td>
      <td>0.884168</td>
      <td>11.489252</td>
      <td>0.42352</td>
      <td>1.99994</td>
      <td>0.88336</td>
      <td>209.74731</td>
      <td>192.45911</td>
      <td>249.99237</td>
      <td>2.200684</td>
      <td>402.537817</td>
      <td>-0.404499</td>
      <td>0.982395</td>
      <td>0.140319</td>
      <td>0.117962</td>
      <td>-23.827772</td>
      <td>32.272927</td>
      <td>1.172342</td>
      <td>100.390645</td>
      <td>0.508248</td>
      <td>0.982395</td>
      <td>0.170067</td>
      <td>51.334672</td>
      <td>63.054598</td>
      <td>46.357962</td>
      <td>1.172342</td>
      <td>100.390645</td>
      <td>2.287888</td>
      <td>5.121102</td>
      <td>2.253189</td>
      <td>2.909120</td>
      <td>0.874133</td>
      <td>1.642037</td>
      <td>1.609205</td>
      <td>0.912965</td>
      <td>5.287888</td>
      <td>8.121102</td>
      <td>5.253189</td>
      <td>5.909120</td>
      <td>3.874133</td>
      <td>4.642037</td>
      <td>4.609205</td>
      <td>3.912965</td>
      <td>-1.881949</td>
      <td>3.720042</td>
      <td>2.976412</td>
      <td>-1.148191</td>
      <td>0.489399</td>
      <td>3.008563</td>
      <td>3.103867</td>
      <td>2.690862</td>
      <td>0.059843</td>
      <td>0.000199</td>
      <td>0.002916</td>
      <td>0.250890</td>
      <td>0.624559</td>
      <td>0.002625</td>
      <td>0.001910</td>
      <td>0.007127</td>
      <td>2.509387</td>
      <td>1.955879</td>
      <td>0.486191</td>
      <td>0.318682</td>
      <td>0.263602</td>
      <td>93.439821</td>
      <td>90.904384</td>
      <td>84.176649</td>
      <td>0.425967</td>
      <td>124.945397</td>
      <td>2.180151</td>
      <td>0.872698</td>
      <td>1.776374</td>
      <td>2.553869</td>
      <td>1.416671</td>
      <td>1.758194</td>
      <td>0.29254</td>
      <td>0.11310</td>
      <td>0.19110</td>
      <td>31.02875</td>
      <td>75.94299</td>
      <td>15.40375</td>
      <td>0.093831</td>
      <td>74.923886</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>2</td>
      <td>-2</td>
      <td>-1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>698</th>
      <td>45</td>
      <td>58</td>
      <td>p5</td>
      <td>fu</td>
      <td>1.20673</td>
      <td>1.58637</td>
      <td>1.82086</td>
      <td>349.67804</td>
      <td>198.05146</td>
      <td>295.71533</td>
      <td>1.767802</td>
      <td>334.061223</td>
      <td>-1.63556</td>
      <td>0.41357</td>
      <td>-0.28125</td>
      <td>-250.00000</td>
      <td>-185.66132</td>
      <td>-78.63617</td>
      <td>0.866762</td>
      <td>6.420334</td>
      <td>-0.42883</td>
      <td>1.99994</td>
      <td>1.53961</td>
      <td>99.67804</td>
      <td>12.39014</td>
      <td>217.07916</td>
      <td>2.634563</td>
      <td>340.481557</td>
      <td>-0.588932</td>
      <td>0.995665</td>
      <td>0.207407</td>
      <td>-1.652644</td>
      <td>-62.042236</td>
      <td>24.875346</td>
      <td>1.265225</td>
      <td>103.131326</td>
      <td>0.588932</td>
      <td>0.995665</td>
      <td>0.258441</td>
      <td>40.888859</td>
      <td>65.362784</td>
      <td>50.989005</td>
      <td>1.265225</td>
      <td>103.131326</td>
      <td>6.470940</td>
      <td>3.438881</td>
      <td>2.431687</td>
      <td>4.841932</td>
      <td>-1.382371</td>
      <td>0.841579</td>
      <td>2.929300</td>
      <td>-0.824248</td>
      <td>9.470940</td>
      <td>6.438881</td>
      <td>5.431687</td>
      <td>7.841932</td>
      <td>1.617629</td>
      <td>3.841579</td>
      <td>5.929300</td>
      <td>2.175752</td>
      <td>-4.124948</td>
      <td>2.699335</td>
      <td>3.136805</td>
      <td>-3.446421</td>
      <td>-1.209956</td>
      <td>2.420720</td>
      <td>3.273037</td>
      <td>1.554218</td>
      <td>0.000037</td>
      <td>0.006948</td>
      <td>0.001708</td>
      <td>0.000568</td>
      <td>0.226296</td>
      <td>0.015490</td>
      <td>0.001064</td>
      <td>0.120133</td>
      <td>2.507226</td>
      <td>1.935432</td>
      <td>0.314480</td>
      <td>0.341741</td>
      <td>0.473857</td>
      <td>78.873198</td>
      <td>77.652799</td>
      <td>86.202744</td>
      <td>0.469723</td>
      <td>116.260631</td>
      <td>2.610048</td>
      <td>1.017748</td>
      <td>1.908615</td>
      <td>1.725933</td>
      <td>1.896395</td>
      <td>2.052608</td>
      <td>0.10645</td>
      <td>0.06219</td>
      <td>0.19422</td>
      <td>13.09204</td>
      <td>141.96777</td>
      <td>26.34430</td>
      <td>0.168585</td>
      <td>175.439030</td>
      <td>-6</td>
      <td>1</td>
      <td>-1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>-4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>699</th>
      <td>69</td>
      <td>82</td>
      <td>p5</td>
      <td>fu</td>
      <td>0.92609</td>
      <td>1.60431</td>
      <td>0.99548</td>
      <td>349.95270</td>
      <td>354.40063</td>
      <td>277.06909</td>
      <td>1.452005</td>
      <td>359.364040</td>
      <td>-1.17041</td>
      <td>0.39563</td>
      <td>-0.11682</td>
      <td>-250.00000</td>
      <td>-196.02203</td>
      <td>-67.37518</td>
      <td>0.787704</td>
      <td>3.341512</td>
      <td>-0.24432</td>
      <td>1.99994</td>
      <td>0.87866</td>
      <td>99.95270</td>
      <td>158.37860</td>
      <td>209.69391</td>
      <td>2.239709</td>
      <td>362.705551</td>
      <td>-0.501235</td>
      <td>0.922058</td>
      <td>0.165711</td>
      <td>-3.267728</td>
      <td>-23.279043</td>
      <td>28.847328</td>
      <td>1.129473</td>
      <td>98.632700</td>
      <td>0.501235</td>
      <td>0.922058</td>
      <td>0.206182</td>
      <td>40.539082</td>
      <td>64.543503</td>
      <td>46.970074</td>
      <td>1.129473</td>
      <td>98.632700</td>
      <td>2.567992</td>
      <td>4.446137</td>
      <td>0.555225</td>
      <td>4.419368</td>
      <td>-0.147539</td>
      <td>0.561026</td>
      <td>5.513202</td>
      <td>0.431971</td>
      <td>5.567992</td>
      <td>7.446137</td>
      <td>3.555225</td>
      <td>7.419368</td>
      <td>2.852461</td>
      <td>3.561026</td>
      <td>8.513202</td>
      <td>3.431971</td>
      <td>-3.037800</td>
      <td>3.209248</td>
      <td>2.429445</td>
      <td>-3.223296</td>
      <td>-0.197750</td>
      <td>2.341897</td>
      <td>3.860129</td>
      <td>2.180814</td>
      <td>0.002383</td>
      <td>0.001331</td>
      <td>0.015122</td>
      <td>0.001267</td>
      <td>0.843241</td>
      <td>0.019186</td>
      <td>0.000113</td>
      <td>0.029197</td>
      <td>2.526650</td>
      <td>1.987748</td>
      <td>0.232413</td>
      <td>0.348783</td>
      <td>0.298481</td>
      <td>79.775717</td>
      <td>89.222231</td>
      <td>77.034156</td>
      <td>0.343417</td>
      <td>109.157608</td>
      <td>2.619728</td>
      <td>1.045836</td>
      <td>2.038466</td>
      <td>2.449268</td>
      <td>1.464202</td>
      <td>1.983248</td>
      <td>0.03967</td>
      <td>0.20886</td>
      <td>0.23236</td>
      <td>13.49640</td>
      <td>97.22138</td>
      <td>9.20868</td>
      <td>0.064975</td>
      <td>137.515372</td>
      <td>-3</td>
      <td>1</td>
      <td>-2</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>700</th>
      <td>81</td>
      <td>94</td>
      <td>p5</td>
      <td>fu</td>
      <td>1.22577</td>
      <td>1.51904</td>
      <td>1.55267</td>
      <td>372.13898</td>
      <td>344.24591</td>
      <td>322.64709</td>
      <td>1.799844</td>
      <td>387.426092</td>
      <td>-1.26520</td>
      <td>0.48090</td>
      <td>-0.37671</td>
      <td>-250.00000</td>
      <td>-172.08862</td>
      <td>-72.65472</td>
      <td>0.520543</td>
      <td>5.809610</td>
      <td>-0.03943</td>
      <td>1.99994</td>
      <td>1.17596</td>
      <td>122.13898</td>
      <td>172.15729</td>
      <td>249.99237</td>
      <td>2.320387</td>
      <td>393.235702</td>
      <td>-0.430428</td>
      <td>0.932721</td>
      <td>0.106164</td>
      <td>-11.193495</td>
      <td>-20.749017</td>
      <td>29.621416</td>
      <td>1.116082</td>
      <td>96.506893</td>
      <td>0.430428</td>
      <td>0.932721</td>
      <td>0.217979</td>
      <td>44.254597</td>
      <td>55.661126</td>
      <td>51.328804</td>
      <td>1.116082</td>
      <td>96.506893</td>
      <td>3.461460</td>
      <td>4.403753</td>
      <td>2.382071</td>
      <td>2.991704</td>
      <td>0.925268</td>
      <td>1.605117</td>
      <td>3.503179</td>
      <td>1.316625</td>
      <td>6.461460</td>
      <td>7.403753</td>
      <td>5.382071</td>
      <td>5.991704</td>
      <td>3.925268</td>
      <td>4.605117</td>
      <td>6.503179</td>
      <td>4.316625</td>
      <td>-2.909900</td>
      <td>3.221791</td>
      <td>2.920885</td>
      <td>-2.758182</td>
      <td>0.858543</td>
      <td>2.739131</td>
      <td>3.060898</td>
      <td>2.550360</td>
      <td>0.003615</td>
      <td>0.001274</td>
      <td>0.003490</td>
      <td>0.005812</td>
      <td>0.390592</td>
      <td>0.006160</td>
      <td>0.002207</td>
      <td>0.010761</td>
      <td>2.507606</td>
      <td>1.992736</td>
      <td>0.280705</td>
      <td>0.345201</td>
      <td>0.380437</td>
      <td>82.764348</td>
      <td>79.572164</td>
      <td>83.421737</td>
      <td>0.404426</td>
      <td>110.727218</td>
      <td>2.394218</td>
      <td>1.081624</td>
      <td>1.474020</td>
      <td>2.455259</td>
      <td>1.168834</td>
      <td>2.069957</td>
      <td>0.08435</td>
      <td>0.09619</td>
      <td>0.10676</td>
      <td>7.56836</td>
      <td>46.14257</td>
      <td>49.88861</td>
      <td>0.061489</td>
      <td>116.900698</td>
      <td>-2</td>
      <td>-1</td>
      <td>-2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>-1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>701</th>
      <td>94</td>
      <td>107</td>
      <td>p5</td>
      <td>fu</td>
      <td>1.66601</td>
      <td>1.66358</td>
      <td>1.50628</td>
      <td>432.15179</td>
      <td>292.90771</td>
      <td>277.49634</td>
      <td>2.076707</td>
      <td>362.938980</td>
      <td>-1.52429</td>
      <td>0.33636</td>
      <td>-0.11395</td>
      <td>-250.00000</td>
      <td>-137.80212</td>
      <td>-27.50397</td>
      <td>0.379642</td>
      <td>3.567399</td>
      <td>0.14172</td>
      <td>1.99994</td>
      <td>1.39233</td>
      <td>182.15179</td>
      <td>155.10559</td>
      <td>249.99237</td>
      <td>2.456349</td>
      <td>366.506379</td>
      <td>-0.442868</td>
      <td>0.956378</td>
      <td>0.196504</td>
      <td>-3.515977</td>
      <td>-17.043481</td>
      <td>34.084027</td>
      <td>1.148788</td>
      <td>101.025612</td>
      <td>0.464671</td>
      <td>0.956378</td>
      <td>0.250658</td>
      <td>49.459015</td>
      <td>58.438225</td>
      <td>47.829850</td>
      <td>1.148788</td>
      <td>101.025612</td>
      <td>4.259990</td>
      <td>4.276866</td>
      <td>4.278747</td>
      <td>2.760766</td>
      <td>-0.063817</td>
      <td>1.522542</td>
      <td>2.345717</td>
      <td>0.305391</td>
      <td>7.259990</td>
      <td>7.276866</td>
      <td>7.278747</td>
      <td>5.760766</td>
      <td>2.936183</td>
      <td>4.522542</td>
      <td>5.345717</td>
      <td>3.305391</td>
      <td>-3.034476</td>
      <td>2.850310</td>
      <td>3.493072</td>
      <td>-1.690301</td>
      <td>0.584845</td>
      <td>3.057354</td>
      <td>2.554324</td>
      <td>2.325555</td>
      <td>0.002410</td>
      <td>0.004368</td>
      <td>0.000477</td>
      <td>0.090970</td>
      <td>0.558652</td>
      <td>0.002233</td>
      <td>0.010639</td>
      <td>0.020042</td>
      <td>2.488405</td>
      <td>1.962664</td>
      <td>0.354359</td>
      <td>0.344662</td>
      <td>0.384934</td>
      <td>90.004129</td>
      <td>77.859377</td>
      <td>92.934915</td>
      <td>0.471409</td>
      <td>118.566915</td>
      <td>2.439376</td>
      <td>0.859855</td>
      <td>1.872985</td>
      <td>1.802575</td>
      <td>2.138398</td>
      <td>1.722939</td>
      <td>0.13483</td>
      <td>0.07702</td>
      <td>0.35669</td>
      <td>16.20483</td>
      <td>70.03784</td>
      <td>13.66425</td>
      <td>0.081454</td>
      <td>134.898264</td>
      <td>-1</td>
      <td>-1</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>-1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>702</th>
      <td>119</td>
      <td>132</td>
      <td>p5</td>
      <td>fu</td>
      <td>1.41938</td>
      <td>1.71655</td>
      <td>1.34949</td>
      <td>357.39899</td>
      <td>385.08606</td>
      <td>297.56164</td>
      <td>2.493884</td>
      <td>429.484652</td>
      <td>-1.67200</td>
      <td>0.28339</td>
      <td>-0.10620</td>
      <td>-250.00000</td>
      <td>-135.09369</td>
      <td>-47.56927</td>
      <td>0.394215</td>
      <td>3.519239</td>
      <td>-0.25262</td>
      <td>1.99994</td>
      <td>1.24329</td>
      <td>107.39899</td>
      <td>249.99237</td>
      <td>249.99237</td>
      <td>2.888099</td>
      <td>433.003892</td>
      <td>-0.580989</td>
      <td>0.890381</td>
      <td>0.186539</td>
      <td>1.698422</td>
      <td>-8.527316</td>
      <td>30.921350</td>
      <td>1.118666</td>
      <td>91.953486</td>
      <td>0.580989</td>
      <td>0.890381</td>
      <td>0.202878</td>
      <td>41.296152</td>
      <td>59.456458</td>
      <td>46.621468</td>
      <td>1.118666</td>
      <td>91.953486</td>
      <td>4.911958</td>
      <td>3.452469</td>
      <td>3.495880</td>
      <td>4.970360</td>
      <td>2.191444</td>
      <td>1.961507</td>
      <td>5.673185</td>
      <td>2.696652</td>
      <td>7.911958</td>
      <td>6.452469</td>
      <td>6.495880</td>
      <td>7.970360</td>
      <td>5.191444</td>
      <td>4.961507</td>
      <td>8.673185</td>
      <td>5.696652</td>
      <td>-3.738362</td>
      <td>2.608093</td>
      <td>3.468649</td>
      <td>-3.461613</td>
      <td>2.280737</td>
      <td>2.811293</td>
      <td>3.744180</td>
      <td>2.978898</td>
      <td>0.000185</td>
      <td>0.009105</td>
      <td>0.000523</td>
      <td>0.000537</td>
      <td>0.022564</td>
      <td>0.004934</td>
      <td>0.000181</td>
      <td>0.002893</td>
      <td>2.469384</td>
      <td>1.881457</td>
      <td>0.344061</td>
      <td>0.376966</td>
      <td>0.353406</td>
      <td>79.677244</td>
      <td>92.591841</td>
      <td>78.754135</td>
      <td>0.546887</td>
      <td>117.047905</td>
      <td>2.876548</td>
      <td>0.859518</td>
      <td>2.475066</td>
      <td>2.660137</td>
      <td>1.000725</td>
      <td>2.198701</td>
      <td>0.11212</td>
      <td>0.08630</td>
      <td>0.10022</td>
      <td>21.04187</td>
      <td>64.44549</td>
      <td>27.54974</td>
      <td>0.038353</td>
      <td>144.247824</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>-1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>703</th>
      <td>132</td>
      <td>145</td>
      <td>p5</td>
      <td>fu</td>
      <td>1.35834</td>
      <td>2.05310</td>
      <td>0.96545</td>
      <td>372.93243</td>
      <td>271.50726</td>
      <td>259.23157</td>
      <td>1.876420</td>
      <td>300.915973</td>
      <td>-1.69641</td>
      <td>-0.05316</td>
      <td>-0.10675</td>
      <td>-250.00000</td>
      <td>-146.39282</td>
      <td>-75.74463</td>
      <td>0.883097</td>
      <td>2.029390</td>
      <td>-0.33807</td>
      <td>1.99994</td>
      <td>0.85870</td>
      <td>122.93243</td>
      <td>125.11444</td>
      <td>183.48694</td>
      <td>2.759517</td>
      <td>302.945363</td>
      <td>-0.631605</td>
      <td>0.939270</td>
      <td>0.195021</td>
      <td>2.792358</td>
      <td>-33.020606</td>
      <td>15.280503</td>
      <td>1.220235</td>
      <td>94.481899</td>
      <td>0.631605</td>
      <td>0.947448</td>
      <td>0.242244</td>
      <td>44.616699</td>
      <td>60.546288</td>
      <td>41.086637</td>
      <td>1.220235</td>
      <td>94.481899</td>
      <td>4.643195</td>
      <td>3.180711</td>
      <td>0.034637</td>
      <td>4.631110</td>
      <td>-0.733157</td>
      <td>1.242598</td>
      <td>6.995153</td>
      <td>-0.664444</td>
      <td>7.643195</td>
      <td>6.180711</td>
      <td>3.034637</td>
      <td>7.631110</td>
      <td>2.266843</td>
      <td>4.242598</td>
      <td>9.995153</td>
      <td>2.335556</td>
      <td>-3.685379</td>
      <td>0.474220</td>
      <td>2.051144</td>
      <td>-3.287507</td>
      <td>0.125731</td>
      <td>2.209831</td>
      <td>4.225476</td>
      <td>1.512740</td>
      <td>0.000228</td>
      <td>0.635343</td>
      <td>0.040253</td>
      <td>0.001011</td>
      <td>0.899945</td>
      <td>0.027117</td>
      <td>0.000024</td>
      <td>0.130346</td>
      <td>2.510734</td>
      <td>2.034631</td>
      <td>0.338826</td>
      <td>0.408471</td>
      <td>0.310402</td>
      <td>81.201132</td>
      <td>77.584624</td>
      <td>65.483865</td>
      <td>0.456305</td>
      <td>96.467386</td>
      <td>2.832273</td>
      <td>1.025408</td>
      <td>2.270798</td>
      <td>2.398363</td>
      <td>1.391613</td>
      <td>1.924570</td>
      <td>0.11004</td>
      <td>0.10571</td>
      <td>0.34125</td>
      <td>18.75306</td>
      <td>134.35364</td>
      <td>32.73010</td>
      <td>0.100989</td>
      <td>147.711395</td>
      <td>-1</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>704</th>
      <td>149</td>
      <td>162</td>
      <td>p5</td>
      <td>fu</td>
      <td>1.42700</td>
      <td>1.97888</td>
      <td>1.78961</td>
      <td>364.83765</td>
      <td>210.99090</td>
      <td>394.41681</td>
      <td>1.454818</td>
      <td>341.643571</td>
      <td>-1.40851</td>
      <td>0.02106</td>
      <td>-0.52997</td>
      <td>-250.00000</td>
      <td>-155.10559</td>
      <td>-238.99078</td>
      <td>0.703426</td>
      <td>5.156961</td>
      <td>0.01849</td>
      <td>1.99994</td>
      <td>1.25964</td>
      <td>114.83765</td>
      <td>55.88531</td>
      <td>155.42603</td>
      <td>2.158244</td>
      <td>346.800531</td>
      <td>-0.508391</td>
      <td>0.823266</td>
      <td>0.204327</td>
      <td>4.591135</td>
      <td>-40.924072</td>
      <td>-15.543425</td>
      <td>1.142790</td>
      <td>96.515313</td>
      <td>0.511235</td>
      <td>0.823266</td>
      <td>0.285861</td>
      <td>44.097313</td>
      <td>55.938720</td>
      <td>42.921213</td>
      <td>1.142790</td>
      <td>96.515313</td>
      <td>4.273535</td>
      <td>1.560028</td>
      <td>2.889112</td>
      <td>4.528096</td>
      <td>-1.093038</td>
      <td>3.002996</td>
      <td>3.608717</td>
      <td>0.778120</td>
      <td>7.273535</td>
      <td>4.560028</td>
      <td>5.889112</td>
      <td>7.528096</td>
      <td>1.906962</td>
      <td>6.002996</td>
      <td>6.608717</td>
      <td>3.778120</td>
      <td>-2.877333</td>
      <td>0.998138</td>
      <td>2.040874</td>
      <td>-3.255157</td>
      <td>-0.426850</td>
      <td>-1.677504</td>
      <td>3.193002</td>
      <td>2.166933</td>
      <td>0.004011</td>
      <td>0.318212</td>
      <td>0.041263</td>
      <td>0.001133</td>
      <td>0.669489</td>
      <td>0.093444</td>
      <td>0.001408</td>
      <td>0.030240</td>
      <td>2.526799</td>
      <td>2.068763</td>
      <td>0.296889</td>
      <td>0.460599</td>
      <td>0.376320</td>
      <td>82.014933</td>
      <td>60.600476</td>
      <td>80.972205</td>
      <td>0.337523</td>
      <td>97.865294</td>
      <td>2.379215</td>
      <td>1.491720</td>
      <td>1.843796</td>
      <td>2.046006</td>
      <td>1.682529</td>
      <td>0.572162</td>
      <td>0.05451</td>
      <td>0.19342</td>
      <td>0.26959</td>
      <td>29.85382</td>
      <td>92.51404</td>
      <td>18.52417</td>
      <td>0.117532</td>
      <td>124.772386</td>
      <td>-2</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>-3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>705</th>
      <td>179</td>
      <td>192</td>
      <td>p5</td>
      <td>fu</td>
      <td>2.07898</td>
      <td>1.41156</td>
      <td>2.10010</td>
      <td>358.79517</td>
      <td>499.99237</td>
      <td>301.77307</td>
      <td>1.624224</td>
      <td>354.621856</td>
      <td>-1.45007</td>
      <td>0.27551</td>
      <td>-0.94806</td>
      <td>-250.00000</td>
      <td>-250.00000</td>
      <td>-56.64062</td>
      <td>0.513281</td>
      <td>3.434487</td>
      <td>0.62891</td>
      <td>1.68707</td>
      <td>1.15204</td>
      <td>108.79517</td>
      <td>249.99237</td>
      <td>245.13245</td>
      <td>2.137505</td>
      <td>358.056343</td>
      <td>-0.345575</td>
      <td>0.912414</td>
      <td>0.171918</td>
      <td>-4.587613</td>
      <td>-38.396395</td>
      <td>18.143875</td>
      <td>1.113144</td>
      <td>109.730846</td>
      <td>0.442330</td>
      <td>0.912414</td>
      <td>0.317774</td>
      <td>35.386893</td>
      <td>88.165871</td>
      <td>33.497736</td>
      <td>1.113144</td>
      <td>109.730846</td>
      <td>3.402980</td>
      <td>2.243080</td>
      <td>3.166331</td>
      <td>5.410802</td>
      <td>0.402882</td>
      <td>5.686640</td>
      <td>1.117388</td>
      <td>-0.567754</td>
      <td>6.402980</td>
      <td>5.243080</td>
      <td>6.166331</td>
      <td>8.410802</td>
      <td>3.402882</td>
      <td>8.686640</td>
      <td>4.117388</td>
      <td>2.432246</td>
      <td>-0.887520</td>
      <td>0.871530</td>
      <td>-0.901757</td>
      <td>-3.550241</td>
      <td>0.351213</td>
      <td>3.822855</td>
      <td>2.388114</td>
      <td>1.822612</td>
      <td>0.374799</td>
      <td>0.383465</td>
      <td>0.367186</td>
      <td>0.000385</td>
      <td>0.725428</td>
      <td>0.000132</td>
      <td>0.016935</td>
      <td>0.068362</td>
      <td>2.500278</td>
      <td>1.899956</td>
      <td>0.411655</td>
      <td>0.298715</td>
      <td>0.420159</td>
      <td>76.769068</td>
      <td>126.006822</td>
      <td>69.909683</td>
      <td>0.421440</td>
      <td>128.227312</td>
      <td>2.079732</td>
      <td>1.052357</td>
      <td>1.090171</td>
      <td>2.405752</td>
      <td>2.228339</td>
      <td>1.306512</td>
      <td>0.03949</td>
      <td>0.04388</td>
      <td>0.05988</td>
      <td>12.18414</td>
      <td>107.02515</td>
      <td>16.38031</td>
      <td>0.051038</td>
      <td>144.465768</td>
      <td>-2</td>
      <td>-1</td>
      <td>-2</td>
      <td>2</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>706</th>
      <td>192</td>
      <td>205</td>
      <td>p5</td>
      <td>fu</td>
      <td>0.54926</td>
      <td>1.82813</td>
      <td>1.64869</td>
      <td>363.48724</td>
      <td>493.01148</td>
      <td>165.92407</td>
      <td>1.179169</td>
      <td>343.918505</td>
      <td>-0.68732</td>
      <td>0.17181</td>
      <td>-0.49921</td>
      <td>-250.00000</td>
      <td>-244.11774</td>
      <td>-97.43500</td>
      <td>0.952884</td>
      <td>15.440433</td>
      <td>-0.13806</td>
      <td>1.99994</td>
      <td>1.14948</td>
      <td>113.48724</td>
      <td>248.89374</td>
      <td>68.48907</td>
      <td>2.132052</td>
      <td>359.358938</td>
      <td>-0.443595</td>
      <td>0.949867</td>
      <td>0.157983</td>
      <td>-5.881677</td>
      <td>-36.972633</td>
      <td>-2.544110</td>
      <td>1.146511</td>
      <td>104.739716</td>
      <td>0.443595</td>
      <td>0.949867</td>
      <td>0.234785</td>
      <td>40.908812</td>
      <td>81.754832</td>
      <td>32.257667</td>
      <td>1.146511</td>
      <td>104.739716</td>
      <td>0.801818</td>
      <td>2.858941</td>
      <td>3.392843</td>
      <td>4.515269</td>
      <td>1.176139</td>
      <td>0.157434</td>
      <td>6.260796</td>
      <td>0.112243</td>
      <td>3.801818</td>
      <td>5.858941</td>
      <td>6.392843</td>
      <td>7.515269</td>
      <td>4.176139</td>
      <td>3.157434</td>
      <td>9.260796</td>
      <td>3.112243</td>
      <td>0.615347</td>
      <td>1.670843</td>
      <td>2.310968</td>
      <td>-3.199000</td>
      <td>1.018922</td>
      <td>-1.302970</td>
      <td>4.062492</td>
      <td>2.149780</td>
      <td>0.538325</td>
      <td>0.094753</td>
      <td>0.020835</td>
      <td>0.001379</td>
      <td>0.308240</td>
      <td>0.192585</td>
      <td>0.000049</td>
      <td>0.031573</td>
      <td>2.536813</td>
      <td>2.109725</td>
      <td>0.128364</td>
      <td>0.380622</td>
      <td>0.343315</td>
      <td>78.886077</td>
      <td>114.088874</td>
      <td>42.194626</td>
      <td>0.297843</td>
      <td>107.038801</td>
      <td>2.783383</td>
      <td>1.474828</td>
      <td>1.838763</td>
      <td>2.483829</td>
      <td>1.012093</td>
      <td>1.861385</td>
      <td>0.08203</td>
      <td>0.06885</td>
      <td>0.20020</td>
      <td>23.31543</td>
      <td>74.66889</td>
      <td>44.54803</td>
      <td>0.156177</td>
      <td>136.501610</td>
      <td>-2</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>707</th>
      <td>241</td>
      <td>254</td>
      <td>p5</td>
      <td>fu</td>
      <td>1.11701</td>
      <td>1.63764</td>
      <td>1.19897</td>
      <td>371.39893</td>
      <td>455.12390</td>
      <td>304.80957</td>
      <td>1.353696</td>
      <td>401.041711</td>
      <td>-1.40381</td>
      <td>0.36230</td>
      <td>-0.11981</td>
      <td>-250.00000</td>
      <td>-205.13153</td>
      <td>-63.26294</td>
      <td>0.844207</td>
      <td>1.949278</td>
      <td>-0.28680</td>
      <td>1.99994</td>
      <td>1.07916</td>
      <td>121.39893</td>
      <td>249.99237</td>
      <td>241.54663</td>
      <td>2.197902</td>
      <td>402.990989</td>
      <td>-0.551857</td>
      <td>0.963384</td>
      <td>0.195567</td>
      <td>-0.198950</td>
      <td>-22.943935</td>
      <td>29.769898</td>
      <td>1.204626</td>
      <td>112.685276</td>
      <td>0.551857</td>
      <td>0.963384</td>
      <td>0.221352</td>
      <td>38.674576</td>
      <td>80.098665</td>
      <td>54.600642</td>
      <td>1.204626</td>
      <td>112.685276</td>
      <td>4.500257</td>
      <td>3.822534</td>
      <td>1.767414</td>
      <td>5.226224</td>
      <td>0.340941</td>
      <td>0.932627</td>
      <td>2.701235</td>
      <td>-0.106865</td>
      <td>7.500257</td>
      <td>6.822534</td>
      <td>4.767414</td>
      <td>8.226224</td>
      <td>3.340941</td>
      <td>3.932627</td>
      <td>5.701235</td>
      <td>2.893135</td>
      <td>-3.558642</td>
      <td>2.771928</td>
      <td>2.784480</td>
      <td>-3.444438</td>
      <td>0.729491</td>
      <td>2.545526</td>
      <td>3.105804</td>
      <td>1.945050</td>
      <td>0.000373</td>
      <td>0.005573</td>
      <td>0.005361</td>
      <td>0.000572</td>
      <td>0.465701</td>
      <td>0.010911</td>
      <td>0.001898</td>
      <td>0.051769</td>
      <td>2.529349</td>
      <td>1.957082</td>
      <td>0.272123</td>
      <td>0.347257</td>
      <td>0.321865</td>
      <td>78.769760</td>
      <td>116.284978</td>
      <td>86.471981</td>
      <td>0.343529</td>
      <td>126.170574</td>
      <td>2.627626</td>
      <td>1.052050</td>
      <td>1.938188</td>
      <td>2.495380</td>
      <td>1.372489</td>
      <td>2.022928</td>
      <td>0.23480</td>
      <td>0.16956</td>
      <td>0.23352</td>
      <td>20.19501</td>
      <td>84.08356</td>
      <td>50.77362</td>
      <td>0.147301</td>
      <td>167.058223</td>
      <td>-2</td>
      <td>1</td>
      <td>-2</td>
      <td>2</td>
      <td>-2</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>708 rows × 108 columns</p>
</div>




```python
# Total number of features
len(features)
```




    104




```python
# Save all the features in a txt file for later use.
with open('data/features.txt', 'w') as f:
    for feature in features:
        f.write("%s\n" %feature)
```


```python
# Save X_y as csv file for using in (classical) ML models
X_y.to_csv('data/X_y.csv', index=False) 
```

## Classical ML models

Now it's time for some machine learning model stuff! We'll load this saved X_y. Then apply some classical machine learning models and asses their performance by plotting the confusion matrix. For paramter tuning in each of the model, we used grid searching cross-validation.


```python
# Read Features 
with open('data/features.txt') as f:
    features = f.read().strip().split("\n")
f.close()

# Load data
X_y = pd.read_csv('data/X_y.csv')
X_y = X_y.dropna()
shot_labels = X_y.ShotName.unique()

# Train Test split Randomly:
from sklearn.model_selection import train_test_split
train, test = train_test_split(X_y, test_size=0.2, random_state=42)

X_train = train[features].values
Y_train = train["ShotName"].values
X_test  = test[features].values
Y_test  = test["ShotName"].values
```


```python
# Helper function for plotting confusion matrix
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, shots,
                          model_name,
                          normalize=False,
                          cmap=plt.cm.Wistia):
    tick_marks = np.arange(len(shots))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.yticks(tick_marks, shots)
    plt.title("Confusion matrix - " + model_name)
    plt.colorbar()
    plt.xticks(tick_marks, shots, rotation='vertical')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black")
    plt.tight_layout()
    plt.ylabel('True Shot')
    plt.xlabel('Predicted Shot')
    plt.savefig("plots/" + "Confusion matrix - " + model_name)
```

### Logistic Regression


```python
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# Save the hyperparameters ie C value and loss-function type:
parameters = {'C':[0.01,0.1,1,10,20,30], 'penalty':['l2','l1']}
log_reg_clf = linear_model.LogisticRegression()
log_reg_model = GridSearchCV(log_reg_clf, param_grid=parameters, cv=3,verbose=1, n_jobs=8)

log_reg_model.fit(X_train,Y_train)
y_pred = log_reg_model.predict(X_test)
# y_prob = log_reg_model.predict_proba(X_test)
# print(y_prob)
accuracy = metrics.accuracy_score(y_true=Y_test,y_pred=y_pred)

# Accuracy of our stroke detectiony
print('Accuracy of strokes detection:   {}\n\n'.format(accuracy))
     
# confusion matrix
cm = metrics.confusion_matrix(Y_test, y_pred)

# plot confusion matrix
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, model_name="Logistic Regression", 
                      shots=shot_labels, normalize=True)
plt.show()
    
# get classification report
print("Classifiction Report for this model")
classification_report = metrics.classification_report(Y_test, y_pred)
print(classification_report)
```

    Fitting 3 folds for each of 12 candidates, totalling 36 fits


    [Parallel(n_jobs=8)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  21 out of  36 | elapsed:    2.7s remaining:    1.9s
    [Parallel(n_jobs=8)]: Done  36 out of  36 | elapsed:   11.3s finished
    /home/easy/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py:813: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)


    Accuracy of strokes detection:   0.7535211267605634
    
    



![png](output_47_3.png)


    Classifiction Report for this model
                  precision    recall  f1-score   support
    
              bo       0.89      0.76      0.82        42
              bu       0.73      0.90      0.81        21
              fo       0.62      0.64      0.63        28
              fs       0.70      0.66      0.68        32
              fu       0.81      0.89      0.85        19
    
        accuracy                           0.75       142
       macro avg       0.75      0.77      0.76       142
    weighted avg       0.76      0.75      0.75       142
    


### K- Nearest Neighbours


```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_true=Y_test,y_pred=y_pred)

# Accuracy of our stroke detectiony
print('Accuracy of strokes detection:   {}\n\n'.format(accuracy))
     
# confusion matrix
cm = metrics.confusion_matrix(Y_test, y_pred)
    
# plot confusion matrix
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, model_name='KNeighborsClassifier',
                      shots=shot_labels, normalize=True, )
plt.show()
    
# get classification report
print("Classifiction Report for this model")
classification_report = metrics.classification_report(Y_test, y_pred)
print(classification_report)
```

    Accuracy of strokes detection:   0.6197183098591549
    
    



![png](output_49_1.png)


    Classifiction Report for this model
                  precision    recall  f1-score   support
    
              bo       0.60      0.76      0.67        42
              bu       0.58      0.71      0.64        21
              fo       0.48      0.39      0.43        28
              fs       0.64      0.50      0.56        32
              fu       0.93      0.74      0.82        19
    
        accuracy                           0.62       142
       macro avg       0.65      0.62      0.63       142
    weighted avg       0.63      0.62      0.62       142
    


### Linear SVC


```python
from sklearn.svm import LinearSVC
parameters = {'C':[0.125, 0.5, 1, 2, 8, 16]}
lr_svc_reg_clf = LinearSVC(tol=0.00005)
lr_svc_reg_model = GridSearchCV(lr_svc_reg_clf, param_grid=parameters, n_jobs=8, verbose=1)

lr_svc_reg_model.fit(X_train,Y_train)
y_pred = lr_svc_reg_model.predict(X_test)
accuracy = metrics.accuracy_score(y_true=Y_test,y_pred=y_pred)
# Accuracy of our stroke detectiony
print('Accuracy of strokes detection:   {}\n\n'.format(accuracy))
     
# confusion matrix
cm = metrics.confusion_matrix(Y_test, y_pred)
        
# plot confusion matrix
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, model_name='LinearSVC',
                      shots=shot_labels, normalize=True)
plt.show()
    
# get classification report
print("Classifiction Report for this model")
classification_report = metrics.classification_report(Y_test, y_pred)
print(classification_report)
```

    Fitting 3 folds for each of 6 candidates, totalling 18 fits


    [Parallel(n_jobs=8)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  18 out of  18 | elapsed:    1.2s finished
    /home/easy/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py:813: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)
    /home/easy/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    Accuracy of strokes detection:   0.6901408450704225
    
    



![png](output_51_3.png)


    Classifiction Report for this model
                  precision    recall  f1-score   support
    
              bo       0.81      0.69      0.74        42
              bu       0.63      0.81      0.71        21
              fo       0.89      0.29      0.43        28
              fs       0.54      0.78      0.64        32
              fu       0.79      1.00      0.88        19
    
        accuracy                           0.69       142
       macro avg       0.73      0.71      0.68       142
    weighted avg       0.74      0.69      0.67       142
    


### SVC with RBF kernel


```python
from sklearn.svm import SVC
parameters = {'C':[2,8,16],\
              'gamma': [ 0.0078125, 0.125, 2]}
rbf_svm_clf = SVC(kernel='rbf')
rbf_svm_model = GridSearchCV(rbf_svm_clf,param_grid=parameters,n_jobs=8)

rbf_svm_model.fit(X_train,Y_train )
y_pred = rbf_svm_model.predict(X_test)
accuracy = metrics.accuracy_score(y_true=Y_test,y_pred=y_pred)
# Accuracy of our stroke detectiony
print('Accuracy of strokes detection:   {}\n\n'.format(accuracy))
     
# confusion matrix
cm = metrics.confusion_matrix(Y_test, y_pred)

# plot confusion matrix
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, model_name='SVC', shots=shot_labels, normalize=True)
plt.show()
    
# get classification report
print("Classifiction Report for this model")
classification_report = metrics.classification_report(Y_test, y_pred)
print(classification_report)
```

    Accuracy of strokes detection:   0.30985915492957744
    
    



![png](output_53_1.png)


    Classifiction Report for this model
                  precision    recall  f1-score   support
    
              bo       0.30      1.00      0.46        42
              bu       1.00      0.05      0.09        21
              fo       0.00      0.00      0.00        28
              fs       0.00      0.00      0.00        32
              fu       1.00      0.05      0.10        19
    
        accuracy                           0.31       142
       macro avg       0.46      0.22      0.13       142
    weighted avg       0.37      0.31      0.16       142
    


    /home/easy/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)



```python
from sklearn.tree import DecisionTreeClassifier
parameters = {'max_depth':np.arange(3,20,2)}
decision_trees_clf = DecisionTreeClassifier()
decision_trees = GridSearchCV(decision_trees_clf, param_grid=parameters, n_jobs=8)

decision_trees.fit(X_train,Y_train )
y_pred = decision_trees.predict(X_test)
accuracy = metrics.accuracy_score(y_true=Y_test,y_pred=y_pred)
# Accuracy of our stroke detection
print('Accuracy of strokes detection:   {}\n\n'.format(accuracy))
     
# confusion matrix
cm = metrics.confusion_matrix(Y_test, y_pred)

# plot confusion matrix
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, model_name='Decision Tree',
                      shots=shot_labels, normalize=True)
plt.show()
    
# get classification report
print("Classifiction Report for this model")
classification_report = metrics.classification_report(Y_test, y_pred)
print(classification_report)
```

    /home/easy/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py:813: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)


    Accuracy of strokes detection:   0.6690140845070423
    
    



![png](output_54_2.png)


    Classifiction Report for this model
                  precision    recall  f1-score   support
    
              bo       0.82      0.74      0.78        42
              bu       0.64      0.76      0.70        21
              fo       0.57      0.71      0.63        28
              fs       0.68      0.41      0.51        32
              fu       0.60      0.79      0.68        19
    
        accuracy                           0.67       142
       macro avg       0.66      0.68      0.66       142
    weighted avg       0.68      0.67      0.66       142
    


### Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
params = {'n_estimators': np.arange(10,120,20), 'max_depth':np.arange(3,15,2)}
rfclassifier_clf = RandomForestClassifier()
rfclassifier = GridSearchCV(rfclassifier_clf, param_grid=params, n_jobs=8)


rfclassifier.fit(X_train,Y_train )
y_pred = rfclassifier.predict(X_test)
accuracy = metrics.accuracy_score(y_true=Y_test,y_pred=y_pred)
# Accuracy of our stroke detection
print('Accuracy of strokes detection:   {}\n\n'.format(accuracy))
     
# confusion matrix
cm = metrics.confusion_matrix(Y_test, y_pred)

# plot confusion matrix
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, model_name='Random Forest',
                      shots=shot_labels, normalize=True)
plt.show()

# get classification report
print("Classifiction Report for this model")
classification_report = metrics.classification_report(Y_test, y_pred)
print(classification_report)
```

    Accuracy of strokes detection:   0.7816901408450704
    
    



![png](output_56_1.png)


    Classifiction Report for this model
                  precision    recall  f1-score   support
    
              bo       0.86      0.86      0.86        42
              bu       0.72      0.86      0.78        21
              fo       0.62      0.71      0.67        28
              fs       0.77      0.62      0.69        32
              fu       1.00      0.89      0.94        19
    
        accuracy                           0.78       142
       macro avg       0.79      0.79      0.79       142
    weighted avg       0.79      0.78      0.78       142
    


### Gradient Boosting


```python
from sklearn.ensemble import GradientBoostingClassifier
param_grid = {'max_depth': np.arange(1,30,4), \
             'n_estimators':np.arange(1,300,15)}
gbdt_clf = GradientBoostingClassifier()
gbdt_model = GridSearchCV(gbdt_clf, param_grid=param_grid, n_jobs=8)

gbdt_model.fit(X_train,Y_train )
y_pred = gbdt_model.predict(X_test)
accuracy = metrics.accuracy_score(y_true=Y_test,y_pred=y_pred)
# Accuracy of our stroke detectiony
print('Accuracy of strokes detection:   {}\n\n'.format(accuracy))
     
# confusion matrix
cm = metrics.confusion_matrix(Y_test, y_pred)

# plot confusion matrix
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, model_name='GradientBoostingClassifier',
                      shots=shot_labels, normalize=True)
plt.show()
    
# get classification report
print("Classifiction Report for this model")
classification_report = metrics.classification_report(Y_test, y_pred)
print(classification_report)
```

    /home/easy/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py:813: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)


    Accuracy of strokes detection:   0.7183098591549296
    
    



![png](output_58_2.png)


    Classifiction Report for this model
                  precision    recall  f1-score   support
    
              bo       0.88      0.71      0.79        42
              bu       0.62      0.86      0.72        21
              fo       0.56      0.68      0.61        28
              fs       0.68      0.53      0.60        32
              fu       0.90      0.95      0.92        19
    
        accuracy                           0.72       142
       macro avg       0.73      0.75      0.73       142
    weighted avg       0.74      0.72      0.72       142
    


## Deep Learning Models

Applying some deep learning models. The obvious choice will be to apply 1D CNN or RNNs for such a task. We tried only LSTMs.


```python
# Importing tensorflow
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout
```

    Using TensorFlow backend.



```python
# ShotNames are the class labels
# It is a 5 class classification
ShotNames = {
    'bo': [1, 0, 0, 0, 0],
    'bu': [0, 1, 0, 0, 0],
    'fo': [0, 0, 1, 0, 0],
    'fs': [0, 0, 0, 1, 0],
    'fu': [0, 0, 0, 0, 1],
}
```


```python
X = []
y = []
for index, row in X_y.iterrows():
    df = data[row["PersonID"] + "_" + row["ShotName"]][row["StartFrame"]:row["EndFrame"]][cols]
    X.append(df.to_numpy())
    y.append(row["ShotName"])
X = np.array(X)
# One Hot Encoding
y = np.array([ShotNames[i] for i in y])
```


```python
n_classes = len(ShotNames)
timesteps = len(X[0])    # Window size
input_dim = len(X[0][0]) # num of sensors = 6
```


```python
# Initializing parameters
epochs = 100
batch_size = 16
n_hidden = 32
```


```python
# Loading the train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train , Y_test = train_test_split(X, y, test_size=0.1)
```

### LSTM


```python
# Initiliazing the sequential model
model = Sequential()
# Configuring the parameters
model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim)))
# Adding a dropout layer
model.add(Dropout(0.5))
# Adding a dense output layer with sigmoid activation
model.add(Dense(n_classes, activation='sigmoid'))
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_1 (LSTM)                (None, 32)                4992      
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 5)                 165       
    =================================================================
    Total params: 5,157
    Trainable params: 5,157
    Non-trainable params: 0
    _________________________________________________________________



```python
# Compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
```


```python
# Training the model
model.fit(X_train, Y_train,
          batch_size=batch_size,
          validation_data=(X_test, Y_test),
          epochs=epochs)
```

    Train on 637 samples, validate on 71 samples
    Epoch 1/100
    637/637 [==============================] - 3s 5ms/step - loss: 1.6241 - accuracy: 0.2308 - val_loss: 1.5853 - val_accuracy: 0.2676
    Epoch 2/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.6098 - accuracy: 0.2465 - val_loss: 1.5887 - val_accuracy: 0.3099
    Epoch 3/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.5880 - accuracy: 0.2622 - val_loss: 1.5809 - val_accuracy: 0.2958
    Epoch 4/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.5921 - accuracy: 0.2684 - val_loss: 1.5780 - val_accuracy: 0.3099
    Epoch 5/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.5727 - accuracy: 0.2700 - val_loss: 1.5787 - val_accuracy: 0.3099
    Epoch 6/100
    637/637 [==============================] - 1s 1ms/step - loss: 1.5603 - accuracy: 0.2826 - val_loss: 1.5757 - val_accuracy: 0.2817
    Epoch 7/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.5319 - accuracy: 0.3234 - val_loss: 1.5740 - val_accuracy: 0.3099
    Epoch 8/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.5252 - accuracy: 0.3312 - val_loss: 1.5738 - val_accuracy: 0.2958
    Epoch 9/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.5223 - accuracy: 0.3061 - val_loss: 1.5744 - val_accuracy: 0.3380
    Epoch 10/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.5247 - accuracy: 0.3485 - val_loss: 1.5719 - val_accuracy: 0.3380
    Epoch 11/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.5113 - accuracy: 0.3328 - val_loss: 1.5748 - val_accuracy: 0.3239
    Epoch 12/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.5057 - accuracy: 0.3454 - val_loss: 1.5731 - val_accuracy: 0.3521
    Epoch 13/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.4972 - accuracy: 0.3265 - val_loss: 1.5702 - val_accuracy: 0.3239
    Epoch 14/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.4964 - accuracy: 0.3548 - val_loss: 1.5698 - val_accuracy: 0.3239
    Epoch 15/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.4825 - accuracy: 0.3642 - val_loss: 1.5688 - val_accuracy: 0.3521
    Epoch 16/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.4550 - accuracy: 0.3768 - val_loss: 1.5678 - val_accuracy: 0.3662
    Epoch 17/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.4712 - accuracy: 0.3642 - val_loss: 1.5672 - val_accuracy: 0.3803
    Epoch 18/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.4628 - accuracy: 0.3752 - val_loss: 1.5698 - val_accuracy: 0.3803
    Epoch 19/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.4290 - accuracy: 0.3987 - val_loss: 1.5669 - val_accuracy: 0.4085
    Epoch 20/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.4489 - accuracy: 0.3815 - val_loss: 1.5629 - val_accuracy: 0.3803
    Epoch 21/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.4339 - accuracy: 0.4160 - val_loss: 1.5670 - val_accuracy: 0.3662
    Epoch 22/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.4477 - accuracy: 0.3595 - val_loss: 1.5649 - val_accuracy: 0.3662
    Epoch 23/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.4365 - accuracy: 0.3642 - val_loss: 1.5725 - val_accuracy: 0.3521
    Epoch 24/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.4062 - accuracy: 0.4035 - val_loss: 1.5661 - val_accuracy: 0.3521
    Epoch 25/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.3929 - accuracy: 0.4192 - val_loss: 1.5650 - val_accuracy: 0.3662
    Epoch 26/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.3949 - accuracy: 0.4050 - val_loss: 1.5693 - val_accuracy: 0.3239
    Epoch 27/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.3887 - accuracy: 0.4066 - val_loss: 1.5613 - val_accuracy: 0.3099
    Epoch 28/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.4001 - accuracy: 0.4097 - val_loss: 1.5520 - val_accuracy: 0.3380
    Epoch 29/100
    637/637 [==============================] - 2s 2ms/step - loss: 1.3706 - accuracy: 0.4097 - val_loss: 1.5628 - val_accuracy: 0.3662
    Epoch 30/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.3600 - accuracy: 0.4254 - val_loss: 1.5602 - val_accuracy: 0.3521
    Epoch 31/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.3574 - accuracy: 0.4270 - val_loss: 1.5477 - val_accuracy: 0.3380
    Epoch 32/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.3235 - accuracy: 0.4584 - val_loss: 1.5470 - val_accuracy: 0.3239
    Epoch 33/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.3326 - accuracy: 0.4411 - val_loss: 1.5538 - val_accuracy: 0.3239
    Epoch 34/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.3443 - accuracy: 0.4349 - val_loss: 1.5535 - val_accuracy: 0.3099
    Epoch 35/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.3214 - accuracy: 0.4741 - val_loss: 1.5526 - val_accuracy: 0.2958
    Epoch 36/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.3229 - accuracy: 0.4615 - val_loss: 1.5319 - val_accuracy: 0.3099
    Epoch 37/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.3163 - accuracy: 0.4710 - val_loss: 1.5273 - val_accuracy: 0.3239
    Epoch 38/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.2795 - accuracy: 0.4694 - val_loss: 1.5228 - val_accuracy: 0.3099
    Epoch 39/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.2817 - accuracy: 0.4600 - val_loss: 1.5231 - val_accuracy: 0.3521
    Epoch 40/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.2847 - accuracy: 0.4678 - val_loss: 1.5045 - val_accuracy: 0.3380
    Epoch 41/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.2639 - accuracy: 0.4882 - val_loss: 1.5021 - val_accuracy: 0.3380
    Epoch 42/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.2370 - accuracy: 0.4961 - val_loss: 1.4993 - val_accuracy: 0.3944
    Epoch 43/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.2538 - accuracy: 0.4961 - val_loss: 1.4870 - val_accuracy: 0.3944
    Epoch 44/100
    637/637 [==============================] - 2s 2ms/step - loss: 1.2197 - accuracy: 0.5118 - val_loss: 1.4648 - val_accuracy: 0.4225
    Epoch 45/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.1903 - accuracy: 0.5683 - val_loss: 1.4607 - val_accuracy: 0.4085
    Epoch 46/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.1867 - accuracy: 0.5369 - val_loss: 1.4387 - val_accuracy: 0.3803
    Epoch 47/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.1951 - accuracy: 0.5165 - val_loss: 1.4328 - val_accuracy: 0.3944
    Epoch 48/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.1657 - accuracy: 0.5683 - val_loss: 1.4019 - val_accuracy: 0.4507
    Epoch 49/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.1447 - accuracy: 0.5651 - val_loss: 1.3884 - val_accuracy: 0.4507
    Epoch 50/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.1810 - accuracy: 0.5385 - val_loss: 1.3857 - val_accuracy: 0.4225
    Epoch 51/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.1246 - accuracy: 0.5730 - val_loss: 1.3510 - val_accuracy: 0.4366
    Epoch 52/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.1217 - accuracy: 0.5824 - val_loss: 1.3518 - val_accuracy: 0.4789
    Epoch 53/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.1240 - accuracy: 0.5557 - val_loss: 1.3008 - val_accuracy: 0.4930
    Epoch 54/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.0460 - accuracy: 0.5997 - val_loss: 1.3398 - val_accuracy: 0.4789
    Epoch 55/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.0609 - accuracy: 0.5903 - val_loss: 1.3302 - val_accuracy: 0.4507
    Epoch 56/100
    637/637 [==============================] - 2s 3ms/step - loss: 1.0335 - accuracy: 0.6028 - val_loss: 1.2892 - val_accuracy: 0.4648
    Epoch 57/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.0564 - accuracy: 0.5965 - val_loss: 1.2604 - val_accuracy: 0.5070
    Epoch 58/100
    637/637 [==============================] - 1s 2ms/step - loss: 1.0014 - accuracy: 0.6013 - val_loss: 1.2146 - val_accuracy: 0.4648
    Epoch 59/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.9893 - accuracy: 0.6154 - val_loss: 1.1943 - val_accuracy: 0.5070
    Epoch 60/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.9850 - accuracy: 0.6358 - val_loss: 1.1446 - val_accuracy: 0.5070
    Epoch 61/100
    637/637 [==============================] - 2s 3ms/step - loss: 0.9695 - accuracy: 0.6075 - val_loss: 1.1165 - val_accuracy: 0.5493
    Epoch 62/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.9453 - accuracy: 0.6641 - val_loss: 1.1455 - val_accuracy: 0.4789
    Epoch 63/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.9221 - accuracy: 0.6546 - val_loss: 1.1140 - val_accuracy: 0.5493
    Epoch 64/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.8814 - accuracy: 0.6688 - val_loss: 1.0826 - val_accuracy: 0.5211
    Epoch 65/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.8942 - accuracy: 0.6703 - val_loss: 1.0380 - val_accuracy: 0.5915
    Epoch 66/100
    637/637 [==============================] - 2s 3ms/step - loss: 0.8851 - accuracy: 0.6609 - val_loss: 1.0540 - val_accuracy: 0.5634
    Epoch 67/100
    637/637 [==============================] - 2s 3ms/step - loss: 0.8718 - accuracy: 0.6813 - val_loss: 1.0825 - val_accuracy: 0.4930
    Epoch 68/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.8709 - accuracy: 0.6656 - val_loss: 1.0379 - val_accuracy: 0.5915
    Epoch 69/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.8566 - accuracy: 0.6625 - val_loss: 1.0041 - val_accuracy: 0.5634
    Epoch 70/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.8341 - accuracy: 0.6954 - val_loss: 1.0387 - val_accuracy: 0.5634
    Epoch 71/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.8653 - accuracy: 0.6609 - val_loss: 0.9512 - val_accuracy: 0.6197
    Epoch 72/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.8317 - accuracy: 0.6860 - val_loss: 0.9942 - val_accuracy: 0.5915
    Epoch 73/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.8312 - accuracy: 0.6703 - val_loss: 0.9277 - val_accuracy: 0.5915
    Epoch 74/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.7547 - accuracy: 0.7206 - val_loss: 0.9639 - val_accuracy: 0.5493
    Epoch 75/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.7607 - accuracy: 0.7096 - val_loss: 0.9448 - val_accuracy: 0.6197
    Epoch 76/100
    637/637 [==============================] - 2s 2ms/step - loss: 0.7629 - accuracy: 0.7159 - val_loss: 0.9486 - val_accuracy: 0.6620
    Epoch 77/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.7279 - accuracy: 0.7143 - val_loss: 1.0015 - val_accuracy: 0.6338
    Epoch 78/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.7447 - accuracy: 0.7253 - val_loss: 0.9642 - val_accuracy: 0.6056
    Epoch 79/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.7054 - accuracy: 0.7488 - val_loss: 0.9946 - val_accuracy: 0.5634
    Epoch 80/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.7017 - accuracy: 0.7504 - val_loss: 0.9401 - val_accuracy: 0.6056
    Epoch 81/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.7075 - accuracy: 0.7347 - val_loss: 0.9268 - val_accuracy: 0.6197
    Epoch 82/100
    637/637 [==============================] - 2s 2ms/step - loss: 0.6920 - accuracy: 0.7441 - val_loss: 0.9395 - val_accuracy: 0.5915
    Epoch 83/100
    637/637 [==============================] - 2s 3ms/step - loss: 0.7147 - accuracy: 0.7582 - val_loss: 0.8831 - val_accuracy: 0.6620
    Epoch 84/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.6686 - accuracy: 0.7394 - val_loss: 0.9293 - val_accuracy: 0.5634
    Epoch 85/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.6355 - accuracy: 0.7755 - val_loss: 0.9403 - val_accuracy: 0.6056
    Epoch 86/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.6327 - accuracy: 0.7630 - val_loss: 0.8972 - val_accuracy: 0.6197
    Epoch 87/100
    637/637 [==============================] - 2s 2ms/step - loss: 0.6413 - accuracy: 0.7300 - val_loss: 0.9135 - val_accuracy: 0.5915
    Epoch 88/100
    637/637 [==============================] - 1s 2ms/step - loss: 0.6389 - accuracy: 0.7645 - val_loss: 0.8484 - val_accuracy: 0.6338
    Epoch 89/100
    637/637 [==============================] - 2s 2ms/step - loss: 0.6256 - accuracy: 0.7755 - val_loss: 0.8899 - val_accuracy: 0.6338
    Epoch 90/100
    637/637 [==============================] - 2s 3ms/step - loss: 0.6100 - accuracy: 0.7724 - val_loss: 0.8704 - val_accuracy: 0.6338
    Epoch 91/100
    637/637 [==============================] - 2s 2ms/step - loss: 0.5675 - accuracy: 0.8006 - val_loss: 0.9041 - val_accuracy: 0.6056
    Epoch 92/100
    637/637 [==============================] - 2s 3ms/step - loss: 0.6046 - accuracy: 0.7849 - val_loss: 0.8838 - val_accuracy: 0.6338
    Epoch 93/100
    637/637 [==============================] - 2s 3ms/step - loss: 0.6058 - accuracy: 0.7724 - val_loss: 0.8063 - val_accuracy: 0.6479
    Epoch 94/100
    637/637 [==============================] - 2s 3ms/step - loss: 0.5836 - accuracy: 0.7661 - val_loss: 0.8685 - val_accuracy: 0.6338
    Epoch 95/100
    637/637 [==============================] - 2s 3ms/step - loss: 0.6204 - accuracy: 0.7896 - val_loss: 0.8823 - val_accuracy: 0.6056
    Epoch 96/100
    637/637 [==============================] - 2s 3ms/step - loss: 0.5924 - accuracy: 0.7912 - val_loss: 0.8876 - val_accuracy: 0.6197
    Epoch 97/100
    637/637 [==============================] - 2s 2ms/step - loss: 0.5548 - accuracy: 0.7991 - val_loss: 0.9293 - val_accuracy: 0.6338
    Epoch 98/100
    637/637 [==============================] - 2s 3ms/step - loss: 0.5855 - accuracy: 0.7881 - val_loss: 0.9942 - val_accuracy: 0.5915
    Epoch 99/100
    637/637 [==============================] - 2s 3ms/step - loss: 0.5363 - accuracy: 0.8257 - val_loss: 0.8965 - val_accuracy: 0.6338
    Epoch 100/100
    637/637 [==============================] - 2s 3ms/step - loss: 0.6074 - accuracy: 0.7771 - val_loss: 0.9183 - val_accuracy: 0.6338





    <keras.callbacks.callbacks.History at 0x7f9c0086de80>



Data being pretty less for deep learning models, so we didn't try deeper networks.

## Final Notes

This project taught us a lot about doing a machine learning project. Specially the challenges that arise in data collection and it's preprocessing. Overall this is just a small project we did for learning. It can be extended in several ways, first of all by getting more data, so that models like LSTM can be employed. For classical models, we can add more features, do PCA, etc.

Overall a great learning experience. Please feel free to play with it and suggestions are always welcome.
