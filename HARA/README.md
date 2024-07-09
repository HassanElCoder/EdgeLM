# Human Activity Recognition Using Smartphones
This project uses SVM to predicts the type of  human acitivity based on 
accelerometer data. The sensory data are produced by 3-axis accelerometer and gyroscopes embedded in Samsung Galaxy S II smart phone. The the data uses in this experiment was collected from 30 participant volunteers within an age bracket of 19-48 years wearing the smartphone on their waist. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) and the sensory data was recorded during these activities. The data is available on <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>  

## Running the Linear SVM training 
1. Download the Data from <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>
2. Unzip the data in the local ./data folder
3. Change the name of the data folder: mv smartphone+based+recognition+of+human+activities+and+postural+transitions  HAPTDataSet
4. Train and Plot the validation and training scores: python hara.py
