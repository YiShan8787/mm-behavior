# RadHARex



## Dataset
Point cloud dataset collected using mmWave Radar. The data is collected using the ROS package for TI mmWave radar developed by [radar-lab.](https://github.com/radar-lab/mmfall) Raw dataset is available soon.

### Format
```
header: 
  seq: 6264
  stamp: 
    secs: 1538888235
    nsecs: 712113897
  frame_id: "ti_mmwave"   # Frame ID used for multi-sensor scenarios
point_id: 17              # Point ID of the detecting frame (Every frame starts with 0)
x: 8.650390625            # Point x coordinates in m (front from antenna)
y: 6.92578125             # Point y coordinates in m (left/right from antenna, right positive)
z: 0.0                    # Point z coordinates in m (up/down from antenna, up positive)
range: 11.067276001       # Radar measured range in m
velocity: 0.0             # Radar measured range rate in m/s
doppler_bin: 8            # Doppler bin location of the point (total bins = num of chirps)
bearing: 38.6818885803    # Radar measured angle in degrees (right positive)
intensity: 13.6172780991  # Radar measured intensity in dB
```
[For more information about the ROS package used to collect data and its description, please click here!](https://github.com/radar-lab/mm-fall)

## Data Preprocessing
Data preprocessing is done by extracting by the voxels. Voxel extraction code is available [here.](https://github.com/nesl/RadHAR/tree/master/DataPreprocessing). The file have variables which need to be set to the path of the raw data folders. The path is controlled using the below variables.

```
parent_dir = 'Path_to_training_or_test_data'
sub_dirs=['boxing','jack','jump','squats','walk']
extract_path = 'Path_to_put_extracted_data'
```

- Separate train and test files with 71.6 minutes data in train and 21.4 minutes data in test.
- Voxels of dimensions 10x32x32 (where 10 is the depth).
- Windows of 2 seconds (60 frames) having a sliding factor of 0.33 seconds (10 frames). 
- We finally get 12097 samples in training and 3538 samples in testing.
- For deep learning classifiers, we use 20% of the training samples for validation.

Finally the voxels have the format: 60\*10\*32\*32, where 60 represents the time, 10 represent the depth, and 32\*32 represents the x and y dimension. 

## Classifiers:
- Bi-directional LSTM Classifier: [Code]()

## Pretrained Classifiers and Preprocessed Dataset:

## Reference:
- https://github.com/nesl/RadHAR


