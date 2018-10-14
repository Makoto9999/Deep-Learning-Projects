# Bacterium Detection

The dataset on the Blackboard contains various measurements (i.e. size, center, etc) from thousands of bacterium under microscope. The last column with non-zero values indicate the bacterium are interesting enough for further study. Otherwise (i.e. last column with zero values), those bacterium are not interesting candidates for further study.

## Data cleaning
#### Read table and sign column names
#### Apply Z-score to numeric predictors for Normalization
#### Convert Target data to binary data
## Build Neural Network
  4 fully connected layers with activation RELU
  1 classifier layer Sigmoid
