# Flower Detection
150 records of 4-dimension dataset, belonging to 3 types of flowers.
## Build CNN network with 3 convolutional layers and 2 fully connected layers.
### Convolutional layers:
     input size: (200*200*3)
     output size: (25*25*128)
### Fully connected layers:
     input size: flatten(25*25*128)
     output size: 3
### Softmax:
     output size:3
     
## Save the training results as a csv file
      1, val_acc, 0.6666666865348816
      2, val_acc, 0.6666666865348816
      3, val_acc, 0.8666666746139526
      4, val_acc, 0.9000000059604645
      5, val_acc, 0.9000000059604645
      6, val_acc, 0.9333333373069763
      7, val_acc, 0.9333333373069763
      8, val_acc, 0.9666666686534882
      9, val_acc, 0.9333333373069763
      10, val_acc, 0.9666666686534882
      11, val_acc, 0.9000000059604645
      12, val_acc, 0.9666666686534882
      13, val_acc, 0.9333333373069763
      14, val_acc, 0.9333333373069763
      15, val_acc, 0.9666666686534882
      16, val_acc, 0.9333333373069763
      17, val_acc, 0.9666666686534882
      18, val_acc, 0.9333333373069763
      19, val_acc, 0.9666666686534882
      20, val_acc, 0.9333333373069763
      21, val_acc, 0.9333333373069763
      22, val_acc, 0.9666666686534882
      23, val_acc, 0.9666666686534882
      24, val_acc, 0.9333333373069763
      25, val_acc, 0.9333333373069763
      26, val_acc, 0.9666666686534882
      27, val_acc, 0.9666666686534882
      28, val_acc, 0.9666666686534882
      29, val_acc, 0.9666666686534882
      30, val_acc, 0.9666666686534882
      31, val_acc, 0.9333333373069763
      32, val_acc, 0.9000000059604645
      33, val_acc, 0.9000000059604645
      34, val_acc, 0.9333333373069763
      35, val_acc, 0.9333333373069763
      36, val_acc, 0.6666666716337204
      37, val_acc, 0.9333333373069763
      38, val_acc, 0.9333333373069763
      39, val_acc, 0.9666666686534882
      40, val_acc, 0.9333333373069763

## Save the top model with the highest validation accuracy
  saver = tf.train.Saver(max_to_keep=1)
  saver.save(sess, './path/ckpt/flower.ckpt')

## Restore saved model
    # loade the graph of the model
    saver = tf.train.import_meta_graph('./path/ckpt/flower.ckpt.meta')
    # restore the parameters of the model
    saver.restore(sess, './path/ckpt/flower.ckpt')
    print("Model restored.\n")
    
## Predict the test data
    Model restored.

    Predicted label: [0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2]
      True label  : [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2]
       Accuracy  : 0.9666666666666667

     confusion matrix:
     [[ 9  1  0]
     [ 0 10  0]
     [ 0  0 10]]

  
  
  
  
