# Google Stock-prediction

![BogusHandsomeGermanpinscher-size_restricted](https://user-images.githubusercontent.com/100334542/178103399-e7ed3f8d-c9bc-4d18-9ca1-05ad61afb144.gif)


Stock Price Prediction using machine learning helps you discover the future value of company stock and other financial assets traded on an exchange. The entire idea of predicting stock prices is to gain significant profits. Predicting how the stock market will perform is a hard task to do. There are other factors involved in the prediction, such as physical and psychological factors, rational and irrational behavior, and so on. All these factors combine to make share prices dynamic and volatile. This makes it very difficult to predict stock prices with high accuracy.  We will  predicting future stock prices through a Long Short Term Memory (LSTM) method!


## Importance of Stock Market

1.Stock markets help companies to raise capital.

2.It helps generate personal wealth.

3.Stock markets serve as an indicator of the state of the economy.

4.It is a widely used source for people to invest money in companies with high growth potential.

## LSTM METHOD

Long Short Term Memory Network is an advanced RNN, a sequential network, that allows information to persist. It is capable of handling the vanishing gradient problem faced by RNN. A recurrent neural network is also known as RNN is used for persistent memory.

Let’s say while watching a video you remember the previous scene or while reading a book you know what happened in the earlier chapter. Similarly RNNs work, they remember the previous information and use it for processing the current input. The shortcoming of RNN is, they can not remember Long term dependencies due to vanishing gradient. LSTMs are explicitly designed to avoid long-term dependency problems.


![image](https://user-images.githubusercontent.com/100334542/177029908-544cbbcb-fc34-49e3-8746-327dc6385e89.png)

The first part chooses whether the information coming from the previous timestamp is to be remembered or is irrelevant and can be forgotten. In the second part, the cell tries to learn new information from the input to this cell. At last, in the third part, the cell passes the updated information from the current timestamp to the next timestamp.

These three parts of an LSTM cell are known as gates. The first part is called Forget gate, the second part is known as the Input gate and the last one is the Output gate.

![image](https://user-images.githubusercontent.com/100334542/177029952-15204877-3167-43e3-8b7b-7862c1156d12.png)


Just like a simple RNN, an LSTM also has a hidden state where H(t-1) represents the hidden state of the previous timestamp and Ht is the hidden state of the current timestamp. In addition to that LSTM also have a cell state represented by C(t-1) and C(t) for previous and current timestamp respectively.

Here the hidden state is known as Short term memory and the cell state is known as Long term memory. Refer to the following image.

![image](https://user-images.githubusercontent.com/100334542/177030001-1fd48989-e390-4205-8843-3e0d6cd745e8.png)

## Dataset

https://github.com/Debasishsaha123/Stock-prediction-/blob/main/Google_Stock_Price_Train.csv

The dataset contains 6 columns associated with time series like the date and the different variables like close, high, low and volume. We will use opening and closing values for our experimentation of time series with LSTM.

### Opening price

![image](https://user-images.githubusercontent.com/100334542/178118353-5d975d4b-7fd6-45c6-8afe-bbbe8125e81b.png)

### closing price and 30 day mean closing price

![image](https://user-images.githubusercontent.com/100334542/178118428-20faa512-2511-4263-a3d7-07b419c69652.png)

## High price

![image](https://user-images.githubusercontent.com/100334542/178118455-c4a59f9a-0fef-4647-a770-76759b7b4ebf.png)

## Importing the Libaries

As we all know, the first step is to import the libraries required to preprocess GOOGLE stock data and the other libraries required for constructing and visualizing the LSTM model outputs. We’ll be using the Keras library from the TensorFlow framework for this. All modules are imported from the Keras library.

1.NUMPY

2.PANDAS

3.MATPLOTLIB

4.SkLearn

5.KERAS--->(I)SEQUENTIAL,(II) DENSE,LSTM,DROPOUT

### Setting the Target Variable and Selecting the Features

We pick the features that serve as the independent variable to the target variable (dependent variable). We choose four characteristics to account for training purposes:

Open

High

Low

Volume

## Building the LSTM Model for Stock Market Prediction

In this step, we’ll build a Sequential Keras model with one LSTM layer. The LSTM layer has 50 units and is followed by one Dense Layer of one neuron.

We compile the model using Adam Optimizer and the Mean Squared Error as the loss function. For an LSTM model, this is the most preferred combination.

Model: "sequential"

_________________________________________________________________
 
 Layer (type)                Output Shape              Param #   
 
=================================================================

 lstm (LSTM)                 (None, 100, 50)           10400     
 
                                                                 
 dropout (Dropout)           (None, 100, 50)           0         
 
                                                                 
 lstm_1 (LSTM)               (None, 100, 50)           20200     
                                                                 
                                                                 
 dropout_1 (Dropout)         (None, 100, 50)           0         
 
                                                                 
 lstm_2 (LSTM)               (None, 100, 50)           20200     
 
                                                                 
 dropout_2 (Dropout)         (None, 100, 50)           0         
                                                                 
                                                                 
 lstm_3 (LSTM)               (None, 50)                20200     
 
                                                                 
 dropout_3 (Dropout)         (None, 50)                0         
                                                                 
                                                                 
 dense (Dense)               (None, 1)                 51        
                                                                 
=================================================================

Total params: 71,051

Trainable params: 71,051

Non-trainable params: 0

_________________________________________________________________

Finally, now that we’ve projected the values for the test set, we can display the graph to compare both Adj Close’s true values and Adj Close’s predicted value using the LSTM Machine Learning model.

![image](https://user-images.githubusercontent.com/100334542/178459929-de4fee14-8fcd-479a-b08d-8f5f7de840c1.png)

The graph above demonstrates that the extremely basic single LSTM network model created above detects some patterns. We may get a more accurate depiction of every specific company’s stock value by fine-tuning many parameters and adding more LSTM layers to the model.

## Conclusion

However, with the introduction of Machine Learning and its strong algorithms, the most recent market research and Stock Market Prediction advancements have begun to include such approaches in analyzing stock market data. The Opening Value of the stock, the Highest and Lowest values of that stock on the same days, as well as the Closing Value at the end of the day, are all indicated for each date. Furthermore, the total volume of the stocks in the market is provided, With this information, it is up to the job of a Machine LearningData Scientist to look at the data and develop different algorithms that may help in finding appropriate stocks values. 

Predicting the stock market was a time-consuming and laborious procedure a few years or even a decade ago. However, with the application of machine learning for stock market forecasts, the procedure has become much simpler. Machine learning not only saves time and resources but also outperforms people in terms of performance. it will always prefer to use a trained computer algorithm since it will advise you based only on facts, numbers, and data and will not factor in emotions or prejudice.
