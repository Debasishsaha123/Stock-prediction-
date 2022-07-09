# Google Stock-prediction

![BogusHandsomeGermanpinscher-size_restricted](https://user-images.githubusercontent.com/100334542/178103399-e7ed3f8d-c9bc-4d18-9ca1-05ad61afb144.gif)




In this project we will be looking at data from the stock market, particularly some technology stocks. We will learn how to use pandas to get stock information, visualize different aspects of it, and finally we will look at a few ways of analyzing the risk of a stock, based on its previous performance history. We will also be predicting future stock prices through a Long Short Term Memory (LSTM) method!


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


