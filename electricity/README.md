
**Energy Price Prediction based on Weather Data:**


Project Overview :
This project predicts future energy prices using historical energy price data, weather variables, and timestamps.
We apply Machine Learning (ML) and Deep Learning (LSTM) models to capture temporal dependencies and understand how external factors (like temperature, wind speed, and humidity) affect energy prices.



##
Tech Stack
Language : Python 3.12

Libaries and Framworks :

TensorFlow / Keras (LSTM implementation)

Scikit-learn (metrics, preprocessing)

Pandas, NumPy (data manipulation)

Matplotlib, Seaborn (visualization)

FastAPI (API deployment)

Streamlit (interactive dashboard)



**Dataset:**


Input:
Historical energy prices

`date`: timestamp of observation

`RRP`: energy price

`min_temperature`, `max_temperature`, `solar_exposure`, `rainfall`
   `school_day`, `holiday`

   **Preprocessing steps:**

   Encode categorical variables

   Create lag features (1, 7, 14 days)

   Create rolling averages (7, 30 days)

   Add interaction features (temperature range, rainfall × max temperature)




Output:

Predicted energy price for the next time step





**Model Architecture  LSTM :**

Input Layer: Sequences of past energy prices + weather data

LSTM Layer(s): Capture temporal dependencies in the data

Dropout Layer: Prevent overfitting

Dense Layer: Output the predicted price

Activation: Linear (for regression tasks)





**Evaluation Metrics:**

We evaluate performance using:
MAE (Mean Absolute Error):

Average prediction error in price units

RMSE (Root Mean Squared Error):

Highlights large mistakes

R² Score:

Explains how much variance in prices the model captures.



**Results:**




How to run

1. Clone the repository

git clone https://github.com/yourusername/energy-price-prediction.git
cd energy-price-prediction

2. Install dependencies

pip install -r requirements.txt

3. Run preprocessing

python preprocessing.py


4. Train the model


python model.py

5. Start the API

python api.py


6. Launch the dashboard

streamlit run app.py





**Maybe future work:**

Experiment with Transformers.



**Contributors:**


Niko Taheri

Alessio Beverino

Kieren Schmidt

Filippa Gagern
