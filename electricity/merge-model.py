{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "adec5077",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "def average_predictions_regression(X, lstm_model, rf_model):\n",
    "    \"\"\"\n",
    "    Get averaged predictions from an LSTM model and Random Forest model (regression).\n",
    "    \n",
    "    Parameters:\n",
    "        X : np.array or pd.DataFrame\n",
    "            Input data.\n",
    "        lstm_model : trained deep learning model (e.g., Keras LSTM)\n",
    "        rf_model   : trained ML model (e.g., RandomForestRegressor)\n",
    "\n",
    "    Returns:\n",
    "        np.array : Averaged predictions.\n",
    "    \"\"\"\n",
    "    # Deep learning model usually expects numpy arrays\n",
    "    y_lstm = lstm_model.predict(X, verbose=0)  \n",
    "    # RF model expects np.array or DataFrame\n",
    "    y_rf = rf_model.predict(X)  \n",
    "    \n",
    "    # Ensure shapes match (flatten if necessary)\n",
    "    y_lstm = np.squeeze(y_lstm)\n",
    "    y_rf   = np.squeeze(y_rf)\n",
    "\n",
    "    # Simple average\n",
    "    y_avg = (y_lstm + y_rf) / 2.0\n",
    "    return y_avg\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ab17940b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "electricity",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
