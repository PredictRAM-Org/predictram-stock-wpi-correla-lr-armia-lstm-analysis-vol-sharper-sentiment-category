import os
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from pmdarima import auto_arima
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

# Function to prepare data for LSTM
def prepare_data_for_lstm(data, look_back=1):
    x, y = [], []
    for i in range(len(data) - look_back):
        x.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(x), np.array(y)

# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to predict future prices using LSTM
def predict_future_lstm(last_observed_price, model, min_max_scaler, num_steps=1):
    predicted_prices = []
    input_data = last_observed_price.reshape(1, -1, 1)

    for _ in range(num_steps):
        predicted_price = model.predict(input_data)
        predicted_prices.append(predicted_price[0, 0])
        input_data = np.append(input_data[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

    return min_max_scaler.inverse_transform(np.array(predicted_prices).reshape(1, -1))[0]

# Function to get sentiment score for a given text using VADER
def get_sentiment_score(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']
    return sentiment_score

# Function to get news articles and sentiment scores for a given stock
def get_news_sentiment_scores(api_key, stock_name, num_articles=5):
    url = "https://newsapi.org/v2/everything"
    query_params = {
        "apiKey": api_key,
        "q": f"{stock_name} AND (business OR finance) AND India",
        "pageSize": num_articles
    }

    response = requests.get(url, params=query_params)
    news_data = response.json()

    sentiment_scores = []
    articles_list = []

    if 'articles' in news_data:
        articles = news_data['articles']
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            full_text = f"{title}. {description}"
            sentiment_score = get_sentiment_score(full_text)
            sentiment_scores.append(sentiment_score)
            articles_list.append({'Title': title, 'Description': description, 'Sentiment Score': sentiment_score, 'Link': article.get('url', '')})

    return articles_list

# Load WPI data
WPI_data = pd.read_excel("WPI.xlsx")
WPI_data['Date'] = pd.to_datetime(WPI_data['Date'])
WPI_data.set_index('Date', inplace=True)

# Load categorized stocks data
categorized_stocks_data = pd.read_excel("categorized_stocks.xlsx")

# Streamlit UI
st.image("png_2.3-removebg-preview.png", width=400)
st.title("Stock Price-WPI Correlation Analysis with Expected Inflation, Price Prediction, and News Sentiment Analysis")

# User input for uploading Excel file with stocks name column
uploaded_file = st.file_uploader("Upload Excel file with stocks name column", type=["xlsx", "xls"])
if uploaded_file is not None:
    stocks_data = pd.read_excel(uploaded_file)
else:
    st.warning("Please upload an Excel file.")
    st.stop()

# Select data range for training models
data_range = st.selectbox("Select Data Range for Model Training:", ["6 months", "1 year", "3 years", "5 years"])

# Filter data based on the selected range
end_date = pd.to_datetime('today')
if data_range == "6 months":
    start_date = end_date - pd.DateOffset(months=6)
elif data_range == "1 year":
    start_date = end_date - pd.DateOffset(years=1)
elif data_range == "3 years":
    start_date = end_date - pd.DateOffset(years=3)
else:
    start_date = end_date - pd.DateOffset(years=5)

# Filter WPI data
filtered_WPI_data = WPI_data.loc[start_date:end_date]

# User input for expected WPI inflation
expected_inflation = st.number_input("Enter Expected Upcoming WPI Inflation:", min_value=0.0, step=0.01)

# News API key from newsapi.org
news_api_key = "5843e8b1715a4c1fb6628befb47ca1e8"

# Train models
if st.button("Train Models"):
    st.write(f"Training models with data range: {data_range}, expected WPI inflation: {expected_inflation}...")

    correlations = []
    actual_correlations = []
    future_prices_lr_list = []
    future_prices_arima_list = []
    latest_actual_prices = []
    future_price_lstm_list = []
    stock_names = []
    volatilities = []
    sharpe_ratios = []
    news_sentiment_scores = []

    categorized_stocks_list = []  # Initialize the list here

    for index, row in stocks_data.iterrows():
        stock_name = row['Stock']

        additional_info = categorized_stocks_data[categorized_stocks_data['Symbol'] == stock_name]

        if not additional_info.empty:
            st.write(f"\nAdditional Information for {stock_name}:")
            st.table(additional_info)

            categorized_stocks_list.append({'Stock': stock_name, 'Categorized Stocks Data': additional_info})

        stock_file_path = os.path.join("stock_folder", f"{stock_name}.xlsx")
        if os.path.exists(stock_file_path):
            selected_stock_data = pd.read_excel(stock_file_path)
            selected_stock_data['Date'] = pd.to_datetime(selected_stock_data['Date'])
            selected_stock_data.set_index('Date', inplace=True)
            filtered_stock_data = selected_stock_data.loc[start_date:end_date]

            merged_data = pd.merge(filtered_stock_data, filtered_WPI_data, left_index=True, right_index=True, how='inner')

            if merged_data['WPI'].isnull().any():
                st.write(f"Warning: NaN values found in 'WPI' column for {stock_name}. Dropping NaN values.")
                merged_data = merged_data.dropna(subset=['WPI'])

            merged_data['WPI Change'] = merged_data['WPI'].pct_change()
            merged_data = merged_data.dropna()

            correlation_close_WPI = merged_data['Close'].corr(merged_data['WPI Change'])
            correlation_actual = merged_data['Close'].corr(merged_data['WPI'])
            actual_correlations.append(correlation_actual)

            st.write(f"Correlation between 'Close' and 'WPI Change' for {stock_name}: {correlation_close_WPI}")
            st.write(f"Actual Correlation between 'Close' and 'WPI' for {stock_name}: {correlation_actual}")

            model_lr = LinearRegression()
            X_lr = merged_data[['WPI']]
            y_lr = merged_data['Close']
            model_lr.fit(X_lr, y_lr)

            model_arima = auto_arima(y_lr, seasonal=False, suppress_warnings=True)

            min_max_scaler = MinMaxScaler()
            scaled_data = min_max_scaler.fit_transform(y_lr.values.reshape(-1, 1))
            x_train, y_train = prepare_data_for_lstm(scaled_data, look_back=3)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            model_lstm = build_lstm_model(x_train.shape[1])
            model_lstm.fit(x_train, y_train, epochs=50, batch_size=32)

            future_prices_lr = model_lr.predict([[expected_inflation]])
            st.write(f"Predicted Price Change for Future Inflation (Linear Regression): {future_prices_lr[0]}")

            arima_predictions = model_arima.predict(1)
            if isinstance(arima_predictions, pd.Series):
                future_prices_arima = arima_predictions.iloc[0]
            else:
                future_prices_arima = arima_predictions[0]
            st.write(f"Predicted Price Change for Future Inflation (ARIMA): {future_prices_arima}")

            last_observed_price = scaled_data[-3:]
            future_price_lstm = predict_future_lstm(last_observed_price, model_lstm, min_max_scaler)
            st.write(f"Predicted Stock Price for Future Inflation (LSTM): {future_price_lstm}")

            news_articles = get_news_sentiment_scores(news_api_key, stock_name, num_articles=5)

            st.write(f"News Articles and Sentiment Scores for {stock_name}:")

            if news_articles:
                avg_sentiment_score = np.mean([article['Sentiment Score'] for article in news_articles])
                st.write(f"Avg. Sentiment Score: {avg_sentiment_score}")

                for article in news_articles:
                    st.write(f"Title: {article['Title']}")
                    st.write(f"Description: {article['Description']}")
                    st.write(f"Sentiment Score: {article['Sentiment Score']}")
                    st.write(f"Link: {article['Link']}")
                    st.write("-----")

                news_sentiment_scores.append({'Stock': stock_name, 'Avg. Sentiment Score': avg_sentiment_score})
            else:
                st.warning(f"No news found for {stock_name}.")

            latest_actual_price = merged_data['Close'].iloc[-1]
            st.write(f"Latest Actual Price for {stock_name}: {latest_actual_price}")

            daily_returns = merged_data['Close'].pct_change().dropna()
            volatility = daily_returns.std()
            annualized_volatility = volatility * np.sqrt(252)
            average_daily_return = daily_returns.mean()
            annualized_return = average_daily_return * 252
            risk_free_rate = 0.02
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

            st.write(f"Volatility for {stock_name}: {annualized_volatility}")
            st.write(f"Sharpe Ratio for {stock_name}: {sharpe_ratio}")

            correlations.append(correlation_close_WPI)
            future_prices_lr_list.append(future_prices_lr[0])
            future_prices_arima_list.append(future_prices_arima)
            latest_actual_prices.append(latest_actual_price)
            future_price_lstm_list.append(future_price_lstm)
            stock_names.append(stock_name)
            volatilities.append(annualized_volatility)
            sharpe_ratios.append(sharpe_ratio)

    # Create a DataFrame for categorized stocks data
    categorized_stocks_df = pd.DataFrame(categorized_stocks_list)

    # Display results in descending order of correlation
    st.write("\nResults Sorted by Correlation:")
    sorted_results_df = pd.DataFrame({
        'Stock': stock_names,
        'Correlation with WPI Change': correlations,
        'Actual Correlation with WPI': actual_correlations,
        'Predicted Price Change (Linear Regression)': future_prices_lr_list,
        'Predicted Price Change (ARIMA)': future_prices_arima_list,
        'Latest Actual Price': latest_actual_prices,
        'Predicted Stock Price (LSTM)': future_price_lstm_list,
        'Volatility': volatilities,
        'Sharpe Ratio': sharpe_ratios,
        'News Sentiment Scores': news_sentiment_scores
    }).sort_values(by='Correlation with WPI Change', ascending=False)

    st.table(sorted_results_df)

# Display categorized stocks data at the end
st.write("\nCategorized Stocks Data:")
categorized_stocks_df_selected = categorized_stocks_df[selected_columns]
categorized_stocks_df_str = categorized_stocks_df_selected.astype(str)
st.table(categorized_stocks_df_str)
