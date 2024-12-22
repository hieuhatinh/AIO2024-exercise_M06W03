import streamlit as st

import components.financial_news_sentiment_analysis as finalcial_news
import components.hourly_temperature_forecasting as forecasting_temperature


def app():
    model = st.sidebar.selectbox(
        'Select a model',
        ('Financial News Sentiment Analysis', 'Hourly Temperature Forecasting')
    )

    if model == 'Financial News Sentiment Analysis':
        finalcial_news.run()
    elif model == 'Hourly Temperature Forecasting':
        forecasting_temperature.run()


if __name__ == '__main__':
    app()
