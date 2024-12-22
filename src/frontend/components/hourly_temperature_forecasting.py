import streamlit as st

import os
import torch
import torch.nn as nn


device = torch.device('cpu')


class WeatherForecastor(nn.Module):
    def __init__(self, embedding_dim,
                 hidden_size, n_layers,
                 dropout_prob):
        super(WeatherForecastor, self).__init__()
        self.rnn = nn.RNN(embedding_dim, hidden_size,
                          n_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, hn = self.rnn(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


@st.cache_resource
def load_model(model_path):
    model = WeatherForecastor(
        embedding_dim=1,
        hidden_size=8,
        n_layers=3,
        dropout_prob=0.1
    )
    model.load_state_dict(torch.load(
        model_path, weights_only=True, map_location=device))
    model.eval()
    model.dropout.train(False)
    return model


# load model
model_path = absolute_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../weights/hourly_temperature_forecasting.pt'))
model = load_model(model_path)


def run():
    st.title('Hourly temperature Forecasting')
    example = [24.86, 22.75, 20.07, 17.81, 17.16, 15.01]
    target = 14.47
    st.write("Example:", ", ".join(map(str, example)), "Target:", target)

    input_temperature = st.text_input(
        'Enter 6 temperatures separated by commas:',
        value='24.86, 22.75, 20.07, 17.81, 17.16, 15.01'
    )
    temp_input = [float(temp) for temp in input_temperature.split(",")]

    # Convert user input to tensor with the required shape
    temp_tensor = torch.FloatTensor(temp_input).unsqueeze(0).unsqueeze(-1)
    print(temp_tensor)

    if st.button('Predict'):
        with torch.no_grad():
            result = model(temp_tensor)
            print(result)

        st.markdown('### Predicted Temperature for the Next Hour: ')
        st.success(f'Predicted Temperature: {result.item():.2f}\u00B0C')
