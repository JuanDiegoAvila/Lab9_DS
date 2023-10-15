import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa as tsa
import statsmodels as sm
from datetime import datetime
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from statsmodels.tsa.statespace.sarimax import SARIMAX
from shiny import ui, render, App

cuantitativas = ["Gasolina regular", "Gasolina superior", "Diesel alto azufre", "Gas licuado de petróleo", "Total"]


tf.random.set_seed(123)

años_dict = {
    '00': '2000', '01': '2001', '02': '2002', '03': '2003', '04': '2004', '05': '2005',
    '06': '2006', '07': '2007', '08': '2008', '09': '2009', '10': '2010', '11': '2011',
    '12': '2012', '13': '2013', '14': '2014', '15': '2015', '16': '2016', '17': '2017',
    '18': '2018', '19': '2019', '20': '2020', '21': '2021', '22': '2022', '23': '2023',
}

meses_dict = {
    'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04', 'may': '05', 'jun': '06',
    'jul': '07', 'ago': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dic': '12'
}

def dateparse(fecha):
    partes = fecha.split('-')
    partes[0] = meses_dict[partes[0]]
    partes[1] = años_dict[partes[1]]
    return pd.to_datetime('-'.join(partes), format='%m-%Y')

consumo = pd.read_csv('./Data/CONSUMO.csv', parse_dates=['Fecha'], index_col='Fecha',date_parser=dateparse)

# casteamos las variables a float
for col in cuantitativas:
    consumo[col] = consumo[col].str.replace(',', '').astype(float)

train_size = int(len(consumo) * 0.7)
train_consumo, test_consumo = consumo[0:train_size], consumo[train_size:len(consumo)]

tr_consumo = train_consumo.copy()

# precios_promedio

def dateparse(fecha):
    partes = fecha.split('-')
    partes[1] = meses_dict[partes[1]]
    return pd.to_datetime('-'.join(partes), format='%d-%m-%y')

precios_promedio_2021 = pd.read_csv('./Data/Precios-Promedio-Nacionales-Diarios-2021.csv', parse_dates=['FECHA'], index_col='FECHA',date_parser=dateparse)
precios_promedio_2022 = pd.read_csv('./Data/Precios-Promedio-Nacionales-Diarios-2022.csv', parse_dates=['FECHA'], index_col='FECHA',date_parser=dateparse)
precios_promedio_2023 = pd.read_csv('./Data/Precios-Promedio-Nacionales-Diarios-2023.csv', parse_dates=['FECHA'], index_col='FECHA',date_parser=dateparse)

pp = pd.concat([precios_promedio_2021, precios_promedio_2022, precios_promedio_2023])

# Se eliminan las columnas que no se van a utilizar
pp = pp.drop(columns=['Bunker'])

train_size = int(len(pp) * 0.7)
train_precios, test_precios = pp[0:train_size], pp[train_size:(len(pp))]
tr_precios = train_precios.copy()

consumo_diff = tr_consumo.diff()
consumo_diff.dropna(inplace=True)
consumo_log = tr_consumo.copy()

precios_diff = tr_precios.diff().diff()
precios_diff.dropna(inplace=True)
precios_log = tr_precios.copy()


app_ui = ui.page_fixed(
    ui.tags.style(
        """
        body {
            font-family: Times;
            background-color: #020659;
            text-align: center;
        }

        h2 {
            color: #ffffff;
        }
        """
    ),
    ui.h2("Análisis de Consumo y Producto"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_radio_buttons("opcion", "Selecciona una opción:",
                dict(consumo="Consumo", precios="Precios")
            ),
            ui.input_select("producto", "Selecciona un producto:",
                dict(gasolina_regular="Gasolina regular", gasolina_superior="Gasolina superior", diesel="Diesel")
            ),
            ui.input_select("año", "Selecciona un año:",
                dict(_2021="2021", _2022="2022", _2023="2023")
            ),
            ui.input_slider("mes", "Selecciona un mes:", min=1, max=12, step=1, value=1),
        ),
        ui.panel_main(
            ui.output_plot("plot")
        )
    )
)

def generate_date_from_input(año, mes):
    return f"{año.replace('_', '')}-{str(mes).zfill(2)}"

def server(input, output, session):
    @output
    @render.plot
    def plot():
        fig, ax = plt.subplots(figsize=(14, 4))

        año = input.año()
        mes = input.mes()
        producto = input.producto()

        selected_date = generate_date_from_input(año, mes)


        titulo = ""

        if input.opcion() == 'consumo':
            if input.producto() == "gasolina_regular":
                producto = "Gasolina regular"
            elif input.producto() == "gasolina_superior":
                producto = "Gasolina superior"
            elif input.producto() == "diesel":
                producto = "Diesel alto azufre"
            
            # Entrenamiento del modelo
            modelo = SARIMAX(consumo_log[producto], order=(4, 1, 1), seasonal_order=(2, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
            resultado = modelo.fit(disp=False)

            start_date = test_consumo[producto][selected_date:].index[0]
            end_date = test_consumo[producto][selected_date:].index[-1]

            # Obtener las predicciones
            predicciones_diff = resultado.get_prediction(start=start_date, end=end_date, dynamic=False)
            
            # Plot the data
            ax = test_consumo[producto][selected_date:].plot(label='Observado', figsize=(14, 4))
            predicciones_diff.predicted_mean.plot(ax=ax, label='Predicciones', alpha=.7, color='red')

            ax.set_xlabel('Fecha')
            ax.set_ylabel('Valor')
            ax.set_title('Consumo de ' + producto + ' en ' + selected_date)
            ax.legend()

        elif input.opcion() == 'precios':
            if input.producto() == "gasolina_regular":
                producto = "Regular"
            elif input.producto() == "gasolina_superior":
                producto = "Superior"
            elif input.producto() == "diesel":
                producto = "Diesel"

            modelo121 = SARIMAX(precios_log[producto], order=(1, 2, 1), seasonal_order=(1, 2, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
            resultado_m121 = modelo121.fit(disp=False)

            start_date = test_precios[producto][selected_date:].index[0]
            end_date = test_precios[producto][selected_date:].index[-1]

            predicciones_diff = resultado_m121.get_prediction(start=start_date, end = end_date, dynamic=False)
            
            # Plot the data
            ax = test_precios[producto][selected_date:].plot(label='Observado', figsize=(14, 4))
            predicciones_diff.predicted_mean.plot(ax=ax, label='Predicciones', alpha=.7, color='red')

            ax.set_xlabel('Fecha')
            ax.set_ylabel('Precios')
            ax.set_title('Precios de ' + producto + ' en ' + selected_date)
            ax.legend()

        ax.legend()

        # Devolver el gráfico
        return fig



app = App(app_ui, server)
app.run()