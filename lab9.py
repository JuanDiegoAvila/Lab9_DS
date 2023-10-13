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

# Cargamos los datos
precios_promedio_2021 = pd.read_csv('./Data/Precios-Promedio-Nacionales-Diarios-2021.csv')
precios_promedio_2022 = pd.read_csv('./Data/Precios-Promedio-Nacionales-Diarios-2022.csv')
precios_promedio_2023 = pd.read_csv('./Data/Precios-Promedio-Nacionales-Diarios-2023.csv')

# Se renombra la ultima columna a Glp Cilindro Lbs.
precios_promedio_2021.rename(columns={'Unnamed: 7': 'Glp Cilindro Lbs.'}, inplace=True)
precios_promedio_2022.rename(columns={'Unnamed: 7': 'Glp Cilindro Lbs.'}, inplace=True)
precios_promedio_2023.rename(columns={'Unnamed: 7': 'Glp Cilindro Lbs.'}, inplace=True)

# Se unen los tres dataframes en uno solo
precios_promedio = pd.concat([precios_promedio_2021, precios_promedio_2022, precios_promedio_2023])

# Se eliminan las columnas que no se van a utilizar
precios_promedio = precios_promedio.drop(columns=['Bunker'])

cuantitativas = [
    "Tipo de Cambio",
    "Superior",
    "Regular",
    "Diesel",
    "Glp Cilindro 25Lbs.",
    "Glp Cilindro Lbs."
]

meses_dict = {
    'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04', 'may': '05', 'jun': '06',
    'jul': '07', 'ago': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dic': '12'
}

def reemplazar_mes(fecha):
    partes = fecha.split('-')
    partes[1] = meses_dict[partes[1]]
    return '-'.join(partes)

precios_promedio['FECHA'] = precios_promedio['FECHA'].apply(reemplazar_mes)
precios_promedio['FECHA'] = pd.to_datetime(precios_promedio['FECHA'], format='%d-%m-%y')
precios_promedio['AÑO'] = precios_promedio['FECHA'].dt.year
precios_promedio['MES'] = precios_promedio['FECHA'].dt.month

# Cargamos los datos
consumo = pd.read_csv('./Data/CONSUMO.csv')

cuantitativas = ["Gasolina regular", "Gasolina superior", "Diesel alto azufre", "Gas licuado de petróleo", "Total"]

# casteamos las variables a float
for col in cuantitativas:
    consumo[col] = consumo[col].str.replace(',', '').astype(float)

consumo_numericas = consumo[cuantitativas]

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

def reemplazar_mes(fecha):
    partes = fecha.split('-')
    partes[0] = meses_dict[partes[0]]
    partes[1] = años_dict[partes[1]]
    return '-'.join(partes)

consumo['Fecha'] = consumo['Fecha'].apply(reemplazar_mes)

consumo['Fecha'] = pd.to_datetime(consumo['Fecha'], format='%m-%Y')

consumo['AÑO'] = consumo['Fecha'].dt.year
consumo['MES'] = consumo['Fecha'].dt.month

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
print ('\n Parsed Data:')
consumo.head()

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

print ('\n Parsed Data:')
train_size = int(len(pp) * 0.7)
train_precios, test_precios = pp[0:train_size], pp[train_size:(len(pp))]
tr_precios = train_precios.copy()

consumo_diff = tr_consumo.diff()
consumo_diff.dropna(inplace=True)
consumo_log = tr_consumo.copy()

consumo_log_diff = pd.DataFrame(columns = cuantitativas)

precios_diff = tr_precios.diff()
precios_diff.dropna(inplace=True)
precios_log = tr_precios.copy()

precios_log_diff = pd.DataFrame(columns = cuantitativas)

# precios_log_diff_act = precios_log['Superior'].diff().diff()
# precios_log_diff_act.dropna(inplace = True)
# precios_log_diff['Superior'] = precios_log_diff_act
# tsa_acf = acf(precios_log_diff_act, nlags=5,fft=False)
# tsa_pacf = pacf(precios_log_diff_act, nlags=36)


# consumo_log_diff_act = consumo_log['Gasolina regular'].diff().diff()
# consumo_log_diff_act.dropna(inplace = True)
# consumo_log_diff['Gasolina regular'] = consumo_log_diff_act
# tsa_acf = acf(consumo_log_diff_act, nlags=5,fft=False)
# tsa_pacf = pacf(consumo_log_diff_act, nlags=36)

# precios_log_D = precios_log.diff(12)
# precios_log_D.dropna(inplace=True)

# consumo_log_D = consumo_log.diff(12)
# consumo_log_D.dropna(inplace=True)

# modelo441 = SARIMAX(consumo_log['Gasolina regular'], order=(4,4,1), seasonal_order=(2,1,0,12), enforce_stationarity=False, enforce_invertibility=False)
# resultado_m441 = modelo441.fit()

# pred = resultado_m441.get_prediction(start=test_consumo['Gasolina regular']['2021':].index[0], dynamic=False)
# pred_ci = pred.conf_int()
# ax = test_consumo['Gasolina regular']['2021':].plot(label='observed')
# pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 4))
# ax.fill_between( pred_ci.iloc[:,0],
#                 pred_ci.iloc[:,1], color='k', alpha=.2)

# # plt.xlim(test_consumo[column]['2023':].index[0], pred_ci.index[-1])
# plt.title('Consumo de Gasolina regular')
# plt.legend()
# plt.show()

# pred = resultado_m121.get_prediction(start=test_precios['Superior']['2021':].index[0], dynamic=False)
# pred_ci = pred.conf_int()
# ax = test_precios['Superior']['2021':].plot(label='observed')
# pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 4))
# ax.fill_between( pred_ci.iloc[:,0],
#                 pred_ci.iloc[:,1], color='k', alpha=.2)

# # plt.xlim(test_consumo[column]['2023':].index[0], pred_ci.index[-1])
# plt.title('Precios de Gasolina superior')
# plt.legend()
# plt.show()

# app_ui = x.ui.page_sidebar(
#     x.ui.sidebar(
#         ui.input_selectize(
#             "xvar", "X variable", numeric_cols, selected="Bill Length (mm)"
#         ),
#         ui.input_selectize(
#             "yvar", "Y variable", numeric_cols, selected="Bill Depth (mm)"
#         ),
#         ui.input_checkbox_group(
#             "species", "Filter by species", species, selected=species
#         ),
#         ui.hr(),
#         ui.input_switch("by_species", "Show species", value=True),
#         ui.input_switch("show_margins", "Show marginal plots", value=True),
#     ),
#     x.ui.output_plot("scatter")
# )


# def server(input: Inputs, output: Outputs, session: Session):
#     @reactive.Calc
#     def filtered_df() -> pd.DataFrame:
#         """Returns a Pandas data frame that includes only the desired rows"""

#         # This calculation "req"uires that at least one species is selected
#         req(len(input.species()) > 0)

#         # Filter the rows so we only include the desired species
#         return df[df["Species"].isin(input.species())]

#     @output
#     @render.plot
#     def scatter():
#         """Generates a plot for Shiny to display to the user"""

#         # The plotting function to use depends on whether margins are desired
#         plotfunc = sns.jointplot if input.show_margins() else sns.scatterplot

#         plotfunc(
#             data=filtered_df(),
#             x=input.xvar(),
#             y=input.yvar(),
#             hue="Species" if input.by_species() else None,
#             hue_order=species,
#             legend=False,
#         )


# app = App(app_ui, server) 

app_ui = ui.page_fixed(
    ui.h2("Análisis de Consumo y Producto"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_radio_buttons("opcion", "Selecciona una opción:",
                dict(consumo="Consumo", producto="Producto")
            ),
        ),
        ui.panel_main(
            ui.output_plot("plot")
        )
    )
)

def server(input, output, session):
    @output
    @render.plot
    def plot():
        fig, ax = plt.subplots(figsize=(14, 4))
        print(input.opcion())
        
        if input.opcion() == 'consumo':
            # Tu código para graficar "consumo" va aquí ...
            consumo_log_diff_act = consumo_log['Gasolina regular'].diff().diff()
            consumo_log_diff_act.dropna(inplace=True)
            modelo441 = SARIMAX(consumo_log['Gasolina regular'], order=(4,4,1), seasonal_order=(2,1,0,12), enforce_stationarity=False, enforce_invertibility=False)
            resultado_m441 = modelo441.fit()
            pred = resultado_m441.get_prediction(start=test_consumo['Gasolina regular']['2021':].index[0], dynamic=False)
            pred_ci = pred.conf_int()
            line1, = ax.plot(test_consumo['Gasolina regular']['2021':].index, test_consumo['Gasolina regular']['2021':], label='Observed')
            line2, = ax.plot(pred.predicted_mean.index, pred.predicted_mean, label='One-step ahead Forecast', alpha=.7)
            ax.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], color='k', alpha=.2)
            ax.set_title('Consumo de Gasolina regular')
        
        elif input.opcion() == 'producto':
            # Tu código para graficar "producto" va aquí ...
            precios_log_diff_act = precios_log['Superior'].diff().diff()
            precios_log_diff_act.dropna(inplace=True)
            modelo121 = SARIMAX(precios_log['Superior'], order=(1,2,1), seasonal_order=(1,2,1,12), enforce_stationarity=False, enforce_invertibility=False)
            resultado_m121 = modelo121.fit()
            pred = resultado_m121.get_prediction(start=test_precios['Superior']['2021':].index[0], dynamic=False)
            pred_ci = pred.conf_int()
            line1, = ax.plot(test_precios['Superior']['2021':].index, test_precios['Superior']['2021':], label='Observed')
            line2, = ax.plot(pred.predicted_mean.index, pred.predicted_mean, label='One-step ahead Forecast', alpha=.7)
            ax.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], color='k', alpha=.2)
            ax.set_title('Precios de Gasolina superior')

        ax.legend(handles=[line1, line2])  # Asegurando que las líneas estén en la leyenda
        return fig

app = App(app_ui, server)
app.run()