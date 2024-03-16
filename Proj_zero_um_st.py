# Importando as bibliotecas
# Abrir a conexão com o banco de dados
import gradio as gr
import streamlit as st
import numpy as np
import sqlite3
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from PIL import Image 
from sklearn import tree as tr

#-------------------------------------Funções----------------------------------

st.set_page_config(page_title='Projeto Zero Um', page_icon='💾', layout='wide') 

# ==================================================
# Dataset
# ==================================================

conn = sqlite3.connect('database.db')

consulta_atividade="""

    SELECT* 
    FROM flight_activity fa LEFT JOIN flight_loyalty_history flh ON (fa.loyalty_number=flh.loyalty_number)
    
    """
    
df_atividade=pd.read_sql_query(consulta_atividade, conn)

# ==================================================
# Selecionando Somente as Colunas que Contêm Números
# ==================================================
colunas=["year", "month", "flights_booked", "flights_with_companions", "total_flights", "distance",
            "points_accumulated", "salary", "clv", "loyalty_card"]

df_colunas_numericas=df_atividade.loc[:, colunas]

# ==================================================
# Preparação dos Dados
# ==================================================
df_dados_completos=df_colunas_numericas.dropna()

# ==================================================
# Layout no Streamlit
# ==================================================
st.title('Propensão de Compra do Cliente')
st.markdown(
    """ 
    ##### O objetivo principal é criar perfis personalizados para cada cliente com base em sua interação com os programas de fidelidade de voos: Star, Nova e Aurora. 
    ##### A partir desses perfis, desenvolvemos um modelo que pode sugerir o programa de fidelidade mais apropriado para novos clientes, prevendo suas chances de adesão a cada uma das opções disponíveis.
""")
st.subheader('', divider='gray') 

# ==================================================
# Machine Learning
# ==================================================
x_atributos=df_dados_completos.drop(columns="loyalty_card") 

y_rotulos=df_dados_completos.loc[:, "loyalty_card"]

# Definição do Algoritimo
modelo=tr.DecisionTreeClassifier(max_depth=3) 

# Treinamento do Algoritimo
modelo_treinado=modelo.fit(x_atributos, y_rotulos)

tr.plot_tree(modelo_treinado, filled=True)

# ==================================================
# Barra Lateral
# ==================================================
#image=Image.open('logo.png')
st.title('PROJETO 01 - DS')
#st.sidebar.image(image, width=250)

st.sidebar.title('Atributos dos Clientes')
st.sidebar.subheader('', divider='gray')

st.sidebar.subheader('Selecione o(s) atributos(s) que deseja analisar:')

def predict():
    
    year                    = st.sidebar.slider("Ano",  2017, 2018, step=1)
    salary                  = st.sidebar.slider("Rendimento Anual", 58486.00, 407228.00, step=0.1)
    clv                     = st.sidebar.slider("Valor Total para Todos os Voos", 2119.89, 83325.38, step=0.1)
    month                   = st.sidebar.slider("Mês", 1, 12, step=1)
    flights_booked          = st.sidebar.slider("Voos Reservados", 0, 21, step=1)
    flights_with_companions = st.sidebar.slider("Voos com Companhias", 0, 11, step=1)
    total_flights           = st.sidebar.slider("Total de Voos", 0, 32, step=1)
    distance                = st.sidebar.slider("Distância Percorrida no Voo (km)", 0, 6293, step=1)
    points_accumulated      = st.sidebar.slider("Pontos Acumulados", 0.00, 676.50, step=0.1)
          
    x_novo=np.array([[year, month  , flights_booked, flights_with_companions, 
                        total_flights, distance, points_accumulated, salary, clv]])

    previsao = modelo_treinado.predict_proba(x_novo)

    return {"Aurora": previsao[0][0], "Nova": previsao[0][1], "Star": previsao[0][2]}

user = predict()

st.markdown('#### Programa de Fidelidade Recomendado ao Usuário')

melhor_programa = max(user, key=user.get)
cores = {'Aurora': '#0000FF', 'Nova': '#76C5F7', 'Star': '#FF0000'}
texto_formatado = f"##### <span style='color:{cores[melhor_programa]}'>**{melhor_programa}: {user[melhor_programa]*100:.2f}%**</span>"
st.markdown(texto_formatado, unsafe_allow_html=True)

st.subheader('', divider='gray')


df_user = pd.DataFrame.from_dict(user, orient='index', columns=['previsao']).reset_index().rename(columns={'index': 'Programa'})

fig = px.bar(df_user, x='Programa', y='previsao', color='Programa', title='Probabilidades de Adesão ao Programa de Fidelidade', text_auto=True,
            labels={'Programa': 'Programa', 'previsao': 'Probabilidade'})

fig.update_traces(textfont_size=15, textangle=1, textposition="outside", cliponaxis=False, textfont_color='black')

fig.update_layout(yaxis_tickformat='.2%', yaxis_title='Probabilidade (%)')

st.plotly_chart(fig)


st.sidebar.subheader('', divider='gray')                
st.sidebar.subheader('Powered by: Marcela Passos da Silva')
st.sidebar.subheader('Discord: marcelapassos')
st.sidebar.subheader('Linkedin: https://linkedin.com/in/marcela-passos-da-silva')
st.sidebar.subheader('GitHub: github.com/Marcelaps1204') 
st.sidebar.subheader('Portfolio de Projetos: https://marcelaps1204.github.io/portfolio_projetos/')

