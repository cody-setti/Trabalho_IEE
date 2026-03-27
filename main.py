# Cody Stefano Barham Setti 4856322
# Vinicius de Sa Ferreira   15491650

import pandas as pd
import numpy as np

DECIMAL = 2
END = '\n\n'
COLS = ['Sex', 'Age', 'Fare'] # Sexo, Idade e Preco dos bilhetes (convertido do sitema monetario Lsd (libras, xelins, pence) para Lp)
DATA = pd.read_csv('titanic.csv', usecols=COLS)
DATA = DATA.dropna(subset=['Age']) # Remove os que nao possuem idade determinada
DATA['Age'] = DATA['Age'].round() # Arredonda as idades incompletas

## a) Com base no rol, calcular a media, mediana e moda e compare-as
# Rol
rolSexo, rolIdade, rolPreco = np.sort(DATA[COLS[0]]), np.sort(DATA[COLS[1]]), np.sort(DATA[COLS[2]])

# Media
mediaIdade, mediaPreco = DATA[COLS[1]].mean(), DATA[COLS[2]].mean()
print(f'Media: Idades = {mediaIdade:.{DECIMAL}f}; Precos = {mediaPreco:.{DECIMAL}f}', end=END)

# Mediana
medianaIdade, medianaPreco = DATA[COLS[1]].median(), DATA[COLS[2]].median()
print(f'Mediana: Idades = {medianaIdade:.{DECIMAL}f}; Precos = {medianaPreco:.{DECIMAL}f}', end=END)

# Moda
modaSexo, modaIdade, modaPreco = DATA[COLS[0]].mode()[0], DATA[COLS[1]].mode()[0], DATA[COLS[2]].mode()[0]
print(f'Moda: Sexos = {modaSexo}; Idades = {modaIdade:.{DECIMAL}f}; Precos = {modaPreco:.{DECIMAL}f}', end=END)

## b) Com base no rol, calcular os quartis, interpretando-os
# Quartis das idades
idadeQ1, idadeQ2, idadeQ3 = DATA[COLS[1]].quantile(0.25), DATA[COLS[1]].quantile(0.50), DATA[COLS[1]].quantile(0.75)
print(f'Quartis das idades: Q1 = {idadeQ1:.{DECIMAL}f}; Q2 = {idadeQ2:.{DECIMAL}f}; Q3 = {idadeQ3:.{DECIMAL}f}', end=END)

# Quartis dos precos
precoQ1, precoQ2, precoQ3 = DATA[COLS[2]].quantile(0.25), DATA[COLS[2]].quantile(0.50), DATA[COLS[2]].quantile(0.75)
print(f'Quartis dos precos: Q1 = {precoQ1:.{DECIMAL}f}; Q2 = {precoQ2:.{DECIMAL}f}; Q3 = {precoQ3:.{DECIMAL}f}', end=END)

## c) Calcular
# c.1) a amplitude total
ampIdade, ampPreco = np.ptp(DATA[COLS[1]]), np.ptp(DATA[COLS[2]])
print(f'Amplitude: Idades = {ampIdade:.{DECIMAL}f}; Precos = {ampPreco:.{DECIMAL}f}', end=END)

# c.2) a amplitude interquartílica
ampiqIdade, ampiqPreco = (idadeQ3 - idadeQ1), (precoQ3 - precoQ1)
print(f'Amplitude interquartilica: Idades = {ampiqIdade:.{DECIMAL}f}; Precos = {ampiqPreco:.{DECIMAL}f}', end=END)

# c.3) o desvio-padrao
dpIdade, dpPreco = DATA[COLS[1]].std(), DATA[COLS[2]].std()
print(f'Desvio-padrao: Idades = {dpIdade:.{DECIMAL}f}; Precos = {dpPreco:.{DECIMAL}f}', end=END)

# c.4) o coeficiente de variacao;
cvIdade, cvPreco = (dpIdade/mediaIdade)*100, (dpPreco/mediaPreco)*100
print(f'Coeficiente de variacao: Idades = {cvIdade:.{DECIMAL}f}; Precos = {cvPreco:.{DECIMAL}f}', end=END)

# c.5) RESSALTAR ENTRE AS QUANTS. QUAL TEM MAIOR VARIABILIDADE
print('Quantitativa com maior variabilidade: ' + ('Idade' if (cvIdade > cvPreco) else 'Preco'), end=END)

## d) Com a comparação das medidas de tendência central, avaliar a simetria
print('Idade: ', end='')
if (mediaIdade-medianaIdade < -(10**(-DECIMAL))): print('Assimetria a esquerda', end=END)
elif (mediaIdade-medianaIdade > 10**(-DECIMAL)): print('Assimetria a direita', end=END)
else: print('Simetria', end=END)

print('Preco: ',end='')
if (mediaPreco-medianaPreco < -(10**(-DECIMAL))): print('Assimetria a esquerda')
elif (mediaPreco-medianaPreco > 10**(-DECIMAL)): print('Assimetria a direita')
else: print('Simetria')

# e) Sumarizar as informações em tabela

# f) Com base na informação tabelada, calcular
# f.1) a média

# f.2) a mediana

# f.3) a moda

# f.4) os quartis

# f.5) o desvio padrão

# f.6) COMPARE OS VALORES DAS MEDIDAS OBTIDAS COM O ROL


# g) Sumarizar as informações de cada variável graficamente (apresente dois gráficos)
#  'Sex': Barra & Pizza

#  'Age': 

# 'Fare': Histograma & boxplot