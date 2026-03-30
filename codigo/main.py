# SME0221 Introdução à Inferência Estatística
# --- TRABALHO PRÁTICO - ESTATÍSTICA DESCRITIVA ---
# Aluno 1: Cody Stefano Barham Setti  No. USP: 4856322
# Aluno 2: Vinícius de Sá Ferreira    No. USP: 15491650

# --- INICIALIZAÇÃO DO CÓDIGO ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['TeX Gyre Pagella'],
})


# Constantes
DECIMAL = 2
END = '\n\n'


# --- BANCO DE DADOS ---
DATA = pd.read_csv('codigo/titanic.csv', usecols=['Sex', 'Age', 'Fare'])
# OBS.: O preço dos bilhetes, originalmente vendidos no sistema Lsp (libras, xelins e pence), é dado em Lp (libras decimais)

# Higienização do banco de dados
DATA = DATA.dropna()                            # Remove individuos com dados faltantes
DATA['Age'] = DATA['Age'].round().astype(int)   # Torna idades valores inteiros


# --- ESTATÍSTICAS ---
# a) Com base no rol, calcular (comparando-os)
# a.1) a média
meanAge  = DATA['Age'].mean()
meanFare = DATA['Fare'].mean()
print(f'Média: Idades = {meanAge:.{DECIMAL}f}; Preços = {meanFare:.{DECIMAL}f}', end=END)

# a.2) a mediana
medianAge  = DATA['Age'].median()
medianFare = DATA['Fare'].median()
print(f'Mediana: Idades = {medianAge:.{DECIMAL}f}; Preços = {medianFare:.{DECIMAL}f}', end=END)

# a.3) a moda
modeSex  = DATA['Sex'].mode()[0]
modeAge  = DATA['Age'].mode()[0]
modeFare = DATA['Fare'].mode()[0]
print(f'Moda: Sexos = {modeSex}; Idades = {modeAge:.{DECIMAL}f}; Preços = {modeFare:.{DECIMAL}f}', end=END)

## b) Com base no rol, calcular os quartis, interpretando-os
# Quartis das idades
ageQ1, ageQ2, ageQ3 = DATA['Age'].quantile([0.25, 0.50, 0.75])
print(f'Quartis das idades: Q1 = {ageQ1:.{DECIMAL}f}; Q2 = {ageQ2:.{DECIMAL}f}; Q3 = {ageQ3:.{DECIMAL}f}', end=END)

# Quartis dos preços
fareQ1, fareQ2, fareQ3 = DATA['Fare'].quantile([0.25, 0.50, 0.75])
print(f'Quartis dos preços: Q1 = {fareQ1:.{DECIMAL}f}; Q2 = {fareQ2:.{DECIMAL}f}; Q3 = {fareQ3:.{DECIMAL}f}', end=END)

## c) Calcular
# c.1) a amplitude total
ampAge  = np.ptp(DATA['Age'])
ampFare = np.ptp(DATA['Fare'])
print(f'Amplitude: Idades = {ampAge:.{DECIMAL}f}; Preços = {ampFare:.{DECIMAL}f}', end=END)

# c.2) a amplitude interquartílica
ampiqAge  = (ageQ3 - ageQ1)
ampiqFare = (fareQ3 - fareQ1)
print(f'Amplitude interquartílica: Idades = {ampiqAge:.{DECIMAL}f}; Preços = {ampiqFare:.{DECIMAL}f}', end=END)

# c.3) o desvio-padrão
stdAge  = DATA['Age'].std()
stdFare = DATA['Fare'].std()
print(f'Desvio-padrão: Idades = {stdAge:.{DECIMAL}f}; Preços = {stdFare:.{DECIMAL}f}', end=END)

# c.4) o coeficiente de variação;
cvAge  = (stdAge/meanAge)*100
cvFare = (stdFare/meanFare)*100
print(f'Coeficiente de variacao: Idades = {cvAge:.{DECIMAL}f}; Precos = {cvFare:.{DECIMAL}f}', end=END)

# c.5) RESSALTAR ENTRE AS QUANTS. QUAL TEM MAIOR VARIABILIDADE
print('Quantitativa com maior variabilidade: ' + ('Idade' if (cvAge > cvFare) else 'Preço'), end=END)

## d) Com a comparação das medidas de tendência central, avaliar a simetria
print('Idade: ', end='')
if (meanAge - medianAge < -(10**(-DECIMAL))): print('Assimetria à esquerda', end=END)
elif (meanAge - medianAge > 10**(-DECIMAL)): print('Assimetria à direita', end=END)
else: print('Simetria', end=END)

print('Preço: ',end='')
if (meanFare - medianFare < -(10**(-DECIMAL))): print('Assimetria à esquerda', end=END)
elif (meanFare - medianFare > 10**(-DECIMAL)): print('Assimetria à direita', end=END)
else: print('Simetria', end=END)

# e) Sumarizar as informações em tabela
# TABELA DE FREQ. SIMPLES PARA 'SEXO'
SexDF = DATA['Sex'].value_counts()  # Obs: DF = data frame
SexDF = SexDF.reset_index()
SexDF.columns = ['Sexo', 'Frequência']
print(SexDF, end=END)

# TABELA DE FREQ. POR CLASSES PARA 'IDADE'
# Agrupamento em décadas
max_age = DATA['Age'].max()
age_bins = range(0, max_age + 11, 10)   # [0, 10, 20, ..., 80, 90]

# Atribuindo uma faixa etária a cada pessoa
DATA['Age_Class'] = pd.cut(DATA['Age'], bins=age_bins, right=False)
# `right=False` garante que as classes sejam da forma: [a,b)

AgeDF = DATA['Age_Class'].value_counts().sort_index()
AgeDF = AgeDF.reset_index()
AgeDF.columns = ['Faixa Etária', 'Frequência']
print(AgeDF, end=END)

# TABELA DE FREQ. POR CLASSES PARA 'PREÇO'
# Número de classes seguindo a regra de Sturges
n_obs = len(DATA)
k_sturges = int(np.ceil(1 + np.log2(n_obs)))

# Atribuindo uma faixa de preço de bilhete a cada pessoa
DATA['Fare_Class'] = pd.cut(DATA['Fare'], bins=k_sturges, right=False)
# Pandas divide automaticamente a faixa de valores de 'Fare' em 'k' intervalos de tamanhos iguais

FareDF = DATA['Fare_Class'].value_counts().sort_index()
FareDF = FareDF.reset_index()
FareDF.columns = ['Preço', 'Frequência']
print(FareDF, end=END)

# RECONSTRUÇÃO DO BANCO DE DADOS
sexes_reconst = np.repeat(SexDF['Sexo'].values, SexDF['Frequência'].values)
rSexes = pd.Series(sexes_reconst, name='Sex')   # Obs: r = reconstructed

age_midpoints = AgeDF['Faixa Etária'].apply(lambda x: x.mid).astype(float)
ages_reconst = np.repeat(age_midpoints.values, AgeDF['Frequência'].values)
rAges = pd.Series(ages_reconst, name='Age')     # Obs: r = reconstructed

fare_midpoints = FareDF['Preço'].apply(lambda x: x.mid).astype(float)
fares_reconst = np.repeat(fare_midpoints.values, FareDF['Frequência'].values)
rFares = pd.Series(fares_reconst, name='Fare')  # Obs: r = reconstructed

rDATA = pd.concat([rSexes, rAges, rFares], axis=1)

# f) Com base na informação tabelada, calcular
# f.1) a média
rmeanAge  = rDATA['Age'].mean()
rmeanFare = rDATA['Fare'].mean()
print(f'(R) Média: Idades = {rmeanAge:.{DECIMAL}f}; Preços = {rmeanFare:.{DECIMAL}f}', end=END)

# f.2) a mediana
rmedianAge  = rDATA['Age'].median()
rmedianFare = rDATA['Fare'].median()
print(f'(R) Mediana: Idades = {rmedianAge:.{DECIMAL}f}; Preços = {rmedianFare:.{DECIMAL}f}', end=END)

# f.3) a moda
rmodeSex  = rDATA['Sex'].mode()[0]
rmodeAge  = rDATA['Age'].mode()[0]
rmodeFare = rDATA['Fare'].mode()[0]
print(f'(R) Moda: Sexos = {rmodeSex}; Idades = {rmodeAge:.{DECIMAL}f}; Preços = {rmodeFare:.{DECIMAL}f}', end=END)

# f.4) os quartis
# Quartis das idades
rageQ1, rageQ2, rageQ3 = rDATA['Age'].quantile([0.25, 0.50, 0.75])
print(f'(R) Quartis das idades: Q1 = {rageQ1:.{DECIMAL}f}; Q2 = {rageQ2:.{DECIMAL}f}; Q3 = {rageQ3:.{DECIMAL}f}', end=END)

# Quartis dos preços
rfareQ1, rfareQ2, rfareQ3 = rDATA['Fare'].quantile([0.25, 0.50, 0.75])
print(f'(R) Quartis dos preços: Q1 = {rfareQ1:.{DECIMAL}f}; Q2 = {rfareQ2:.{DECIMAL}f}; Q3 = {rfareQ3:.{DECIMAL}f}', end=END)


# f.5) o desvio padrão
rstdAge  = rDATA['Age'].std()
rstdFare = rDATA['Fare'].std()
print(f'(R) Desvio-padrão: Idades = {rstdAge:.{DECIMAL}f}; Preços = {rstdFare:.{DECIMAL}f}', end=END)

# f.6) COMPARE OS VALORES DAS MEDIDAS OBTIDAS COM O ROL
# Vide relatório


# g) Sumarizar as informações de cada variável graficamente (apresente dois gráficos)
os.makedirs('output', exist_ok=True)

# GRÁFICOS PARA 'SEXO'
# Gráfico de Barras
plt.figure(figsize=(5, 4))
SexDF_plot = DATA['Sex'].value_counts()
SexDF_plot = SexDF_plot.rename(index={'male': 'Masculino', 'female': 'Feminino'})
SexDF_plot.plot(kind='bar', color=['#1f77b4', '#ff7f0e'], edgecolor='black')
plt.title('Frequência Absoluta por Sexo', fontsize=14)
plt.xlabel('Sexo', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('output/sexo_barra.png', format='png')
plt.close()

# Gráfico de Setores (Pizza)
plt.figure(figsize=(5, 4))
SexDF_plot.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#1f77b4', '#ff7f0e'])
plt.title('Proporção por Sexo', fontsize=14)
plt.ylabel('')  # Remove o rótulo do eixo Y que o pandas coloca por padrão
plt.tight_layout()
plt.savefig('output/sexo_pizza.png', format='png')
plt.close()

# GRÁFICOS PARA 'IDADE'
# Histograma
plt.figure(figsize=(5, 4))
plt.hist(DATA['Age'], bins=15, color='#2ca02c', edgecolor='black', alpha=0.7)
plt.title('Distribuição de Idades (Histograma)', fontsize=14)
plt.xlabel('Idade (anos)', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.tight_layout()
plt.savefig('output/idade_histograma.png', format='png')
plt.close()

# Boxplot
plt.figure(figsize=(5, 4))
plt.boxplot(DATA['Age'], vert=False, patch_artist=True, boxprops=dict(facecolor='#2ca02c', color='black'))
plt.title('Dispersão de Idades (Boxplot)', fontsize=14)
plt.xlabel('Idade (anos)', fontsize=12)
plt.yticks([1], ['Idade'])
plt.tight_layout()
plt.savefig('output/idade_boxplot.png', format='png')
plt.close()

# GRÁFICOS PARA 'PREÇOS'
# Histograma
plt.figure(figsize=(5, 4))
plt.hist(DATA['Fare'], bins=20, color='#d62728', edgecolor='black', alpha=0.7)
plt.title('Distribuição dos Preços dos Bilhetes (Histograma)', fontsize=14)
plt.xlabel('Preço (Lp)', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.tight_layout()
plt.savefig('output/preco_histograma.png', format='png')
plt.close()

# Boxplot
plt.figure(figsize=(5, 4))
plt.boxplot(DATA['Fare'], vert=False, patch_artist=True, boxprops=dict(facecolor='#d62728', color='black'))
plt.title('Dispersão dos Preços dos Bilhetes (Boxplot)', fontsize=14)
plt.xlabel('Preço (Lp)', fontsize=12)
plt.yticks([1], ['Preço'])
plt.tight_layout()
plt.savefig('output/preco_boxplot.png', format='png')
plt.close()

print("Gráficos gerados e salvos com sucesso na pasta 'output/'.", end=END)