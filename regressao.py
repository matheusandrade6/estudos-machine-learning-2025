# %%
import pandas as pd

df = pd.read_excel('data/dados_cerveja_nota.xlsx')

df.head()

# %%
from sklearn import linear_model

X =  df[['cerveja']] # Representa uma matriz (dataframe) de uma unica coluna
y = df['nota'] # Representa um vetor (séries)

# ISSO AQUI É O ML
reg = linear_model.LinearRegression(fit_intercept=True)  #fit_interception = a
reg.fit(X=X,y=y)

# %%
a, b = reg.intercept_, reg.coef_[0]
print(a,b)

# %%

# novas predições em cima dos mesmos dados
predict = reg.predict(X.drop_duplicates())

# %%
import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title('Relacao cerveja vs nota')
plt.xlabel('Cerveja')
plt.ylabel('Nota')

plt.plot(X.drop_duplicates()['cerveja'], predict)

plt.legend(['Observado', f'y = {a:.3f} + {b:.3f} * x'])
# %%
