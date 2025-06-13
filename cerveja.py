# %%
import pandas as pd

df = pd.read_excel('data/dados_cerveja.xlsx')

df.head()
# %%

features = ['temperatura', 'copo', 'espuma','cor']

target = 'classe'

x = df[features]
y = df[target]

# criando variáveis dummies pq scikit learn não cria árvore com strings

x = x.replace(
    {
        'mud': 1, 'pint': 2,
        'sim': 1, 'não': 0,
        'clara':0, 'escura':1,
    }
)

# %%

from sklearn import tree

model = tree.DecisionTreeClassifier() # instância criada para importar o tipo de modelo

model.fit(X=x, y=y)

# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model,
               feature_names=features,
               class_names=model.classes_,
               filled=True)
