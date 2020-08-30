# Experimentos em Detecção de Fraudes

### Informações Importantes:

Este projeto foi financiado por uma parceria entre o [Instituto de Inovação, Pesquisa e Desenvolvimento Científico e Tecnológico do Amazonas - IPDEC](https://www.ipdec.org/) e o [Méliuz](https://www.meliuz.com.br/).

### Sobre:

Uma série de tarefas do mundo real estão estão associadas ao problema de bases de dados desbalanceadas. Detecção de fraudes em compras em cartões de crédito, detecção de ruídos em sensores e análise de comportamentos de usuários são alguns exemplos destas tarefas. Apesar da eficácia de grande parte dos algorítmos de aprendizagem de máquina existentes na literatua, boa parte desses algoritmos têm problemas ao trabalhar com dados desbalanceados, o que prejudica a inferência e, por consequência, a capacidade de gerar valor a partir destas previsões.

Este trabalho busca explorar e comparar o desempenho de técnicas de balanceamento de bases de dados utilizando 4 algoritmos de classificação em duas bases de dados desbalanceadas. Estes algoritmos foram [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decision%20tree#sklearn.tree.DecisionTreeClassifier), [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) e [K-Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) aplicados nas bases de dados do [Kaggle](https://www.kaggle.com/) [Creditcard Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) e [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit).

Além disso, buscou-se explorar abordagens utilizando aprendizagem profunda, porém estes experimentos estão incompletos. A ideia inicial destes experimentos era usar [Autoencoders](https://en.wikipedia.org/wiki/Autoencoder) como um modelo de detecção de anomalias. Um próximo passo seria explorar o uso de [modelos profundos geradores](https://arxiv.org/pdf/1803.09655.pdf) para balancear a base de dados e assim utilizar os modelos de classificação tradicionais.

### Resultados:

Os resultados obtidos ao utilizar métodos tradicionais de balanceamento de dados podem ser encontrados em `notebooks/exploratory/<dataset>/*-undersampling-oversampling.ipynb`, onde `<dataset>` se refere a um dos 2 datasets utilizados neste trabalho.

### TO-DOS:

- [ ] Reorganizar seção de resultados para incluí-los diretamente neste README.
- [ ] Continuar as abordagens sugeridas neste trabalho.
- [ ] Adicionar outras métricas para comparação dos resultados.