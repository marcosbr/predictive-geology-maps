import pandas as pd
import numpy as np
from math import ceil
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score



class MaskedPCA(BaseEstimator, TransformerMixin):
    """
        MaskedPCA(n_components :: int, mask :: narray)

    Classe que realiza uma Análise de Componentes Principais (ACP) apenas das features definidas pelo
    parâmetro mask. O número de componentes principais pode ser informado por meio do parâmetro
    n_components.

    Parâmetros:
    - n_components : número (int) de componentes principais
    - mask : narray (n, ), sendo n o número de features utilizadas na PCA. Este parâmetro indica os
    índices das colunas das features

    Retorna:
    - instância da classe MaskedPCA

    """

    def __init__(self, n_components=3, mask=None):
        self.n_components = n_components
        self.mask = mask
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X, y=None):
        mask = self.mask if self.mask is not None else slice(None)
        self.pca.fit(X[:, mask])
        return self

    def transform(self, X, y=None):
        mask = self.mask if self.mask is not None else slice(None)
        pca_transformed = self.pca.transform(X[:, mask])
        if self.mask is not None:
            remaining_cols = np.delete(X, mask, axis=1)
            return np.hstack([remaining_cols, pca_transformed])
        else:
            return pca_transformed

def customTrainTestSplit(df, feat_list, coords_list, samp_per_class=100, threshold=0.7, coords=False):
    """
        customTrainTestSplit(df :: dataframe, feat_list :: list, coords_list :: list,
                             samp_per_class :: int, threshold = float, coords :: bool)

    Realiza a divisão dos dados entre treino e teste. O conjunto de treino é obtido a partir de uma amostragem
    aleatória de samp_per_class exemplos por unidade litoestratigráfica. Caso uma unidade apresente um número
    de exemplos menor que samp_per_class, uma porcentagem de suas instâncias são aleatoriamente amostradas, sendo
    essa porcentagem definida pelo parâmetro threshold.

    Parâmetros:
    - df : dataframe (n, m) com os dados brutos
    - feat_list : lista de features presentes em df
    - coords_list : lista de coordenadas presentes em df
    - samp_per_class : número (int) de exemplos amostrados por unidade (default = 100)
    - threshold : porcentagem de exemplos que serão amostrados, caso uma unidade
    apresente um número de ocorrências inferior a samp_per_class (default = 0.7)
    - coords : se True, retorna as coordenadas X e Y de treino e teste (default = false)

    Retorna:
    - X_train : narray (t, m) com as features do conjunto de treino
    - y_train : narray (t, ) com o target do conjunto de treino
    - coord_train : narray (t, 2) com as coordenadas do conjunto de treino (apenas se
    coords = True)
    - X_test : narray (n-t, m) com as features do conjunto de teste
    - y_test : narray (n-t, ) com o target do conjunto de teste
    - coord_test : narray (n-t, 2) com as coordenadas do conjunto de teste (apenas se
    coords = True)

    """
    np.random.seed(42)
    # embaralhando dataframe
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    # lista classes/unidades
    classes = df_shuffled['TARGET'].unique()
    # dataframe vazio de treino
    train = pd.DataFrame()

    for c in classes:
        unid = df_shuffled[df_shuffled['TARGET'] == c]
        len_unid = len(unid)

        if len_unid <= samp_per_class:
            𝒮 = unid.sample(ceil(len_unid * threshold))
        else:
            𝒮 = unid.sample(samp_per_class)

        train = pd.concat([train, 𝒮])

    # embaralhando treino e teste
    test = df_shuffled.drop(train.index).sample(frac=1).reset_index(drop=True)
    train = train.sample(frac=1).reset_index(drop=True)

    # divisão treino e teste
    X_train, y_train, coord_train = train[feat_list].values, train['TARGET'].values, train[coords_list].values
    X_test, y_test, coord_test = test[feat_list].values, test['TARGET'].values, test[coords_list].values

    if coords:
        return X_train, y_train, coord_train, X_test, y_test, coord_test
    else:
        return X_train, y_train, X_test, y_test

def validationReport(pipeline, X_train, y_train, cv):
    """
        validationReport(pipeline :: pipeline, X_train :: narray, y_train :: narray, cv :: object)

    Retorna um report com as métricas resultantes da validação cruzada por modelo. As métricas incluem
    acurácia, F1-score, precisão, revocação (ponderadas pelo número de exemplos de cada unidade).

    Parâmetros:
    - pipeline : pipeline completa com as etapas de processamento até a instanciação do classificador
    - X_train : narray (t, m) das features de treino
    - y_train : narray (t, ) do target de treino
    - cv : objeto de validação cruzada

    Retorna:
    - df_val : dataframe com as métricas resultantes da validação cruzada por modelo

    """

    model_list = pipeline.keys()
    metric_list = ['f1_weighted', 'precision_weighted', 'recall_weighted', 'accuracy']
    df_val = pd.DataFrame(columns=model_list, index=metric_list)

    for model in model_list:
        metrics = []
        for metric in metric_list:
            cv_scores = cross_val_score(pipeline[model], X_train, y_train, scoring=metric, cv=cv)
            # Average of scores from cross validation
            mu_cv = round(cv_scores.mean(), 3)
            metrics.append(mu_cv)

        df_val[model] = metrics

    return df_val



