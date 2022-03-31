import pandas as pd
import os
import numpy as np
from math import ceil
from osgeo import gdal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import seaborn as sns

def truncateVar(data=None, col=None):
    """
        truncateVar(data :: dataframe, col :: string)

    Realiza o truncamento de uma variável radiométrica col, tendo como referência os limiares inferior (lower) e
    superior (upper).

    Parâmetros:
    - data : dataframe que contém a variável radiométrica de interesse
    - col : variável a ser truncada

    Retorna:
    - Variável truncada

    """
    # -----------------------------------------------------------------------------------------------------------
    # Função auxiliar para a etapa de limpeza dos dados
    # -----------------------------------------------------------------------------------------------------------

    lower = data[col].mean() / 10
    upper = data[col].quantile(0.995)
    var_trunc = []

    for v in data[col]:
        if v <= lower:
            v = lower
            var_trunc.append(v)
        elif v >= upper:
            v = upper
            var_trunc.append(v)
        else:
            var_trunc.append(v)

    return pd.Series(var_trunc)

# -----------------------------------------------------------------------------------------------------------
# Função auxiliar para exportação dos resultados como raster
# -----------------------------------------------------------------------------------------------------------

def df2Raster(df, filename, col=None):
    """
        df2Raster(df :: dataframe, filename :: string, col :: string)

    Converte um dataframe em raster (.tif). SIRGAS2000 UTM Zona 23S é o sistema de referência adotado. A resolução
    do raster é de 62.5 m x 62.5 m.

    Parâmetros:
    - df : dataframe(n, p). Deve conter as coordenadas (x e Y) e uma ou mais variáveis de interesse
    - filename : nome do raster. Não é necessário adicionar o sufixo '.tif'
    - col : variável de interesse

    Retorna:
    - raster correspondente aos dados de entrada

    """

    # Diretórios
    csv_path = f"output/rasters/{filename}.csv"
    vrt_path = f"output/rasters/{filename}.vrt"
    tif_path = f"output/rasters/{filename}.tif"

    # Ordenamento do dataframe no padrão GDAL
    df = df[['X', 'Y', col]]
    df_sorted = df.sort_values(by=['Y', 'X'], ascending=[False, True])

    # CSV temporário
    df_sorted.to_csv(csv_path, index=False)

    # VRT temporário
    f = open(vrt_path, "w")
    f.write(f"<OGRVRTDataSource>\n \
        <OGRVRTLayer name=\"{filename}\">\n \
            <SrcDataSource>{csv_path}</SrcDataSource>\n \
            <GeometryType>wkbPoint</GeometryType>\n \
            <GeometryField encoding=\"PointFromColumns\" x=\"X\" y=\"Y\" z=\"{col}\"/>\n \
        </OGRVRTLayer>\n \
</OGRVRTDataSource>")
    f.close()

    # Conversão em raster
    r = gdal.Rasterize(tif_path, vrt_path, outputSRS="EPSG:31983",
                       xRes=62.5, yRes=-62.5, attribute=col, noData=np.nan)
    r = None

    # Remoção dos arquivos temporários
    os.remove(vrt_path)
    os.remove(csv_path)

# -----------------------------------------------------------------------------------------------------------
# Funções auxiliares para predições
# -----------------------------------------------------------------------------------------------------------

def createPredTable(dic_ŷ_train, dic_ŷ_test, train, test):
    """
        createPredTable(dic_ŷ_train :: dict, dic_ŷ_test :: narray,
                        train :: dataframe, test :: dataframe)

    Retorna um dataframe com as coordenadas e as predições de cada modelo treinado.

    Parâmetros:
    - dic_ŷ_train : dicionário com as predições de cada modelo para o conjunto de treino
    - dic_ŷ_test : dicionário com as predições de cada modelo para o conjunto de teste
    - train : dataframe (t, p) representativo dos dados de treino
    - test : dataframe (n-t, p) representativo dos dados de teste

    Retorna:
    - df_pred : dataframe(n, 9) com as coordenadas e as predições de cada modelo

    """

    train_coords = train[['Row', 'Column']]
    test_coords = test[['Row', 'Column']]
    df_pred = pd.concat([train_coords, test_coords])

    for model in dic_ŷ_test.keys():
        ŷ_train = list(dic_ŷ_train[model])
        ŷ_test = list(dic_ŷ_test[model])
        map_labels = ŷ_train + ŷ_test
        df_pred['Litology'] = map_labels

    return df_pred


def createMissClassifTable(df_pred, y_train, y_test):
    """
        createMissClassifTable(df_pred :: dataframe, y_train :: narray, y_test :: narray)

    Retorna um dataframe com as coordenadas e as inconsistências entre o mapa geológico e cada mapa preditivo.
    As colunas de inconsistências por modelo são binárias, de modo que 1 simboliza inconsistência entre os mapas.

    Parâmetros:
    - df_pred : dataframe (n, 9) representativo das predições de cada modelo
    - y_train : narray (t, ) representativo dos labels de treino
    - y_test : narray (n-t, ) representativo dos labels de teste

    Retorna:
    - df_miss : dataframe(n, 9) com as coordenadas e as inconsistências apresentadas por cada modelo

    """

    model_list = df_pred.columns[2:]
    true_labels = list(y_train) + list(y_test)
    df_miss = df_pred[['X', 'Y']]

    for model in model_list:
        diff_list = true_labels - df_pred[model]
        miss_list = []

        for diff in diff_list:
            if diff == 0:
                miss_list.append(0)
            else:
                miss_list.append(1)

        df_miss[model] = miss_list

    return df_miss

def createPredProbaTable(pr_ŷ_train, pr_ŷ_test, train, test):
    """
        createPredProbaTable(pr_ŷ_train :: narray, pr_ŷ_test :: narray,
                             train :: dataframe, test :: dataframe)

    Retorna um dataframe com as probabilidades preditas para cada uma das 6 classes (unidades).

    Parâmetros:
    - pr_ŷ_train : narray (t, 6) representando as predições probabilísticas para cada uma das classes
    no conjunto de treino
    - pr_ŷ_test : narray (n-t, 6) representando as predições probabilísticas para cada uma das classes
    no conjunto de teste
    - train : dataframe (t, p) representativo dos dados de treino
    - test : dataframe (n-t, p) representativo dos dados de teste

    Retorna:
    - df_proba_pred : dataframe (n, 8) com as coordenadas e probabilidades para cada uma das classes

    """

    litho_list = ['MAcgg', 'PP3csbg', 'PP34b', 'PP4esjc', 'PP4esb', 'PP4egm']
    train_coords = train[['X', 'Y']]
    test_coords = test[['X', 'Y']]
    df_proba_pred = pd.concat([train_coords, test_coords])
    pr_ŷ = np.concatenate([pr_ŷ_train, pr_ŷ_test])
    i = 0

    for litho in litho_list:
        df_proba_pred[litho] = pr_ŷ[:, i]
        i += 1

    return df_proba_pred

def InformationEntropy(pr_ŷ_train, pr_ŷ_test, train, test):
    """
        InformationEntropy(pr_ŷ_train :: narray, pr_ŷ_test :: narray,
                           train :: dataframe, test :: dataframe)

    Retorna um dataframe com as coordenadas e valores de entropia da informação (de Shannon). Probabilidades
    nulas são ignoradas para o cálculo da entropia.

    Parâmetros:
    - pr_ŷ_train : narray (t, 6) representando as predições probabilísticas para cada uma das classes
    do conjunto de treino
    - pr_ŷ_test : narray (n-t, 6) representando as predições probabilísticas para cada uma das classes
    do conjunto de teste
    - train : dataframe (t, p) representativo dos dados de treino
    - test : dataframe (n-t, p) representativo dos dados de teste

    Retorna:
    - df_entropy : dataframe(n, 3) com as coordenadas e entropia

    """

    train_coords = train[['X', 'Y']]
    test_coords = test[['X', 'Y']]
    df_entropy = pd.concat([train_coords, test_coords])

    pr_ŷ = np.concatenate([pr_ŷ_train, pr_ŷ_test])

    size = len(df_entropy)
    entropy_list = []

    for i in range(size):
        pred_prob = pr_ŷ[i, :]
        h = 0

        for p in pred_prob:
            if p != 0:
                h += - (p * (np.log2(p)))

        entropy_list.append(h)

    df_entropy['ENTROPY'] = entropy_list

    return df_entropy

# ---------------------------------------------------------------------------------------------------
# Classe auxiliar para realização da PCA personalizada
# ---------------------------------------------------------------------------------------------------

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

    def __init__(self, n_components = 3, mask = None):
        self.n_components = n_components
        self.mask = mask

    def fit(self, X, y = None):
        self.pca = PCA(n_components = self.n_components)
        mask = self.mask
        mask = self.mask if self.mask is not None else slice(None)
        self.pca.fit(X[:, mask])
        return self

    def transform(self, X, y = None):
        mask = self.mask if self.mask is not None else slice(None)
        pca_transformed = self.pca.transform(X[:, mask])
        if self.mask is not None:
            remaining_cols = np.delete(X, mask, axis = 1)
            return np.hstack([remaining_cols, pca_transformed])
        else:
            return pca_transformed

# -----------------------------------------------------------------------------------
# Funções auxiliares estatísticas
# -----------------------------------------------------------------------------------

def sumStats(df=None):
    """
        sumStats(df :: dataframe)

    Gera um sumário estatístico completo de um dataframe df. As estatísticas incluem
    medidas de tendência central (X̅ e P50%), medidas de posição (Min, P10%, P99.5% e Max),
    medidas de dispersão (Amp, S², S e Cᵥ) e medida de forma (Skew).

    Parâmetro:
    - df : dataframe com as features utilizadas para o cálculo do sumário estatístico

    Retorna:
    - stats : dataframe com o sumário estatístico

    """

    stats = df.describe(percentiles=[0.1, 0.5, 0.995]).T

    stats['Amp'] = (df.max() - df.min()).tolist()  # amplitude (max = min)
    stats['S²'] = df.var().tolist()  # variância
    stats['Cᵥ'] = (df.std() / df.mean()).tolist()  # coeficiente de variação
    stats['Skew'] = df.skew().tolist()  # coeficiente de assimetria

    stats = stats.rename(columns={'mean': 'X̅', 'std': 'S', 'min': 'Min', 'max': 'Max'})

    return stats[['X̅', '50%', 'Min', '10%', '99.5%', 'Max', 'Amp', 'S²', 'S', 'Cᵥ', 'Skew']]

def plotBoxplots(df, cols=None):
    """
        plotBoxplots(df :: dataframe, cols :: list)

    Plota n boxplots, sendo n o número de features presentes na lista cols.

    Parâmetros:
    - df : dataframe com os dados
    - cols : lista de features

    Retorna:
    - Um boxplot por feature presente na lista cols

    """

    n = len(cols)
    fig, axs = plt.subplots(n, 1, figsize=(10, n * 2))

    for ax, f in zip(axs, cols):
        sns.boxplot(y=f, x='COD', data=df, ax=ax)
        if f != cols[n - 1]:
            ax.axes.get_xaxis().set_visible(False)

# -----------------------------------------------------------------------------------------------------------
# Função auxiliar para divisão entre dados de treino e teste
# -----------------------------------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------------------------
# Funções auxiliares para geração de reports de validação
# ---------------------------------------------------------------------------------------------------

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
            # média dos scores de validação cruzada
            μ_cv = round(cv_scores.mean(), 3)
            metrics.append(μ_cv)

        df_val[model] = metrics

    return df_val

def testReport(dic_ŷ, y_test):
    """
        testReport(dic_ŷ :: dict, y_test :: narray)

    Retorna um report com as métricas resultantes do conjunto de teste por modelo. As métricas incluem
    acurácia, F1-score, precisão, revocação (ponderadas pelo número de exemplos de cada unidade).

    Parâmetros:
    - dic_ŷ : dicionário com as predições de cada modelo
    - y_test : narray (n-t, ) com o target do conjunto de teste

    Retorna:
    - df_metrics : dataframe com as métricas resultantes do conjunto de teste por modelo

    """

    model_list = dic_ŷ.keys()
    metric_list = ['f1_weighted', 'precision_weighted', 'recall_weighted', 'accuracy']
    df_metrics = pd.DataFrame(columns=model_list, index=metric_list)

    for ŷ in dic_ŷ:
        metrics = []
        # f1-score
        f1 = round(f1_score(y_test, dic_ŷ[ŷ], average='weighted'), 3)
        metrics.append(f1)
        # precisão
        p = round(precision_score(y_test, dic_ŷ[ŷ], average='weighted'), 3)
        metrics.append(p)
        # revocação
        r = round(recall_score(y_test, dic_ŷ[ŷ], average='weighted'), 3)
        metrics.append(r)
        # acurácia
        acc = round(accuracy_score(y_test, dic_ŷ[ŷ]), 3)
        metrics.append(acc)

        df_metrics[ŷ] = metrics

    return df_metrics

def plotModelScores(report, models, col, ec):
    """
        plotModelScores(report :: dataframe, models :: list, col :: string, ec :: string)

    Plota um gráfico de barras dos modelos organizados em ordem descrescente com relação os seus
    respectivos F1-scores.

    Parâmetros:
    - report : dataframe de report das performances
    - models : lista com os nomes de cada um dos modelos
    - col : cor do gráfico de barras
    - ec : cor da borda do gráfico

    Retorna:
    - Gráfico de barras com os valores F1-score de para cada um dos modelos

    """

    f1_scores = list(report.iloc[0, :])
    dic_f1_scores = {'MODEL': models, 'F1SCORE': f1_scores}
    df_mean_scores = pd.DataFrame(dic_f1_scores).sort_values('F1SCORE', ascending=False)

    plt.figure(figsize=(9, 4))
    plt.bar('MODEL', 'F1SCORE', data=df_mean_scores, color=col, edgecolor=ec)
    plt.ylabel('F1-score', size=14)
    plt.yticks(np.arange(0.0, 1.1, 0.1))

    plt.tight_layout();
