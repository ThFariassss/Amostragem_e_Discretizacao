import pandas as pd
import numpy as np
import random

def carregar_dados(file_path):
    df = pd.read_csv(file_path)
    df['floor'] = df['floor'].replace('-', np.nan)
    return df

def amostra_aleatoria(df, n, reposicao=False):
    if n > len(df) and not reposicao:
        raise ValueError("Amostras sem reposição excedem o tamanho do dataset. Tente Novamente!!")
    indices = list(df.index)
    escolhidos = []
    for _ in range(n):
        escolhido = random.choice(indices)
        escolhidos.append(escolhido)
        if not reposicao:
            indices.remove(escolhido)
    return df.loc[escolhidos].reset_index(drop=True)

def amostra_estratificada(df, coluna, n):
    contagem = df[coluna].value_counts()
    proporcao = contagem / len(df)
    qtd_por_grupo = (proporcao * n).round().astype(int)
    diferenca = n - qtd_por_grupo.sum()
    if diferenca != 0:
        grupo_ajuste = qtd_por_grupo.idxmax() if diferenca > 0 else qtd_por_grupo.idxmin()
        qtd_por_grupo[grupo_ajuste] += diferenca
    amostras = []

    for valor, qtd in qtd_por_grupo.items():
        grupo = df[df[coluna] == valor]
        qtd = min(qtd, len(grupo))
        amostras.append(amostra_aleatoria(grupo, qtd, reposicao=False))
    return pd.concat(amostras).reset_index(drop=True)


def discretizar_area(df):
    df_clone = df.copy()
    def faixa_area(v):
        if v <= 50: return "PEQUENO"
        elif v <= 100: return "MÉDIO"
        else: return "GRANDE"
    df_clone["area_categoria"] = df_clone["area"].apply(faixa_area)
    return df_clone

def discretizar_total_ohe(df):
    df_clone = df.copy()
    valores = sorted(df_clone["total"])
    t = len(valores)
    q1 = valores[int(t * 0.25)]
    q2 = valores[int(t * 0.50)]
    q3 = valores[int(t * 0.75)]
    def faixa_total(v):
        if v <= q1: return "Q1"
        elif v <= q2: return "Q2"
        elif v <= q3: return "Q3"
        else: return "Q4"
    df_clone["total_categoria"] = df_clone["total"].apply(faixa_total)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        df_clone[f"total_cat_{q}"] = (df_clone["total_categoria"] == q).astype(int)
    return df_clone

if __name__ == "__main__":
    path = r"C:\Users\ThF\OneDrive\Área de Trabalho\Amostragem\alugueis.csv"
    df = carregar_dados(path)
    print("\n Amostragem Aleatória")
    amostra100 = amostra_aleatoria(df, 100, reposicao=False)

    print(amostra100.head())
    print("\n Amostragem Estratificada")
    amostra200 = amostra_estratificada(df, "city", 200)
    print(amostra200["city"].value_counts())

    print("\n Discretização Área")
    df_area = discretizar_area(df)
    print(df_area[["area", "area_categoria"]].head(10))
    print("\n Discretização Total ")
    
    df_total = discretizar_total_ohe(df)
    print(df_total[["total", "total_categoria"]].head())
