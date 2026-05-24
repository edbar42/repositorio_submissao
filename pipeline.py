"""
pipeline.py — Função obrigatória para o corretor automático.

Grupo D — Trabalho 2 de Ciência de Dados.

A função `prever_precos(caminho_arquivo_teste)` é a única interface chamada
pelo corretor. A assinatura NÃO PODE SER ALTERADA, pois o sistema de
correção a invoca por nome e posição de parâmetro.

Arquitetura:
    1. Lê o CSV de teste (mesmo formato de treino.csv, com Id, sem SalePrice).
    2. Aplica as transformações estruturais determinísticas (idênticas ao
       treinamento): drop de Id, cast de MSSubClass para string, drop das
       4 colunas colineares identificadas na EDA.
    3. Carrega o `modelo.joblib` (sklearn Pipeline serializado contendo
       ColumnTransformer + XGBoost + TransformedTargetRegressor com
       log1p/expm1).
    4. Invoca `modelo.predict()`. O TransformedTargetRegressor desfaz
       automaticamente a transformação `log1p` aplicada no alvo durante
       o treino, devolvendo as predições já em escala de dólares.
    5. Aplica clip em 0 (RMSLE rejeita valores negativos).
"""
import os

import joblib
import numpy as np
import pandas as pd


# Path absoluto baseado na localização deste script — robusto a qualquer
# diretório de trabalho do corretor automático.
_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELO_PATH = os.path.join(_DIR, "modelo.joblib")

# Colunas removidas por multicolinearidade durante o treino (limiar |corr|>0,8).
# Devem ser removidas também na inferência para casar com o ColumnTransformer.
_COLS_COLINEARES = ["GarageArea", "1stFlrSF", "TotRmsAbvGrd", "GarageYrBlt"]


def prever_precos(caminho_arquivo_teste):
    """
    Gera predições de SalePrice para o CSV fornecido.

    Parâmetros
    ----------
    caminho_arquivo_teste : str
        Caminho local para o arquivo CSV de teste. O arquivo deve seguir o
        mesmo schema de `treino.csv` (mesmas 79 colunas explicativas + Id),
        sem a coluna alvo `SalePrice`.

    Retorna
    -------
    np.ndarray
        Array 1D de predições contínuas (float, em USD), uma por linha do
        CSV, RIGOROSAMENTE na mesma ordem das linhas de entrada. Valores
        garantidamente não-negativos (clip em 0).
    """
    # ---- 1. Carregar dados de teste ----
    df = pd.read_csv(caminho_arquivo_teste)

    # ---- 2. Transformações estruturais (idênticas ao treino) ----
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    if "MSSubClass" in df.columns:
        # Códigos categóricos nominais armazenados como int — vide §3.6 do relatório.
        df["MSSubClass"] = df["MSSubClass"].astype(str)

    cols_colineares_presentes = [c for c in _COLS_COLINEARES if c in df.columns]
    if cols_colineares_presentes:
        df = df.drop(columns=cols_colineares_presentes)

    # Defensivo: se por algum motivo SalePrice estiver no arquivo de teste,
    # remover para não confundir o ColumnTransformer.
    if "SalePrice" in df.columns:
        df = df.drop(columns=["SalePrice"])

    # ---- 3. Carregar modelo treinado ----
    if not os.path.exists(_MODELO_PATH):
        raise FileNotFoundError(
            f"Arquivo de modelo não encontrado em '{_MODELO_PATH}'. "
            f"Regenere com 'python treinar.py' antes de submeter."
        )
    modelo = joblib.load(_MODELO_PATH)

    # ---- 4. Predizer ----
    # TransformedTargetRegressor reverte automaticamente o log1p via expm1.
    # Resultado já vem em escala de dólares.
    predicoes = modelo.predict(df)

    # ---- 5. Garantir não-negatividade ----
    # RMSLE = sqrt(mean( (log(1+pred) - log(1+true))^2 )); preds <= -1 quebram log1p.
    predicoes = np.clip(predicoes, a_min=0.0, a_max=None)

    return predicoes


# ---------------------------------------------------------------------------
# Bloco de teste local — NÃO é executado pelo corretor automático
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    teste_local = os.path.join(_DIR, "teste_publico.csv")

    print("=== Validação local do pipeline.py ===\n")

    if not os.path.exists(teste_local):
        print(f"[Aviso] '{teste_local}' não encontrado para validação local.")
    else:
        t0 = time.perf_counter()
        try:
            preds = prever_precos(teste_local)
            elapsed = time.perf_counter() - t0

            print(f"✓ Sucesso. {len(preds)} predições geradas em {elapsed:.2f}s.\n")
            print(f"  Tipo:    {type(preds).__name__}")
            print(f"  Shape:   {preds.shape}")
            print(f"  Dtype:   {preds.dtype}")
            print(f"  Min:     ${preds.min():>12,.2f}")
            print(f"  Mediana: ${float(np.median(preds)):>12,.2f}")
            print(f"  Max:     ${preds.max():>12,.2f}")
            print(f"  Média:   ${preds.mean():>12,.2f}")
            print(f"\n  Primeiras 5 predições: {np.round(preds[:5], 0)}")

            # Se o arquivo de teste local tiver SalePrice (útil só para teste_publico),
            # computa o RMSLE como sanity check.
            df_check = pd.read_csv(teste_local)
            if "SalePrice" in df_check.columns:
                from sklearn.metrics import mean_squared_log_error

                rmsle = float(np.sqrt(mean_squared_log_error(df_check["SalePrice"], preds)))
                print(f"\n  RMSLE local (com SalePrice presente no CSV): {rmsle:.5f}")
                print(f"  Baseline do professor: 0.17543")

            # Validação do tempo limite
            if elapsed > 60:
                print(f"\n  ⚠ ATENÇÃO: tempo de inferência ({elapsed:.1f}s) > 60s "
                      f"(limite do corretor).")
            else:
                print(f"\n  ✓ Tempo de inferência dentro do limite de 60s.")

        except Exception:
            import traceback

            print("✗ ERRO durante a execução do pipeline:\n")
            traceback.print_exc()
