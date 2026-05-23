import pandas as pd
import joblib
import numpy as np
import os
from sklearn.metrics import mean_squared_log_error

def prever_precos(caminho_arquivo_teste):
    """
    Função obrigatória para o corretor automático.
    Lê o arquivo de teste, aplica o pré-processamento numérico e retorna as predições.
    
    Parâmetros:
    caminho_arquivo_teste (str): Caminho local para o arquivo CSV de teste.
    
    Retorna:
    np.array: As predições de preços (não negativas).
    """
    # 1. Leitura dos dados de teste
    df_teste = pd.read_csv(caminho_arquivo_teste)

    # 2. Pré-processamento (Espelhando o treinamento do baseline)
    # Selecionar apenas colunas numéricas
    X = df_teste.select_dtypes(include=[np.number])
    
    # Remover coluna Id se ela estiver presente
    if 'Id' in X.columns:
        X = X.drop(columns=['Id'])
        
    # Preencher valores nulos com 0
    X = X.fillna(0)

    # 3. Carregamento do modelo
    caminho_modelo = 'modelo_baseline.joblib'
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"O arquivo do modelo '{caminho_modelo}' não foi encontrado na raiz do projeto.")
        
    modelo = joblib.load(caminho_modelo)

    # 4. Alinhamento de colunas (Garante consistência com o treino)
    if hasattr(modelo, 'feature_names_in_'):
        X = X.reindex(columns=modelo.feature_names_in_, fill_value=0)

    # 5. Predição
    predicoes = modelo.predict(X)

    # 6. Pós-processamento
    # Garante valores >= 0 para evitar erro no cálculo do RMSLE
    predicoes_finais = np.clip(predicoes, a_min=0, a_max=None)

    return predicoes_finais

if __name__ == "__main__":
    # Bloco de teste local para o aluno
    arquivo_teste_exemplo = 'teste_publico.csv'
    
    print(f"--- Executando Validação Local do Pipeline ---")
    
    if not os.path.exists(arquivo_teste_exemplo):
        print(f"[Aviso] Arquivo '{arquivo_teste_exemplo}' não encontrado.")
        print(f"Dica: Crie um CSV fictício com as colunas numéricas para testar o script.")
    else:
        try:
            # Executa a predição
            resultados = prever_precos(arquivo_teste_exemplo)
            
            print("\n✅ Sucesso! O pipeline rodou corretamente.")
            print("-" * 30)
            print("Primeiras 5 predições:")
            print(resultados[:5])
            print("-" * 30)
            
            # Tenta calcular o RMSLE se a coluna alvo estiver no arquivo de teste
            df_val = pd.read_csv(arquivo_teste_exemplo)
            if 'SalePrice' in df_val.columns:
                y_true = df_val['SalePrice']
                # Cálculo do Root Mean Squared Log Error
                rmsle = np.sqrt(mean_squared_log_error(y_true, resultados))
                print(f"Métrica RMSLE Local: {rmsle:.5f}")
            else:
                print("[Nota] Coluna 'SalePrice' não encontrada no CSV. Cálculo do RMSLE pulado.")
            
        except Exception as e:
            print(f"\n❌ Erro encontrado no pipeline:")
            print(str(e))
            print("\nVerifique se o seu modelo espera as mesmas colunas presentes no CSV de teste.")
