# Análise de Reclamações Financeiras - Datathon Fase 5

Este repositório contém a análise e o modelo de Deep Learning desenvolvidos para o desafio da Fase 5 do Datathon, focado na classificação de sentimentos de reclamações financeiras e na identificação das principais dores dos clientes.

## Conteúdo do Repositório:
*   `analise_reclamacoes_financeiras.ipynb`: O notebook Jupyter com todo o código, explicações e visualizações.
*   `complaints_143k.csv`: A base de dados utilizada, contendo reclamações de consumidores sobre produtos e serviços financeiros.
*   `requirements.txt`: Lista das bibliotecas Python necessárias para rodar o projeto.
*   `.gitignore`: Arquivo para ignorar itens desnecessários ao subir para o Git.

## Como Rodar o Projeto:

### Opção 1: Google Colab (Recomendado para facilidade)
1.  Faça o upload do arquivo `complaints_143k.csv` para a raiz do seu Google Drive.
2.  Abra o `analise_reclamacoes_financeiras.ipynb` no Google Colab.
3.  Na primeira célula de código, descomente as linhas para montar o Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
4.  Certifique-se de que o caminho para o CSV esteja correto: `df = pd.read_csv('/content/drive/MyDrive/complaints_143k.csv')`.
5.  Instale as dependências no Colab executando a seguinte linha em uma célula de código:
    ```bash
    !pip install -r requirements.txt
    ```
6.  Execute as células do notebook sequencialmente.

### Opção 2: Ambiente Local (VS Code, Jupyter Notebook)
1.  Clone este repositório para o seu computador.
2.  Certifique-se de ter Python (versão 3.x) instalado.
3.  Instale as dependências usando o `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
4.  Abra o `analise_reclamacoes_financeiras.ipynb` no VS Code (com a extensão Jupyter) ou em um Jupyter Notebook.
5.  Execute as células uma a uma.

## Visão Geral da Análise:
-   **Pré-processamento de Texto**: Limpeza dos dados textuais, incluindo remoção de pontuações, stopwords e termos anonimizados (como 'XXXX'), além de lematização para padronizar as palavras.
-   **Classificação de Sentimento**: Criação de uma variável alvo de sentimento (Positivo/Negativo) baseada na resposta da empresa à reclamação, utilizando uma abordagem heurística.
-   **Modelagem com Deep Learning**: Treinamento de um modelo LSTM (Long Short-Term Memory) para classificar o sentimento das reclamações.
-   **Análise das Dores dos Clientes**: Geração de visualizações como Nuvens de Palavras e Gráficos de Frequência para identificar os temas mais recorrentes de insatisfação por categoria de produto.

Este projeto visa fornecer insights sobre as principais preocupações dos consumidores no setor financeiro, utilizando técnicas de Processamento de Linguagem Natural (NLP) e Deep Learning.
