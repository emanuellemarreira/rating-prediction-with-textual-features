# Rating Prediction com Features Textuais

Este projeto aborda o desafio de rating prediction, que consiste em prever automaticamente a nota (rating) que um usuário atribuiria a um produto. Para isso, são utilizadas features extraídas dos comentários escritos pelos usuários, explorando o conteúdo textual como fonte de informação para os modelos de aprendizado de máquina.

## 📦 Dataset

O dataset conta com +50k de comentários de usuários da Amazon, todos escritos em português brasileiro.

Devido ao seu tamanho, o dataset está **compactado**.  
⚠️ Para que os experimentos funcionem, você **precisa descompactar** os dados na pasta `dataset/`.


## 🔍 Objetivos de Pesquisa (Research Questions)

Este projeto busca responder quatro perguntas de pesquisa principais:

**RQ1:** Existe um algoritmo de aprendizado de máquina ideal para *rating prediction* em diferentes categorias de produtos?

**RQ2:** Existe um grupo de *features* (características) mais relevante para essa tarefa?

**RQ3:** Como a combinação de *features* impacta o desempenho do classificador?

**RQ4:** A melhor configuração do modelo mantém a eficiência ao comparar diferentes categorias de produtos?

O código está organizado por perguntas de pesquisa.

## 💻 Como Usar

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/rating-prediction-with-textual-features.git
cd rating-prediction-with-textual-features
```

### 2. Criar o ambiente Conda

```bash
conda env create -f environment.yml
conda activate rating-prediction
```
### 3. Descompactar o dataset

### 4. Feature Extraction (resolver)

### 5. Rodar os experimentos (por pergunta de pesquisa)
```bash
python run-rq1.py
python run-rq2.py
python run-rq3.py
python run-rq4.py
```

