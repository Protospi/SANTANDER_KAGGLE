# SANTANDER_KAGGLE


Competição Kaggle Transações de Clientes Santander


# 1 - Entendimento do Negócio


No ano 2019 o banco [Santander](https://www.santander.com.br/) publicou a competição [Predição das Transações de Clientes](https://www.kaggle.com/c/santander-customer-transaction-prediction/overview)  na comunidade de cientistas de dados [Kaggle](https://www.kaggle.com/), subsidiária da Google LLC. Apesar da competição já ter sido encerrada, os dados ainda estão disponibilizados assim como a submissão de predições ainda é permitida para que a comunidade participe da competição com objetivo de aprendizado.

A descrição da competição apresentou a missão do banco de oferecer serviços e produtos para indivíduos e empresas que forneçam saúde financeira e auxiliem seus clientes a alcançarem seus objetivos financeiros.

Esta competição foi aberta para a comunidade Kaggle pela equipe de ciência de dados do banco Santander. Diversos problemas de classificação como a satisfação do cliente, a compra de um produto  ou a inadimplência de um empréstimo são cotidianos para esta equipe. Com intuito de melhorar e validar seus modelos, a empresa desafiou a comunidade com a competição, O objetivo era identificar quais clientes farão uma transação específica no futuro independente da quantia movimentada.

 
## 1.1 Objetivos do Negócio
 
Os critérios do sucesso do negócio podem ser definidos na ocupação de uma colocação entre os 5 primeiros competidores o que levaria a premiação caso a competição ainda estivesse ativa. Pelo fato da competição já ter sido encerrada, foi possível observar a performance dos melhores 5 competidores premiados. Sendo que o quinto colocado atingiu a pontuação de 0.9244 enquanto o primeiro 0.9257.As premiações variam entre $5.000 dólares para o quinto colocado até $25.000 dólares  para o primeiro colocado.   

Os critérios de sucesso da modelagem serão determinados pela capacidade de predição do modelo sobre o banco de testes avaliado pela submissão destes resultados no formato de probabilidades das classes. Desta forma será possível avaliar a performance do modelo utilizando os resultados da competição como referência. 

 
## 1.2 Avaliação da Situação
 
Os dados disponibilizados pela equipe de ciência de dados do banco Santander para a competição consistem em 3 arquivos no formato de separação por vírgulas. Estes arquivos são referentes aos bancos de treino, teste e amostra para submissão e contem informações estruturadas na forma de tabelas. 
 
Para a execução do projeto foi utilizada uma máquina com processador Intel I7-9700 com 3GhZ e 16 GB de memória do tipo RAM no sistema Windows 10. A plataforma de desenvolvimento integrado RStudio, bem como a linguagem R foram utilizadas para carregar, preparar, explorar, desenhar, modelar os dados e produzir o relatório. 
 
 
## 1.3 Objetivos da Mineração de Dados
 
A competição consistiu na predição de transações para as variáveis do banco de testes. Para tanto a mineração de dados consistiu na implementação de modelo de aprendizado de máquinas supervisionados do tipo classificação para determinar se diante das novas observações das 200 variáveis do banco de testes, uma transação poderia ou não ser observada.

Neste sentido, foram utilizadas diversas técnicas de análise descritiva dos dados para explorar as características da variável binária transações e das variáveis explicativas atributos. Em seguida será realizada a investigação por valores atípicos nas distribuições, limpeza e engenharia de variáveis.

O algoritimo maquina de gradientes foi utilizado para a modelagem das transações. A métrica utilizada para determinar a qualidade dos modelos na validação será a área abaixo da curva (AUC) com intuito de evitar a tomada de decisão sobre uma probabilidade de corte para definir as classes da predição. A métrica da performance do modelo final sobre o banco de teste será avaliada pela classificação no painel de liderança da competição. 
 
 
## 1.4 Descrição do Projeto
 
Este projeto foi elaborado para ser apresentado como terceiro trabalho prático da disciplina Mineração de Dados do Departamento de Ciência da Computação da Universidade Federal de Minas Gerais. Técnicas de análise descritiva, exploratória e modelagem serão utilizadas no desenvolvimento da análise. O objetivo final do projeto será produzir um relatório segundo as especificações do manual __CRISP__ que será submetido para análise na plataforma Moodle no formato PDF. 

