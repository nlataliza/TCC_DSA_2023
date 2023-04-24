#SCRIPT TCC DATA SCIENCE E ANALYTICS

#PACOTES
install.packages("readxl","dplyr","neuralnet", "tidyverse", "caret","NeuralNetTools")
library(readxl)
library(writexl)
library(dplyr)
library(neuralnet)
library(tidyverse)
library(caret)
library(NeuralNetTools)

#PREPARAÇÃO DA BASE DE DADOS
#abrindo base de dados originais
dados_brutos = read_excel("F:/Meu Drive/mba/rede neural/dados_brutos.xlsx")
dados_tratados = dados_brutos

#tratamento das variáveis
dados_tratados$exp = NULL #excluindo variável que nomeia o experimento
dados_tratados$solvent = recode(dados_tratados$solvent, "Anid" = "1", "MIBK" = "2", "DMC" = "3", "DEC" = "4", "MEC" = "5", "PPC" = "6") #convertendo o nome dos solventes em valores numéricos
dados_tratados$solvent = as.numeric(dados_tratados$solvent) #transformando a variável em numérica

#simplificando os dados para facilitar a construção do modelo
dados_tratados$selectivity_mono = dados_tratados$selectivity1 + dados_tratados$selectivityB + dados_tratados$selectivity4 + dados_tratados$selectivityA + dados_tratados$selectivityG 
#unindo variáveis por tipo de produto
dados_tratados$selectivity_di = dados_tratados$selectivityD1 + dados_tratados$selectivityD2 + dados_tratados$selectivityD3 
#unindo variáveis por tipo de produto

dados_tratados$selectivity1 = NULL #excluindo variaveis unidas
dados_tratados$selectivityA = NULL #excluindo variaveis unidas
dados_tratados$selectivity4 = NULL #excluindo variaveis unidas
dados_tratados$selectivityB = NULL #excluindo variaveis unidas
dados_tratados$selectivityG = NULL #excluindo variaveis unidas
dados_tratados$selectivityD1 = NULL #excluindo variaveis unidas
dados_tratados$selectivityD2 = NULL #excluindo variaveis unidas
dados_tratados$selectivityD3 = NULL #excluindo variaveis unidas

#normalizando as variáveis considerando os intervalos experimentais
dados_tratados$substrate = (dados_tratados$substrate - 0.00) / (0.45 - 0.00)
dados_tratados$reactant = (dados_tratados$reactant - 0.00) / (4.00 - 0.00)
dados_tratados$catalyst = (dados_tratados$catalyst - 0.00) / (20.00 - 0.00)
dados_tratados$temperature = (dados_tratados$temperature - 15.00) / (80.00 - 15.00)
dados_tratados$solvent = (dados_tratados$solvent - 1.00) / (6.00 - 1.00)
dados_tratados$time = (dados_tratados$substrate - 0.00) / (20.00 - 0.00)
dados_tratados$conversion = (dados_tratados$conversion - 0.00) / (100.00 - 0.00)
dados_tratados$selectivity_mono = (dados_tratados$selectivity_mono - 0.00) / (100.00 - 0.00)
dados_tratados$selectivity_di = (dados_tratados$selectivity_di - 0.00) / (100.00 - 0.00)

#dividindo o conjunto de dados em treinamento e teste
set.seed(123)
amostra = sample(1:nrow(dados_tratados), nrow(dados_tratados)*0.3)
test_data = dados_tratados[amostra, ]
test_data = as.matrix(test_data)
train_data = dados_tratados[-amostra, ]
train_data = as.matrix(train_data)

#CONSTRUÇÃO DA REDE NEURAL
#construindo a rede neural com os dados de treinamento
neural_net = neuralnet(conversion + selectivity_mono + selectivity_di ~ 
                         substrate + reactant + catalyst + temperature + solvent + time, 
                       data=train_data,
                       hidden=c(7,8,7,6),
                       act.fct = "logistic",
                       threshold = 0.01,
                       err.fct = "sse")

#plotando a rede neural
plot(neural_net)
plotnet(neural_net, 
        node_labs = TRUE,
        cex_val = 0.75,
        circle_col = "white",
        bord_col = "black",
        max_sp = TRUE,
        rel_rsc = 1,
        nid = FALSE,
        weights = TRUE)


#realizando previsões com os dados de teste
predictions = neuralnet::compute(neural_net, test_data[,1:6])$net.result
previsoes = as.numeric(predictions)
valores_reais = test_data[,7:9]
results = data.frame(predictions, valores_reais)

#comparando os resultados previstos com os resultados reais
#comparação da conversão prevista com a conversão real
ggplot(results, aes(x = results[, 4], y = results [,1])) +
  geom_point() + geom_smooth(method = "lm", se = FALSE) +
  xlab("Valores Reais") + ylab("Valores Previstos")

#comparação da selectivity_mono prevista com a selectivity_mono real
ggplot(results, aes(x = results[, 5], y = results [,2])) +
  geom_point() + geom_smooth(method = "lm", se = FALSE) +
  xlab("Valores Reais") + ylab("Valores Previstos")

#comparação da selectivity_di prevista com a selectivity_di real
ggplot(results, aes(x = results[, 6], y = results [,3])) +
  geom_point() + geom_smooth(method = "lm", se = FALSE) +
  xlab("Valores Reais") + ylab("Valores Previstos")

#avaliando o desempenho da rede neural

#calculando MAE
mae = mean(abs(previsoes - valores_reais))
print(mae)

# calculando RMSE
RMSE = sqrt(mean((previsoes - valores_reais)^2))
print(RMSE)

# calculando R²
SSE = sum((previsoes - valores_reais)^2)
SST = sum((valores_reais - mean(valores_reais))^2)
R2 = 1 - SSE/SST
print(R2)


#################
str(dados_tratados)
#################

