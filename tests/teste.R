# 1. Configuração Inicial -----------------------------------------------------
# Instalar pacotes se necessário
if (!require("tidyverse")) install.packages("tidyverse")

# Carregar bibliotecas
library(tidyverse)

# Importar os dados diretamente do link fornecido
url <- "https://gist.githubusercontent.com/slopp/ce3b90b9168f2f921784de84fa445651/raw/4ecf3041f0ed4913e7c230758733948bc561f434/penguins.csv"
penguins <- read_csv(url)

# Visualizar as primeiras linhas dos dados
head(penguins)

# 2. Estrutura dos Dados ------------------------------------------------------
# Verificar a estrutura dos dados
glimpse(penguins)

# Resumo das variáveis
summary(penguins)

# Identificar valores ausentes
colSums(is.na(penguins))

# 3. Análise Descritiva -------------------------------------------------------
# Estatísticas descritivas para variáveis numéricas
penguins %>%
  select(where(is.numeric)) %>%
  summary()

# Histogramas para variáveis numéricas
penguins %>%
  pivot_longer(cols = where(is.numeric), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(binwidth = 5, fill = "blue", color = "black") +
  facet_wrap(~ variable, scales = "free") +
  labs(title = "Distribuição das Variáveis Numéricas")

# Frequências para variáveis categóricas
penguins %>%
  select(where(is.character)) %>%
  summary()

# Gráficos de barras para variáveis categóricas
penguins %>%
  pivot_longer(cols = where(is.character), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value, fill = value)) +
  geom_bar() +
  facet_wrap(~ variable, scales = "free") +
  labs(title = "Frequência das Variáveis Categóricas")

# 4. Análise Exploratória -----------------------------------------------------
# Matriz de correlação e mapa de calor
cor_matrix <- penguins %>%
  select(where(is.numeric)) %>%
  cor(use = "complete.obs")

ggcorrplot(cor_matrix, type = "upper", lab = TRUE) +
  labs(title = "Matriz de Correlação entre Variáveis Numéricas")

# Comparar distribuições por espécie
penguins %>%
  ggplot(aes(x = species, y = body_mass_g, fill = species)) +
  geom_boxplot() +
  labs(title = "Distribuição de Massa Corporal por Espécie")

# Comparar distribuições por ilha
penguins %>%
  ggplot(aes(x = island, y = flipper_length_mm, fill = island)) +
  geom_violin() +
  labs(title = "Distribuição do Comprimento da Nadadeira por Ilha")

# Gráfico de dispersão entre comprimento do bico e profundidade do bico
penguins %>%
  ggplot(aes(x = bill_length_mm, y = bill_depth_mm, color = species)) +
  geom_point() +
  labs(title = "Relação entre Comprimento e Profundidade do Bico",
       x = "Comprimento do Bico (mm)",
       y = "Profundidade do Bico (mm)")

# 5. Tratamento de Valores Ausentes --------------------------------------------
# Contagem de valores ausentes por coluna
colSums(is.na(penguins))

# Remover linhas com valores ausentes
penguins_clean <- penguins %>%
  drop_na()

# Verificar novamente
colSums(is.na(penguins_clean))

# 6. Exportação dos Resultados -------------------------------------------------
# Exportar dados limpos para CSV
write_csv(penguins_clean, "penguins_clean.csv")

# Salvar gráficos como arquivos
ggsave("correlation_heatmap.png", width = 10, height = 8)
