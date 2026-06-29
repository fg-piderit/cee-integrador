# Script para separar el conjunto de datos en dos subconjuntos, uno de
# entrenamiento y otro de prueba
#
# {Descripción detallada, etc.}

rm(list = ls())

source("R/setup.R")

set.seed(setup$seed)

# ============================================================================
datos <- read_csv(setup$datos_originales)
datos_split <- initial_split(datos, prop = 0.7)

write_csv(training(datos_split), file = setup$datos_train)
write_csv(testing(datos_split), file = setup$datos_test)