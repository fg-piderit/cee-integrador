# Script para preparar el ambiente de trabajo
#
# {Descripción, etc.}

library(tidyverse)
library(tidymodels)
library(discrim)
library(ranger)
library(kknn)
library(kableExtra)
library(glmnet)

setup <- list(
  fecha            = Sys.Date(),
  seed             = 260221,
  datos_originales = "datos/datos_originales.csv",
  datos_train      = "datos/datos_train.csv",
  datos_test       = "datos/datos_test.csv"
)
