# Script para la evaluación de los modelos
#
# {Descripción, etc.}

rm(list = ls())

source("R/setup.R")

set.seed(setup$seed)

# Funciones ==================================================================
predecir <- function(.fit, .umbral, .datos) {
  .fit |>
    augment(new_data = .datos, type = "prob") |> 
    mutate(
      .pred = factor(if_else(.pred_M > .umbral, "M", "B"), levels = c("M" ,"B"))
    )
}
metricas_pred <- function(.datos) {
  bind_rows(
    .datos |> f_meas(truth = target, estimate = .pred, beta = 2),
    .datos |> accuracy(truth = target, estimate = .pred),
    .datos |> precision(truth = target, estimate = .pred),
    .datos |> recall(truth = target, estimate = .pred),
    .datos |> pr_auc(.pred_M, truth = target)
  )
}

# ============================================================================
load("salidas/modelos.RData")
datos_test <- read_csv("datos/datos_test.csv") |> 
  mutate(target = factor(target, levels = c("M", "B")))

# ----------------------------------------------------------------------------
modelos$metricas_test <- 1:6 |> 
  map(\(.i) {
    modelos$fit[[.i]] |>
      predecir(.umbral = modelos$umbral[[.i]], .datos = datos_test) |> 
      metricas_pred()
  })
modelos$matrices_conf <- 1:6 |> 
  set_names(modelos$modelo) |> 
  map(\(.i) {
    modelos$fit[[.i]] |> 
      predecir(.umbral = modelos$umbral[[.i]], .datos = datos_test) |> 
      conf_mat(truth = target, estimate = .pred)
  })

# Gráficos -------------------------------------------------------------------
metricas <- modelos |> 
  select(-fit) |> 
  unnest(cols = metricas_test)
orden <- metricas |> 
  filter(.metric == "f_meas") |> 
  arrange(.estimate) |> 
  pull(modelo)
grafico_metricas <- metricas|>
  mutate(modelo = factor(modelo, levels = orden)) |> 
  ggplot(aes(y = modelo, x = .estimate)) +
  geom_point() +
  facet_wrap(~ .metric) +
  labs(y = "Modelo", x = "Valor de la métrica") +
  theme_bw()

# Cuadros --------------------------------------------------------------------
cuadro_metricas <- metricas |> 
  pivot_wider(
    id_cols = modelo,
    names_from = .metric,
    values_from = .estimate
  ) |>
  arrange(desc(f_meas))

# ============================================================================
ggsave(
  plot = grafico_metricas,
  path = "informe/figuras/",
  filename = "metricas.pdf",
  width = 8.27 * 0.8,
  height = 5.83 * 0.8,
  units = "in"
)
cuadro_metricas |> 
  mutate(across(-modelo, \(.m) round(.m, 4))) |> 
  kbl(format = "latex", booktabs = TRUE) |> 
  save_kable(file = "informe/cuadros/metricas.tex")
