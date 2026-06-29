# Script para el ajuste de los modelos
#
# {Descripción, etc.}

rm(list = ls())

source("R/setup.R")

set.seed(setup$seed)

# Funciones ==================================================================
umbrales <- function(.preds, .umbrales, .beta = 1) {
  map(.umbrales, \(.t) {
    .preds |>
      mutate(
        pred = factor(
          ifelse(.pred_M > .t, "M", "B"),
          levels = c("M", "B")
        )
      ) |>
      f_meas(target, pred, .beta) |>
      mutate(.threshold = .t)
  }) |> list_rbind()
}

# Variables ==================================================================
datos_train <- read_csv(setup$datos_train) |> 
  mutate(target = factor(target, levels = c("M", "B")))
folds <- vfold_cv(datos_train, v = 5, strata = target)
metricas <- metric_set(pr_auc, precision, recall)
grilla_umbral <- seq(0.01, 0.99, 0.01)
base_recipe <-
  recipe(target ~ ., data = datos_train) |> 
  step_normalize(all_predictors())

# Logística con PCA ==========================================================
# Hiperparámetros:
# - Número de componentes (num_comp)
grilla_comp <- tibble(num_comp = 1:30)

# Procedimiento --------------------------
logit_recipe <-
  base_recipe |> 
  step_pca(all_predictors(), num_comp = tune())
logit_model <-
  logistic_reg() |> 
  set_mode("classification") |> 
  set_engine("glm")
logit_wf <-
  workflow() |> 
  add_model(logit_model) |> 
  add_recipe(logit_recipe)

# Calibración ----------------------------
logit_tune <- logit_wf |> 
  tune_grid(
    resamples = folds,
    grid      = grilla_comp,
    metrics   = metricas,
    control   = control_grid(save_pred = TRUE)
  )
logit_best <- logit_tune |> 
  select_best(metric = "pr_auc")
logit_met <- logit_tune |> 
  collect_metrics()
logit_pred <- logit_tune |> 
  collect_predictions() |> 
  filter(num_comp == logit_best$num_comp)

# Umbral
logit_umbrales <- umbrales(logit_pred, grilla_umbral, .beta = 2)
logit_umbral <- logit_umbrales |> 
  filter(.estimate == max(.estimate)) |> 
  pull(.threshold) |> 
  mean()

# Ajuste ---------------------------------
logit_fit <- logit_wf |> 
  finalize_workflow(logit_best) |> 
  fit(datos_train)

# Logística con LASSO (vars. originales) =====================================
# Hiperparámetros:
# - Penalización o lambda (penalty)
grilla_lambda <- tibble(penalty = 10^seq(-6, -1, length.out = 25))

# Procedimiento --------------------------
lasso1_model <- 
  logistic_reg(penalty = tune(), mixture = 1) |> 
  set_mode("classification") |> 
  set_engine("glmnet")
lasso1_wf <- 
  workflow() |> 
  add_recipe(base_recipe) |> 
  add_model(lasso1_model)

# Calibración ----------------------------
lasso1_tune <- lasso1_wf |> 
  tune_grid(
    resamples = folds,
    grid      = grilla_lambda,
    metrics   = metricas,
    control   = control_grid(save_pred = TRUE)
  )
lasso1_best <- lasso1_tune |> 
  select_best(metric = "pr_auc")
lasso1_met <- lasso1_tune |> 
  collect_metrics()
lasso1_pred <- lasso1_tune |> 
  collect_predictions() |> 
  filter(penalty == lasso1_best$penalty)

# Umbral
lasso1_umbrales <- umbrales(lasso1_pred, grilla_umbral, .beta = 2)
lasso1_umbral <- lasso1_umbrales |> 
  filter(.estimate == max(.estimate)) |> 
  pull(.threshold) |> 
  mean()

# Ajuste ---------------------------------
lasso1_fit <- lasso1_wf |> 
  finalize_workflow(lasso1_best) |> 
  fit(datos_train)

# Logística con LASSO (PCA) ==================================================
# Hiperparámetros:
# - Penalización o lambda (penalty)
# Procedimiento --------------------------
lasso2_recipe <-
  base_recipe |> 
  step_pca(all_predictors(), num_comp = 30)
lasso2_model <- 
  logistic_reg(penalty = tune(), mixture = 1) |> 
  set_mode("classification") |> 
  set_engine("glmnet")
lasso2_wf <- 
  workflow() |> 
  add_recipe(lasso2_recipe) |> 
  add_model(lasso2_model)

# Calibración ----------------------------
lasso2_tune <- lasso2_wf |> 
  tune_grid(
    resamples = folds,
    grid      = grilla_lambda,
    metrics   = metricas,
    control   = control_grid(save_pred = TRUE)
  )
lasso2_best <- lasso2_tune |> 
  select_best(metric = "pr_auc")
lasso2_met <- lasso2_tune |> 
  collect_metrics()
lasso2_pred <- lasso2_tune |> 
  collect_predictions() |> 
  filter(penalty == lasso2_best$penalty)

# Umbral
lasso2_umbrales <- umbrales(lasso2_pred, grilla_umbral, .beta = 2)
lasso2_umbral <- lasso2_umbrales |> 
  filter(.estimate == max(.estimate)) |> 
  pull(.threshold) |> 
  mean()

# Ajuste ---------------------------------
lasso2_fit <- lasso2_wf |> 
  finalize_workflow(lasso2_best) |> 
  fit(datos_train)

# Random Forests =============================================================
# Hiperparámetros:
# - Número de árboles (trees)
# - Variables seleccionadas (mtry)
# - Mínimo de obs. por hoja (min_n)
grilla_rf <- expand_grid(mtry = 1:10, min_n = c(10, 20, 30))

# Procedimiento --------------------------
rf_recipe <-
  recipe(target ~ ., data = datos_train) |> 
  step_rm(fractal_dimension_mean, fractal_dimension_se, fractal_dimension_worst,
          symmetry_se, smoothness_se, texture_se) |> 
  step_normalize(all_predictors())
rf_model <-
  rand_forest(trees = 100, min_n = tune(), mtry = tune()) |> 
  set_mode("classification") |> 
  set_engine("ranger", seed = 260224)
rf_wf <-
  workflow() |> 
  add_recipe(rf_recipe) |> 
  add_model(rf_model)

# Calibración ----------------------------
rf_tune <- rf_wf |> 
  tune_grid(
    resamples = folds,
    grid      = grilla_rf,
    metrics   = metricas,
    control   = control_grid(save_pred = TRUE)
  )
rf_best <- rf_tune |> 
  select_best(metric = "pr_auc")
rf_met <- rf_tune |> 
  collect_metrics()
rf_pred <- rf_tune |> 
  collect_predictions() |> 
  filter(mtry == rf_best$mtry)

# Umbral
rf_umbrales <- umbrales(rf_pred, grilla_umbral, .beta = 2)
rf_umbral <- rf_umbrales |> 
  filter(.estimate == max(.estimate)) |> 
  pull(.threshold) |> 
  mean()

# Ajuste ---------------------------------
rf_fit <- rf_wf |> 
  finalize_workflow(rf_best) |> 
  fit(datos_train)


# LDA ========================================================================
# No hay hiperparámetros, a menos que haga algún tipo de regularización
# Procedimiento --------------------------
lda_recipe <-
  base_recipe |> 
  step_pca(all_predictors(), num_comp = tune())
lda_model <-
  discrim_linear() |> 
  set_engine("MASS")
lda_wf <-
  workflow() |> 
  add_recipe(lda_recipe) |> 
  add_model(lda_model)

# Evaluación -----------------------------
lda_tune <- lda_wf |> 
  tune_grid(
    resamples = folds,
    metrics   = metricas,
    grid      = grilla_comp,
    control   = control_resamples(save_pred = TRUE)
  )
lda_best <- lda_tune |> select_best(metric = "pr_auc")
lda_met <- lda_tune |> collect_metrics()
lda_pred <- lda_tune |>
  collect_predictions() |>
  filter(num_comp == lda_best$num_comp)

# Umbral
lda_umbrales <- umbrales(lda_pred, grilla_umbral, .beta = 2)
lda_umbral <- lda_umbrales |> 
  filter(.estimate == max(.estimate)) |> 
  pull(.threshold) |> 
  mean()

# Ajuste ---------------------------------
lda_fit <- lda_wf |> 
  finalize_workflow(lda_best) |> 
  fit(datos_train)

# KNN ========================================================================
# Hiperparámetros:
# - Número de vecinos (k)
grilla_nb <- tibble(neighbors = seq(5, 100, 5))

# Procedimiento --------------------------
knn_model <-
  nearest_neighbor(neighbors = tune()) |> 
  set_mode("classification")
knn_wf <-
  workflow() |> 
  add_recipe(base_recipe) |> 
  add_model(knn_model)

# Calibración ----------------------------
knn_tune <- knn_wf |> 
  tune_grid(
    resamples = folds,
    grid      = grilla_nb,
    metrics   = metricas,
    control   = control_grid(save_pred = TRUE)
  )
knn_best <- knn_tune |> 
  select_best(metric = "pr_auc")
knn_met <- knn_tune |> 
  collect_metrics()
knn_pred <- knn_tune |> 
  collect_predictions() |> 
  filter(neighbors == knn_best$neighbors)

knn_umbrales <- umbrales(knn_pred, grilla_umbral, .beta = 2)
knn_umbral <- knn_umbrales |> 
  filter(.estimate == max(.estimate)) |> 
  pull(.threshold) |> 
  mean()

# Ajuste ---------------------------------
knn_fit <- knn_wf |> 
  finalize_workflow(knn_best) |> 
  fit(datos_train)

# ----------------------------------------------------------------------------
# Meter todos los modelos en un tibble
modelos <- tibble(
  modelo = c("logit", "lasso_original", "lasso_pca", "rf" ,"lda", "knn"),
  umbral = c(logit_umbral, lasso1_umbral, lasso2_umbral, rf_umbral, lda_umbral, knn_umbral),
  fit    = list(logit_fit, lasso1_fit, lasso2_fit, rf_fit, lda_fit, knn_fit)
)
save(modelos, file = "salidas/modelos.RData")

# Gráficos -------------------------------
grafico_rf_met <- rf_met |> 
  ggplot(aes(x = mtry, y = mean, color = .metric)) +
  geom_point() +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err), width = 0.2) +
  facet_wrap(~ min_n) +
  labs(x = "Número de variables", y = "Métrica promedio", color = "Métrica") +
  theme_bw()
grafico_rf_preds <- rf_pred |> 
  ggplot(aes(x = .pred_M, color = id, linetype = target)) +
  stat_ecdf(geom = "step") +
  geom_vline(xintercept = rf_umbral, linetype = "dashed") +
  labs(x = "Probabilidad predicha", y = "Proporción acumulada", color = "Fold") +
  theme_bw() +
  theme(legend.position = "left")
grafico_rf_umbrales <- rf_pred |> 
  group_by(id) |> 
  umbrales(grilla_umbral, .beta = 2) |> 
  ggplot(aes(x = .threshold, y = .estimate, color = id)) +
  geom_vline(xintercept = rf_umbral, linetype = "dashed") +
  geom_step() +
  labs(x = "Umbral", y = expression(F[2]), color = "Fold") +
  theme_bw() +
  theme(legend.position = "none")
grafico_rf_pu <- grafico_rf_preds + grafico_rf_umbrales + plot_annotation(tag_levels = "A")
  
grafico_lasso1_met <- lasso1_met |> 
  ggplot(aes(x = penalty, y = mean, color = .metric)) +
  geom_vline(xintercept = lasso1_best$penalty, linetype = "dashed", alpha = 0.5) +
  geom_point() +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err), width = 0.05) +
  labs(x = expression(log[10](lambda)), y = "Métrica promedio", color = "Métrica") +
  scale_x_log10() +
  theme_bw()
grafico_lasso1_preds <- lasso1_pred |> 
  ggplot(aes(x = .pred_M, color = id, linetype = target)) +
  stat_ecdf(geom = "step") +
  geom_vline(xintercept = lasso1_umbral, linetype = "dashed") +
  labs(x = "Probabilidad predicha", y = "Proporción acumulada", color = "Fold") +
  theme_bw() +
  theme(legend.position = "left")
grafico_lasso1_umbrales <- lasso1_pred |> 
  group_by(id) |> 
  umbrales(grilla_umbral, .beta = 2) |> 
  ggplot(aes(x = .threshold, y = .estimate, color = id)) +
  geom_vline(xintercept = lasso1_umbral, linetype = "dashed") +
  geom_step() +
  labs(x = "Umbral", y = expression(F[2]), color = "Fold") +
  theme_bw() +
  theme(legend.position = "none")
grafico_lasso1_pu <- grafico_lasso1_preds + grafico_lasso1_umbrales + plot_annotation(tag_levels = "A")

ggsave(
  grafico_rf_pu,
  path = "informe/figuras/",
  file = "rf_pred_umbral.pdf",
  width = 8.27 * 0.8,
  height = 5.83 * 0.6,
  unit = "in"
)
ggsave(
  grafico_rf_met,
  path = "informe/figuras/",
  file = "rf_metricas.pdf",
  width = 8.27 * 0.8,
  height = 5.83 * 0.6,
  unit = "in"
)
ggsave(
  grafico_lasso1_pu,
  path = "informe/figuras/",
  file = "lasso1_pred_umbral.pdf",
  width = 8.27 * 0.8,
  height = 5.83 * 0.6,
  unit = "in"
)
ggsave(
  grafico_lasso1_met,
  path = "informe/figuras/",
  file = "lasso1_metricas.pdf",
  width = 8.27 * 0.8,
  height = 5.83 * 0.6,
  unit = "in"
)

lasso1_coefs <- lasso1_fit |> 
  extract_fit_parsnip() |>
  tidy() |> 
  mutate(estimate = -estimate, exp_estimate = exp(estimate)) |> 
  filter(estimate != 0) |> 
  select(-penalty)
lasso1_coefs |> 
  mutate(estimate = round(estimate, 4), exp_estimate = round(exp_estimate, 4)) |> 
  kbl(format = "latex", booktabs = TRUE) |> 
  save_kable("informe/cuadros/lasso1_coefs.tex")
