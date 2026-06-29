# Script para la exploración de los datos de entrenamiento
#
# {Descripción, etc.}

rm(list = ls())

source("R/setup.R")

set.seed(setup$seed)

# ============================================================================
datos_train <- read_csv(setup$datos_train)
datos_long <- pivot_longer(
  data          = datos_train,
  cols          = ends_with(c("_mean", "_se", "_worst")),
  names_to      = c("variable", "medida"),
  names_pattern = "^(.*)_(.*)$",
  values_to     = "valor"
)
variables <- unique(datos_long$variable)

# Cuadros --------------------------------------------------------------------
descriptivos <- datos_long |>
  summarise(
    perdidos = sum(is.na(valor)),
    n_unicos = length(unique(valor)),
    min      = min(valor),
    max      = max(valor),
    p_25     = quantile(valor, probs = 0.25),
    p_50     = quantile(valor, probs = 0.50),
    p_75     = quantile(valor, probs = 0.75),
    media = mean(valor),
    sd    = sd(valor),
    .by = c("variable", "medida")
  )

# Gráficos -------------------------------------------------------------------
distribuciones <- variables |>
  set_names() |>
  map(\(.var) {
    datos_long |> 
      filter(variable == .var) |> 
      mutate(n = n(), .by = c(medida)) |> 
      #ggplot(aes(x = valor, y = after_stat(3 * count / sum(count) / width), fill = target)) +
      ggplot(aes(x = valor, y = after_stat(density), color = target, fill = target)) +
      geom_histogram(bins = 30, position = "identity") +
      facet_wrap(vars(medida), nrow = 1, scales = "free") +
      #facet_grid(cols = vars(medida), rows = vars(target), scales = "free_x") +
      scale_fill_manual(values = c("M" = fill_alpha(2, 0.5), "B" = fill_alpha(4, 0.5))) +
      scale_color_manual(values = c("M" = 2, "B" = 4)) +
      labs(x = .var, y = "densidad") +
      theme_bw()
  })
conjuntas <- variables |> 
  set_names() |>
  map(\(.var) {
    datos <- select(datos_train, starts_with(.var), target) |>
      rename_with(\(.nom) str_replace(.nom, paste0(.var, "_"), ""), starts_with(.var)) 
    matriz_cor <- round(cor(datos |> select(mean, se, worst)), 4)
    svm <- ggplot(datos, aes(x = se, y = mean, color = target)) +
      geom_point(shape = 1) +
      scale_color_manual(values = c("M" = 2, "B" = 4)) +
      labs(subtitle = bquote(rho == .(matriz_cor[1, 2]))) +
      theme_bw() +
      theme(legend.position = "none")
    mvw <- ggplot(datos, aes(x = mean, y = worst, color = target)) +
      geom_point(shape = 1) +
      scale_color_manual(values = c("M" = 2, "B" = 4)) +
      labs(subtitle = bquote(rho == .(matriz_cor[1, 3]))) +
      theme_bw() +
      theme(legend.position = "none")
    wvs <- ggplot(datos, aes(x = worst, y = se, color = target)) +
      geom_point(shape = 1) +
      scale_color_manual(values = c("M" = 2, "B" = 4)) +
      labs(subtitle = bquote(rho == .(matriz_cor[2, 3]))) +
      theme_bw()
    mvw + svm + wvs
  })

# ============================================================================
descriptivos |>
  select(variable, medida, min, max, p_50, media, sd) |> 
  mutate(across(min:sd, \(.x) round(.x, 2))) |> 
  group_split(medida) |> 
  walk(\(.cuadro) {
    .medida <- .cuadro$medida[1]
    .cuadro |>
      select(-medida) |>
      kable(format = "latex", booktabs = TRUE) |>
      save_kable(file = paste0("informe/cuadros/descriptivos_", .medida, ".tex"))
  })
variables |> 
  walk(\(.v) {
    grafico <- distribuciones[[.v]] / conjuntas[[.v]]
    ggsave(
      plot = grafico,
      path = "informe/figuras/",
      filename = paste0("distribuciones_", .v, ".pdf"),
      width = 8.27,
      height = 5.83 * 0.95,
      units = "in"
    )
  })
