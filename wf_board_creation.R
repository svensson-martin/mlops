library(tidyverse)
library(tidymodels)
library(palmerpenguins)
library(gt)
library(ranger)
library(brulee)
library(pins)
library(vetiver)
library(plumber)
library(conflicted)
tidymodels_prefer()
conflict_prefer("penguins", "palmerpenguins")


penguins %>% 
  filter(!is.na(sex)) %>% 
  ggplot(aes(x = flipper_length_mm, 
             y = bill_length_mm, 
             color = sex, 
             size = body_mass_g)) + 
  geom_point(alpha = 0.5) + 
  facet_wrap(~ species) + 
  theme_classic()

penguins_df <- 
  penguins %>% 
  drop_na(sex) %>% 
  select(-year, -island)

set.seed(1234)

penguin_split <- 
  penguins_df %>% 
  initial_split(strata = sex)

train <- 
  penguin_split %>% 
  training()
test <- 
  penguin_split %>% 
  testing()

folds <- 
  train %>% 
  vfold_cv()

rec <- 
  recipe(sex ~ ., data = train) %>% 
  step_YeoJohnson(all_numeric_predictors()) %>% 
  step_dummy(species) %>% 
  step_normalize(all_numeric_predictors())

glm_spec <- 
  logistic_reg(penalty = 1) %>% 
  set_engine("glm")

tree_spec <- 
  rand_forest(min_n = tune()) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

mlp_spec <- 
  mlp(
    hidden_units = tune(), 
    epochs = tune(), 
    penalty = tune(), 
    learn_rate = tune()
  ) %>% 
  set_engine("brulee") %>% 
  set_mode("classification")

bayes_control <- control_bayes(no_improve = 10L, 
                               time_limit = 20, 
                               save_pred = TRUE, 
                               verbose = TRUE)

wf_set <- 
  workflow_set(
    preproc = list(rec), 
    models = list(glm = glm_spec, 
                  tree = tree_spec, 
                  torch = mlp_spec)
  ) %>% 
  workflow_map("tune_bayes", 
               iter = 50L, 
               resamples = folds, 
               control = bayes_control)

wf_set %>% 
  rank_results(
    rank_metric = "roc_auc", 
    select_best = FALSE
  ) %>% View()

wf_set %>% 
  autoplot()

wf_set %>% 
  unnest(result) %>% 
  unnest(.metrics) %>% 
  select(wflow_id, id, .metric, .estimate) %>% 
  group_by(.metric) %>% 
  arrange(desc(.estimate))

best_id <- "recipe_glm"

best_fit <- 
  wf_set %>% 
  extract_workflow_set_result(best_id) %>% 
  select_best(metric = "accuracy") 

final_wf <- 
  wf_set %>% 
  extract_workflow(best_id) %>% 
  finalize_workflow(best_fit)

final_fit <- 
  final_wf %>% 
  last_fit(penguin_split)

final_fit %>% 
  collect_metrics() 

final_fit %>% 
  collect_predictions() %>% 
  roc_curve(sex, .pred_female) %>% 
  autoplot()

final_fit_to_deploy <- 
  final_fit %>% 
  extract_workflow()

v <- 
  final_fit_to_deploy %>% 
  vetiver_model(model_name = "penguins_model")

model_board <- board_folder(path = "pins_r", versioned = TRUE)
model_board %>% 
  vetiver_pin_write(v)

model_board %>% 
  vetiver_write_plumber("penguins_model")
model_board %>% 
  write_board_manifest()
