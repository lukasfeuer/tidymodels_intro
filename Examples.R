# -------------------------------------------------------------------------
#     Example Modeling Workflows
# -------------------------------------------------------------------------

# 01 --- Regression -------------------------------------------------------

library(tidymodels)

urchins <-
  readr::read_csv("https://tidymodels.org/start/models/urchins.csv") %>%
  setNames(c("food_regime", "initial_volume", "width")) %>%
  mutate(food_regime = factor(food_regime, levels = c("Initial", "Low", "High")))

# Model Specification and Fit
lm_fit <-
  linear_reg() |>
  set_engine("lm") |>
  fit(width ~ initial_volume * food_regime, data = urchins)

tidy(lm_fit, conf.int = TRUE)

# New Data for Prediction
new_points <- expand.grid(initial_volume = 20,
                          food_regime = c("Initial", "Low", "High")) |>
  arrange(food_regime)

mean_pred <- predict(lm_fit, new_data = new_points)
conf_int_pred <- predict(lm_fit,
                         new_data = new_points,
                         type = "conf_int")
res_data <-
  new_points %>%
  bind_cols(mean_pred) %>%
  bind_cols(conf_int_pred)

res_data


# 02 --- Logistic Regression ----------------------------------------------

library(tidymodels)
library(nycflights13)

# Data & Split
flight_data <- flights |>
  mutate(
    arr_delay = ifelse(arr_delay >= 30, "late", "on_time"),
    arr_delay = factor(arr_delay),
    date = lubridate::as_date(time_hour)
  ) |>
  inner_join(weather, by = c("origin", "time_hour")) |>
  select(dep_time, flight, origin, air_time, distance,
         date, arr_delay, time_hour, wind_speed, visib) |>
  na.omit() |>
  mutate_if(is.character, as.factor)

data_split <- initial_split(flight_data, prop = 3/4)
train_data <- training(data_split)
test_data <- testing(data_split)

# Create Recipe, Roles and Features
flights_rec <-
  recipe(arr_delay ~ ., data = train_data) |>
  update_role(flight, time_hour, new_role = "ID") |>
  step_date(date, features = c("dow", "month")) |>
  step_holiday(date,
               holidays = timeDate::listHolidays("US"),
               keep_original_cols = FALSE) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors())
# info: flights_rec$steps

# Model Specification
lr_mod <-
  logistic_reg() |>
  set_engine("glm")

# Workflow
flights_workflow <-
  workflow() |>
  add_model(lr_mod) |>
  add_recipe(flights_rec)

# Fit
flights_fit <-
  flights_workflow |>
  fit(data = train_data)

# Extract Results
flights_fit %>%
  extract_fit_parsnip() %>%
  tidy()

# Make Predictions for test data from fitted model
flights_aug <-
  augment(flights_fit, test_data)

# Evaluate Model
flights_aug %>%
  roc_curve(truth = arr_delay, .pred_late) %>%
  autoplot()

flights_aug %>%
  roc_auc(truth = arr_delay, .pred_late)


# 03 --- Random Forest ----------------------------------------------------

library(tidymodels)
library(modeldata)

# Data ans Split
data(cells, package = "modeldata")
set.seed(123)
cells_split <- initial_split(cells, strata = class)
cell_train <- training(cells_split)
cell_test <- testing(cells_split)

# Folds for Resampling
set.seed(345)
folds <- vfold_cv(cell_train, v = 10)

# Model Specification
rf_mod <-
  rand_forest(trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Recipe
rf_recipe <-
  recipe(class ~ ., data = cell_train) |>
  step_rm(case) |>
  step_zv()

# Generate a resampling workflow
rf_workflow <-
  workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(rf_recipe)#add_formula(class ~ .)

# Fit resampling Workflow
rf_fit_resample <-
  rf_workflow %>%
  fit_resamples(folds)

# Evaluate Resampling Fit
collect_metrics(rf_fit_resample)

# If confident with model, do final fit on whole training data
# Fit (no Recipe)
set.seed(234)
rf_fit <-
  rf_workflow |>
  fit(cell_train)

# Predict Test Data
rf_testing_pred <-
  predict(rf_fit, cell_test) %>%
  bind_cols(predict(rf_fit, cell_test, type = "prob")) %>%
  bind_cols(cell_test %>% select(class))

# Evaluate Final Performance
rf_testing_pred %>%
  roc_auc(truth = class, .pred_PS)
rf_testing_pred %>%
  accuracy(truth = class, .pred_class)


# 04 --- Decision Tree Tuning ---------------------------------------------

library(tidymodels)

# Data and data split
data(cells, package = "modeldata")
cell_split <- initial_split(cells |> select(-case),
                            strata = class)
cell_train <- training(cell_split)
cell_test <- testing(cell_split)

# parsnip Model Specification for hyperparameter tuning
tune_spec <-
  decision_tree(
    cost_complexity = tune(),
    tree_depth = tune()
  ) |>
  set_engine("rpart") |>
  set_mode("classification")

# grid of possible values to try for tuning
tree_grid <- grid_regular(cost_complexity(),
                          tree_depth(),
                          levels = 5) # 5 example values for each parameter

# Cross-Validation folds for tuning
cell_folds <- vfold_cv(cell_train)

# Create Tuning Workflow
tree_wf <-
  workflow() |>
  add_model(tune_spec) |>
  add_formula(class ~ .) # or use add_recipe() if a recipe is needed

# Run Tuning Workflow
tree_res <-
  tree_wf |>
  tune_grid(
    resamples = cell_folds,
    grid = tree_grid
  )

# Visualize Tuning results
tree_res |>
  collect_metrics() |>
  mutate(tree_depth = factor(tree_depth)) |>
  ggplot(aes(cost_complexity, mean, color = tree_depth)) +
  geom_line() +
  geom_point() +
  facet_wrap(~ .metric, scales = "free", nrow = 2) +
  scale_x_log10(labels = scales::label_number())

# Extract best Result from tuning
best_tree <- tree_res |>
  select_best("accuracy")

# Finalize Workflow
final_wf <-
  tree_wf |>
  finalize_workflow(best_tree)

# Fit final Workflow with final model
final_fit <-
  final_wf |>
  last_fit(cell_split)

final_fit |>
  collect_metrics()

# Final Workflow for prediction
final_tree <- extract_workflow(final_fit) # final_fit$.workflow[[1]]
final_tree

predict(final_tree, cells)

