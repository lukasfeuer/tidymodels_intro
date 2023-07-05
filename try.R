# -------------------------------------------------------------------------
#     Try
# -------------------------------------------------------------------------
#


library(nycflights13)

# Format data
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

# Split data
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

# Build Model Specification with parsnip
lr_mod <-
  logistic_reg() |>
  set_engine("glm")

# Create a workflow
flights_workflow <-
  workflow() |>
  add_model(lr_mod) |>
  add_recipe(flights_rec)

# Fit model with workflow
flights_fit <-
  flights_workflow |>
  fit(data = train_data)

# Extract Results
flights_fit %>%
  extract_fit_parsnip() %>%
  tidy()


flights_aug <-
  augment(flights_fit, test_data)

# The data look like:
flights_aug %>%
  select(arr_delay, time_hour, flight, .pred_class, .pred_on_time)

flights_aug %>%
  roc_curve(truth = arr_delay, .pred_late) %>%
  autoplot()

flights_aug %>%
  roc_auc(truth = arr_delay, .pred_late)
