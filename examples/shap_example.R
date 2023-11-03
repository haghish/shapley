# load the required libraries for building the base-learners and the ensemble models
library(h2o)
library(h2otools)
library(autoEnsemble)

# initiate the h2o server
h2o.init(ignore_config = TRUE, nthreads = 2, bind_to_localhost = FALSE, insecure = TRUE)

# upload data to h2o cloud
prostate_path <- system.file("extdata", "prostate.csv", package = "h2o")
df <- read.csv(prostate_path)
prostate <- h2o.importFile(path = prostate_path, header = TRUE)

### H2O provides 2 types of grid search for tuning the models, which are
### AutoML and Grid. Below, I tune 2 set of model grids and use them both
### for building the ensemble, just to set an example ...

#######################################################
### PREPARE AutoML Grid (takes a couple of minutes)
#######################################################
# run AutoML to tune various models (GLM, GBM, XGBoost, DRF, DeepLearning) for 120 seconds
y <- "CAPSULE"
prostate[,y] <- as.factor(prostate[,y])  #convert to factor for classification
aml <- h2o.automl(y = y, training_frame = prostate, max_runtime_secs = 30,
                  include_algos=c("GBM"),

                  # this setting ensures the models are comparable for building a meta learner
                  seed = 2023, nfolds = 10,
                  keep_cross_validation_predictions = TRUE)



#######################################################
### PREPARE ENSEMBLE MODELS
#######################################################

ids <- h2o.get_ids(aml)
search <- ensemble(models = ids, training_frame = prostate, strategy = "search")

#######################################################
### EVALUATE THE MODELS
#######################################################
h2o.auc(aml@leader)                          # best model identified by h2o.automl
h2o.auc(h2o.getModel(grid@model_ids[[1]]))   # best model identified by grid search
h2o.auc(top$model)                           # ensemble model with 'top' strategy
h2o.auc(search$model)                        # ensemble model with 'search' strategy

p <- h2o.predict_contributions(object = aml@leader,
                                newdata = prostate,
                                top_n=5)
View(as.data.frame(p))

# Compute SHAP and pick the top two highest regardless of the sign
pabs <- h2o.predict_contributions(aml@leader, prostate, top_n=5, compare_abs=TRUE)
# Compute SHAP and pick the top two lowest regardless of the sign
View(as.data.frame(pabs))

h2o.shap_summary_plot(
  model = aml@leader,
  newdata = prostate,
  columns = NULL,
  top_n_features = 5)

# h2o.shap_summary_plot(
#   model = h2o.getModel("GBM_grid_1_AutoML_2_20230920_225003_model_7"),
#   newdata = prostate,
#   columns = NULL,
#   top_n_features = 5)

# plt <- h2o.shap_summary_plot(
#   model = aml@leader,
#   newdata = prostate,
#   columns = NULL #get SHAP for all columns
#   #top_n_features = 5
#   #sample_size = 100
# )

mdl1 <- h2o.shap_summary_plot(
  model = h2o.getModel(aml@leaderboard[2,"model_id"]),
  newdata = prostate,
  columns = NULL #get SHAP for all columns
  #top_n_features = 5
  #sample_size = 100
)


mdl2 <- h2o.shap_summary_plot(
  model = h2o.getModel(aml@leaderboard[2,"model_id"]),
  newdata = prostate,
  columns = NULL #get SHAP for all columns
  #top_n_features = 5
  #sample_size = 100
)

#get the data
dd <- mdl1$data
View(dd) #dd has 3040 rows, which is number of subjects * predictors of the model

#create the empty datasets for array
holder <- dd
holder <- holder[, c("Row.names", "contribution")]


d2 <- mdl2$data
View(d2) #dd has 3040 rows, which is number of subjects * predictors of the model


#id.x is the subject row number
#id.y I am not sure what this is
#feature is the predictor name
# 'normalized_value' is the original value of the observation, used for coloring
# 'original_value' is the original value of the observation, which serves the
#                  'normalized_value' later on
# THEREFORE 'original_valie' and 'normalized_value' are identical across models
#           for the same set of participants. The only thing that changes is the
#           'contribution' values
#

# create an array to store the shap values for each subject,
# and for each feature, presenting both 'contribution', 'original' values, and

#
# 1) sort shap data based on id.x
# 2) extract data of each subject and store it an array
# 3) the array should include:
#    - number of models > number of features > feature contributions for each subject
# 4) For the overall importance of each feature, we can report distrbution
#

FEATURES <- 7
COLUMNS <- 2 # 'Row.names' and 'contribution'
shaparray <- array(NA, dim=c(data.frame(nrow(df), c("Row.names", "contribution")), FEATURES))










# > both 'top' and 'search' strategies had identical results, but outperform the grid search and AutoML search. Yet, this was a small dataset, and a quick test...
