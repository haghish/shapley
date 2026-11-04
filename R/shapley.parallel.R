# Internal SHAPLEY function.
# Compute SHAP contributions for different models in parallel.
# ============================================================

shapley.parallel <- function(tasks, worker, shared_data, n_core = -1L,
                             h2o = FALSE, dependencies = character()) {

  # Limit the number of workers to the number of models.
  # ============================================================
  if (!length(tasks)) return(list())
  n_core <- min(shapley.cores(n_core), length(tasks))

  # For one worker, compute SHAP without creating a cluster.
  # ============================================================
  if (n_core == 1L) {
    return(lapply(tasks, worker, shared_data = shared_data))
  }

  # PSOCK workers work on Windows, Linux and macOS.
  # ============================================================
  cl <- parallel::makeCluster(n_core, type = "PSOCK")
  on.exit(parallel::stopCluster(cl), add = TRUE)

  # Match the package library paths of the main R session.
  # ============================================================
  parallel::clusterCall(cl, function(paths) {
    .libPaths(paths)
    NULL
  }, .libPaths())

  # H2O workers connect to the same running H2O server.
  # Each worker obtains the model and data from the server.
  # ============================================================
  if (h2o) {
    connection <- h2o::h2o.getConnection()
    h2o_ip <- connection@ip
    h2o_port <- connection@port
    parallel::clusterCall(cl, function(h2o_ip, h2o_port) {
      h2o::h2o.connect(ip = h2o_ip, port = h2o_port)
      NULL
    }, h2o_ip = h2o_ip, h2o_port = h2o_port)
  }

  # Export shared observations and the required internal functions once.
  # Rebinding these helpers to the worker global environment also allows
  # this implementation to work when the package is loaded with load_all().
  # ============================================================
  helper_names <- unique(dependencies)
  if (any(!nzchar(helper_names))) stop("Invalid parallel helper name")
  helper_env <- new.env(parent = emptyenv())
  helper_env$.shapley_shared_data <- shared_data

  worker_copy <- worker
  environment(worker_copy) <- globalenv()
  helper_env$.shapley_model_worker <- worker_copy

  for (name in helper_names) {
    fn <- get(name, envir = environment(shapley.parallel), inherits = TRUE)
    if (!is.function(fn)) stop("Parallel dependency is not a function: ", name)
    fn_copy <- fn
    environment(fn_copy) <- globalenv()
    assign(name, fn_copy, envir = helper_env)
  }
  parallel::clusterExport(cl, ls(helper_env, all.names = TRUE),
                          envir = helper_env)

  # Each task contains one model (or one H2O model ID).
  # Results retain the original order, even with load balancing.
  # ============================================================
  run.one.model <- function(task) {
    .shapley_model_worker(task, shared_data = .shapley_shared_data)
  }
  # Do not serialize the enclosing environment containing the cluster.
  environment(run.one.model) <- globalenv()
  parallel::parLapplyLB(cl, tasks, run.one.model)
}
