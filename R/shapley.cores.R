
shapley.cores <- function(n_core = -1L) {

  if (length(n_core) != 1L || !is.numeric(n_core) ||
      !is.finite(n_core) || n_core != as.integer(n_core) ||
      (n_core != -1L && n_core < 1L)) {
    stop("'n_core' must be -1 or a positive integer")
  }
  if (n_core != -1L) return(as.integer(n_core))

  # On Linux, use physical package and core IDs to avoid counting
  # hyperthreads as separate physical CPU cores.
  # ============================================================
  cores <- NA_integer_
  if (Sys.info()[["sysname"]] == "Linux") {
    cpu_dirs <- list.files("/sys/devices/system/cpu", pattern = "^cpu[0-9]+$",
                           full.names = TRUE)
    physical <- character(0)
    for (cpu in cpu_dirs) {
      package_file <- file.path(cpu, "topology", "physical_package_id")
      core_file <- file.path(cpu, "topology", "core_id")
      if (!file.exists(package_file) || !file.exists(core_file)) next
      package <- suppressWarnings(readLines(package_file, n = 1L, warn = FALSE))
      core <- suppressWarnings(readLines(core_file, n = 1L, warn = FALSE))
      if (length(package) == 1L && length(core) == 1L) {
        physical <- c(physical, paste(package, core, sep = ":"))
      }
    }
    if (length(physical) > 0L) cores <- length(unique(physical))
  }

  if (!is.finite(cores) || cores < 1L) {
    cores <- suppressWarnings(parallel::detectCores(logical = FALSE))
  }
  if (length(cores) != 1L || is.na(cores) || cores < 1L) {
    warning("Could not detect physical CPU cores; using one worker")
    cores <- 1L
  }
  return(as.integer(cores))
}
