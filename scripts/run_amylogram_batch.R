#!/usr/bin/env Rscript
# Batch AmyloGram scoring for Fig. 3 sequence sets.
# Output: results/sequence/amylogram/<dataset>/
#
# Usage:
#   Rscript scripts/run_amylogram_batch.R [/path/to/amyAMP]

suppressPackageStartupMessages({
  library(AmyloGram)
  library(seqinr)
})

args <- commandArgs(trailingOnly = TRUE)
path_root <- if (length(args) >= 1) {
  normalizePath(args[[1]], mustWork = TRUE)
} else {
  cand <- normalizePath(getwd())
  if (dir.exists(file.path(cand, "results", "sequence"))) {
    cand
  } else if (dir.exists(file.path(cand, "..", "results", "sequence"))) {
    normalizePath(file.path(cand, ".."))
  } else {
    "/home/anupkumar/workspace/amyAMP"
  }
}

path_seq <- file.path(path_root, "results", "sequence")
path_out <- file.path(path_seq, "amylogram")
dir.create(path_out, recursive = TRUE, showWarnings = FALSE)

# dataset_label -> FASTA basename
datasets <- list(
  amyAMP   = "seqs_generated_postprocessed.fasta",
  trainPep = "trainPep.fasta",
  `AMP(D)` = "dbaasp_non_APR.fasta",
  `AMY(D)` = "AMY_nonAntibacterial.fasta",
  randPep  = "random_peptides_1000.fasta"
)

# Safe subdirectory / filename stem (AMP(D) -> AMPD)
subdir_name <- function(label) {
  gsub("[()]", "", label)
}

data(AmyloGram_model)

is_standard_aa <- function(seq_vec) {
  aa <- c("A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y")
  chars <- toupper(as.character(seq_vec))
  length(chars) > 0 && all(chars %in% aa)
}

# AmyloGram uses hexamer n-grams; length < 6 raises "n-gram too long"
MIN_LEN <- 6L

score_one_chunk <- function(seqs_chunk) {
  tryCatch(
    {
      pred <- predict(AmyloGram_model, seqs_chunk)
      if (is.data.frame(pred)) {
        pred
      } else if (!is.null(pred$overview)) {
        pred$overview
      } else {
        stop("Unexpected predict() return type")
      }
    },
    error = function(e) {
      # Fall back to per-sequence if a chunk fails (e.g. edge lengths)
      message(sprintf("    Chunk failed (%s); scoring sequence-wise...", conditionMessage(e)))
      rows <- list()
      for (j in seq_along(seqs_chunk)) {
        rows[[j]] <- tryCatch(
          {
            p <- predict(AmyloGram_model, seqs_chunk[j])
            if (is.data.frame(p)) p else p$overview
          },
          error = function(e2) {
            data.frame(
              Name = names(seqs_chunk)[j],
              Probability = NA_real_,
              stringsAsFactors = FALSE
            )
          }
        )
      }
      do.call(rbind, rows)
    }
  )
}

score_fasta <- function(fasta_path, label, out_dir) {
  message(sprintf("\n=== %s ===\n  FASTA: %s", label, fasta_path))
  if (!file.exists(fasta_path)) {
    stop("Missing FASTA: ", fasta_path)
  }

  seqs <- read.fasta(fasta_path, seqtype = "AA", as.string = FALSE, set.attributes = TRUE)
  n_raw <- length(seqs)
  std_ok <- vapply(seqs, is_standard_aa, logical(1))
  len_ok <- lengths(seqs) >= MIN_LEN
  keep <- std_ok & len_ok
  seqs_ok <- seqs[keep]
  n_skip_nonstd <- sum(!std_ok)
  n_skip_short <- sum(std_ok & !len_ok)
  if (n_skip_nonstd > 0) {
    message(sprintf("  Skipped %d sequences with non-standard residues", n_skip_nonstd))
  }
  if (n_skip_short > 0) {
    message(sprintf("  Skipped %d sequences with length < %d (AmyloGram hexamer)", n_skip_short, MIN_LEN))
  }
  message(sprintf("  Scoring %d / %d sequences...", length(seqs_ok), n_raw))

  if (length(seqs_ok) == 0) {
    stop("No valid sequences for ", label)
  }

  # Chunk to limit memory (n-gram expansion)
  chunk_size <- 500L
  idx <- seq_along(seqs_ok)
  chunks <- split(idx, ceiling(idx / chunk_size))

  # AmyloGram::predict returns a data.frame with columns Name, Probability
  overview_list <- list()
  for (ci in seq_along(chunks)) {
    ids <- chunks[[ci]]
    message(sprintf("  Chunk %d/%d (n=%d)", ci, length(chunks), length(ids)))
    overview_list[[ci]] <- score_one_chunk(seqs_ok[ids])
  }

  overview <- do.call(rbind, overview_list)

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  stem <- subdir_name(label)
  overview_path <- file.path(out_dir, paste0(stem, "_AmyloGram_scores.csv"))
  write.csv(overview, overview_path, row.names = FALSE)

  prob_col <- if ("Probability" %in% names(overview)) {
    overview$Probability
  } else {
    overview[[grep("(?i)prob", names(overview), perl = TRUE)[1]]]
  }

  meta <- data.frame(
    dataset = label,
    fasta = basename(fasta_path),
    n_fasta = n_raw,
    n_scored = sum(!is.na(prob_col)),
    n_skipped_nonstandard = n_skip_nonstd,
    n_skipped_short = n_skip_short,
    n_failed = sum(is.na(prob_col)),
    mean_probability = mean(prob_col, na.rm = TRUE),
    median_probability = median(prob_col, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
  write.csv(meta, file.path(out_dir, "summary.csv"), row.names = FALSE)
  message(sprintf(
    "  Wrote %s (n=%d, mean P=%.3f)",
    overview_path, meta$n_scored, meta$mean_probability
  ))
  meta
}

all_meta <- list()
for (label in names(datasets)) {
  fasta <- file.path(path_seq, datasets[[label]])
  out_dir <- file.path(path_out, subdir_name(label))
  all_meta[[label]] <- score_fasta(fasta, label, out_dir)
}

summary_all <- do.call(rbind, all_meta)
write.csv(summary_all, file.path(path_out, "summary_all_datasets.csv"), row.names = FALSE)
message("\nDone. Summary: ", file.path(path_out, "summary_all_datasets.csv"))
print(summary_all)
