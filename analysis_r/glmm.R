# Mixed-effects logistic regression on response-level correctness.
# Input: results/<experiment>/analysis/responses_long.csv (written by `prompteval analyze`).
#
#   Rscript analysis_r/glmm.R results/main/analysis/responses_long.csv C0
#
# Model: correct ~ condition * model * language + (1 | question_code)
# Runs of the same question share the random intercept. Reported:
#   - Type II Wald chi-square tests for each term (car::Anova)
#   - odds ratios of each condition vs the reference within model x language (emmeans),
#     Holm-adjusted within each model x language.
# Packages: lme4, emmeans, car  (install.packages(c("lme4", "emmeans", "car")))
# Not executed in this repository's automated tests (no R in CI); the Python
# GEE in prompteval.stats is the population-averaged counterpart.

suppressPackageStartupMessages({
  library(lme4)
  library(emmeans)
  library(car)
})

args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args) >= 1) args[1] else "results/main/analysis/responses_long.csv"
ref  <- if (length(args) >= 2) args[2] else "C0"

d <- read.csv(path, stringsAsFactors = FALSE)
d$condition <- relevel(factor(d$condition), ref = ref)
d$model_id <- factor(d$model_id)
d$language <- factor(d$language)
d$question_code <- factor(d$question_code)

terms <- "condition"
if (nlevels(d$model_id) > 1) terms <- paste(terms, "* model_id")
if (nlevels(d$language) > 1) terms <- paste(terms, "* language")
f <- as.formula(paste("correct ~", terms, "+ (1 | question_code)"))

fit <- glmer(f, data = d, family = binomial,
             control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
cat("Formula:", deparse(f), "\n\n")
print(summary(fit))
cat("\nType II Wald chi-square tests\n")
print(Anova(fit, type = 2))

by <- c()
if (nlevels(d$model_id) > 1) by <- c(by, "model_id")
if (nlevels(d$language) > 1) by <- c(by, "language")
spec <- if (length(by)) as.formula(paste("~ condition |", paste(by, collapse = " * "))) else ~ condition
em <- emmeans(fit, spec)
ct <- contrast(em, method = "trt.vs.ctrl", ref = 1, adjust = "holm")
out <- summary(ct, type = "response", infer = c(TRUE, TRUE))
cat("\nOdds ratios vs", ref, "(Holm within model x language)\n")
print(out)
write.csv(as.data.frame(out), sub("responses_long.csv$", "glmm_condition_effects.csv", path),
          row.names = FALSE)
