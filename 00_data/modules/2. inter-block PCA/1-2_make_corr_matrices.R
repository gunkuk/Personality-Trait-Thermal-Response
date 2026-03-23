# 요약: Excel에서 BI/SUR/EUP/TLX 시트를 읽고, 블록별로 Pearson / polychoric / mixed correlation을 계산해 CSV로 저장한다.

library(readxl)
library(psych)

# =========================
# 1) 경로 설정
# =========================
input_xlsx <- "C:/Users/rjs11/Desktop/1/PAPER/00_SCI/MBTI/model/1. data cleaning/domain knowledge/rawdata_wrong corrected_0319.xlsx"
out_dir <- "C:/Users/rjs11/Desktop/1/PAPER/00_SCI/MBTI/model/OUT/corr_matrices_from_R"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# =========================
# 2) 유틸
# =========================
upper_cols <- function(df) {
  names(df) <- ifelse(is.na(names(df)), paste0("unnamed_", seq_along(names(df))), trimws(names(df)))
  names(df)[tolower(names(df)) == "sex"] <- "SEX"
  names(df)[tolower(names(df)) == "age"] <- "AGE"
  names(df)[tolower(names(df)) == "bmi"] <- "BMI"
  names(df)[names(df) == "ID"] <- "ID"
  names(df)[tolower(names(df)) == "time (min)"] <- "time_min"
  names(df)[tolower(names(df)) == "env"] <- "env"
  df
}

to_num_df <- function(df, exclude = c("ID", "no.", "name", "env", "time_min", "scenerio")) {
  for (nm in names(df)) {
    if (!(nm %in% exclude)) {
      df[[nm]] <- suppressWarnings(as.numeric(df[[nm]]))
    }
  }
  df
}

drop_constant_cols <- function(df) {
  keep <- sapply(df, function(v) length(unique(v[!is.na(v)])) > 1)
  df[, keep, drop = FALSE]
}

# =========================
# 3) 데이터 로드
# =========================
BI  <- upper_cols(read_excel(input_xlsx, sheet = "BI",  skip = 1))
SUR <- upper_cols(read_excel(input_xlsx, sheet = "SUR", skip = 1))
EUP <- upper_cols(read_excel(input_xlsx, sheet = "EUP", skip = 1))
TLX <- upper_cols(read_excel(input_xlsx, sheet = "TLX", skip = 1))

BI  <- to_num_df(BI)
SUR <- to_num_df(SUR)
EUP <- to_num_df(EUP)
TLX <- to_num_df(TLX)

if ("SEX" %in% names(BI)) {
  BI$SEX <- ifelse(toupper(trimws(as.character(BI$SEX))) %in% c("M", "MALE"), 1,
                   ifelse(toupper(trimws(as.character(BI$SEX))) %in% c("F", "FEMALE"), 0, NA))
}

# =========================
# 4) SUR aggregation
# =========================
sur_num_cols <- setdiff(names(SUR), c("ID", "name", "env", "time_min"))
SUR_AGG <- aggregate(SUR[, sur_num_cols], by = SUR[c("ID", "name", "env")], FUN = function(x) mean(x, na.rm = TRUE))

for (i in 1:7) {
  p <- paste0("P", i)
  m <- paste0("M", i)
  if (p %in% names(SUR_AGG) && m %in% names(SUR_AGG)) {
    SUR_AGG[[paste0("P", i, "-M", i)]] <- SUR_AGG[[p]] - SUR_AGG[[m]]
  }
}

# =========================
# 5) 블록 정의
# =========================
BLOCKS <- list(
  SUR_TSV_TCV_TA_TP = c("TSV", "TCV", "TA", "TP"),
  SUR_P_M = c(paste0("P", 1:7), paste0("M", 1:7)),
  SUR_P_minus_M = paste0("P", 1:7, "-M", 1:7),
  EUP_ONLY = paste0("EUP", 1:18),
  EUP_SP_ONLY = c("SP1", "SP2", "SP3", "SP4"),
  BI_SEX_BMI_AGE = c("SEX", "BMI", "AGE"),
  BI_SEX_BMI = c("SEX", "BMI"),
  TLX_S = c("TLX1", "TLX2", "TLX3", "TLX4", "TLX5", "TLX6", "S1", "S2"),
  BI_MBTI = c("E", "N", "T", "J", "A"),
  BI_FFM = c("o1", "c1", "e1", "a1", "n1"),
  BI_MBTI_FFM = c("E", "N", "T", "J", "A", "o1", "c1", "e1", "a1", "n1")
)

# block별 ordinal 변수
ORDINAL_VARS <- list(
  SUR_TSV_TCV_TA_TP = c("TSV", "TCV", "TA", "TP"),
  SUR_P_M = c(paste0("P", 1:7), paste0("M", 1:7)),
  SUR_P_minus_M = character(0),  # 차이값은 연속형처럼 처리
  EUP_ONLY = paste0("EUP", 1:18),
  EUP_SP_ONLY = c("SP1", "SP2", "SP3", "SP4"),
  BI_SEX_BMI_AGE = c("SEX"),
  BI_SEX_BMI = c("SEX"),
  TLX_S = c("TLX1", "TLX2", "TLX3", "TLX4", "TLX5", "TLX6", "S1", "S2"),
  BI_MBTI = c("E", "N", "T", "J", "A"),
  BI_FFM = c("o1", "c1", "e1", "a1", "n1"),
  BI_MBTI_FFM = c("E", "N", "T", "J", "A", "o1", "c1", "e1", "a1", "n1")
)

# block별 correlation method
BLOCK_METHOD <- list(
  SUR_TSV_TCV_TA_TP = "polychoric",
  SUR_P_M = "polychoric",
  SUR_P_minus_M = "pearson",
  EUP_ONLY = "polychoric",
  EUP_SP_ONLY = "polychoric",
  BI_SEX_BMI_AGE = "mixed",
  BI_SEX_BMI = "mixed",
  TLX_S = "polychoric",
  BI_MBTI = "polychoric",
  BI_FFM = "polychoric",
  BI_MBTI_FFM = "polychoric"
)

choose_df <- function(block_name) {
  if (startsWith(block_name, "SUR_")) return(list(df = SUR_AGG, native_unit = "SUR_ID_ENV_MEAN"))
  if (startsWith(block_name, "EUP_")) return(list(df = EUP, native_unit = "PERSON"))
  if (startsWith(block_name, "TLX_")) return(list(df = TLX, native_unit = "ID_ENV"))
  if (startsWith(block_name, "BI_"))  return(list(df = BI, native_unit = "PERSON"))
  stop(paste("Unknown block:", block_name))
}

# =========================
# 6) 상관행렬 생성
# =========================
manifest <- data.frame()

for (block_name in names(BLOCKS)) {
  cat("[RUN]", block_name, "\n")

  obj <- choose_df(block_name)
  df <- obj$df
  native_unit <- obj$native_unit

  cols <- intersect(BLOCKS[[block_name]], names(df))
  x <- df[, cols, drop = FALSE]
  x <- drop_constant_cols(x)

  if (ncol(x) < 2) {
    cat("  -> skipped: usable variable count < 2\n")
    next
  }

  ords <- intersect(ORDINAL_VARS[[block_name]], names(x))
  conts <- setdiff(names(x), ords)
  method <- BLOCK_METHOD[[block_name]]

  cat("  cols   :", paste(names(x), collapse = ", "), "\n")
  cat("  ords   :", paste(ords, collapse = ", "), "\n")
  cat("  conts  :", paste(conts, collapse = ", "), "\n")
  cat("  class  :", paste(sapply(x, class), collapse = ", "), "\n")

  corr <- NULL
  corr_method_label <- NULL

  if (method == "polychoric") {
    pc <- polychoric(x, correct = 0, smooth = TRUE)
    corr <- pc$rho
    corr_method_label <- "polychoric"
  } else if (method == "mixed") {
    ord_idx  <- which(names(x) %in% ords)
    cont_idx <- which(names(x) %in% conts)

    mc <- mixedCor(x, c = cont_idx, p = ord_idx, smooth = TRUE)
    corr <- mc$rho
    corr_method_label <- "mixedCor(polychoric/polyserial/Pearson)"
  } else {
    corr <- cor(x, use = "pairwise.complete.obs", method = "pearson")
    corr_method_label <- "pearson"
  }

  corr_path <- file.path(out_dir, paste0(block_name, "_corr.csv"))
  vars_path <- file.path(out_dir, paste0(block_name, "_vars.csv"))

  write.csv(corr, corr_path, row.names = TRUE)
  write.csv(data.frame(variable = colnames(x)), vars_path, row.names = FALSE)

  manifest <- rbind(
    manifest,
    data.frame(
      block = block_name,
      native_unit = native_unit,
      corr_method = corr_method_label,
      n = nrow(x),
      p = ncol(x),
      corr_csv = corr_path,
      vars_csv = vars_path,
      stringsAsFactors = FALSE
    )
  )
}

write.csv(manifest, file.path(out_dir, "manifest.csv"), row.names = FALSE)
cat("[DONE] Saved manifest and correlation matrices to:", out_dir, "\n")