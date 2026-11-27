#!/usr/bin/env Rscript
# ============================================================
# 021 - 小管 pseudotime V3: 基于基底膜JSON提取的 ~110 个新特征
# 方法: Seurat + corral CA + UMAP + slingshot
# 输入: tubule_features_all_*.csv (小管级, 皮质+W0/W1/W2)
# 输出: tubule_pseudotime_newfeats.csv + patient 级聚合
# ============================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(corral)
  library(dplyr)
  library(ggplot2)
  library(slingshot)
  library(SingleCellExperiment)
  library(S4Vectors)
})

proj_root <- Sys.getenv("PSMP_DATA_ROOT", "./discovery_data")
csv_dir <- file.path(proj_root, "009_csv")
img_dir <- file.path(proj_root, "010_image")

ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
next_seq <- function(dir) {
  files <- list.files(dir); nums <- as.integer(sub("^(\\d+).*$", "\\1", files))
  nums <- nums[!is.na(nums)]; sprintf("%03d", if (length(nums)) max(nums) + 1 else 1)
}

# ============================================================
# § 1 读入新特征 CSV
# ============================================================
cat("=== Loading tubule features CSV ===\n")
feat_files <- list.files(csv_dir, pattern = "^tubule_features_all_.*\\.csv$", full.names = TRUE)
feat_csv <- tail(sort(feat_files), 1)
cat(sprintf("  Input: %s\n", basename(feat_csv)))

df <- read.csv(feat_csv, stringsAsFactors = FALSE, check.names = FALSE)
cat(sprintf("  Shape: %d x %d\n", nrow(df), ncol(df)))
cat(sprintf("  Samples: %d\n", length(unique(df$sample))))

# ============================================================
# § 2 选择数值特征列 (排除 meta/坐标/分类)
# ============================================================
meta_cols <- c("sample", "tubule_id", "tubule_type", "TBM_class", "cx_um", "cy_um", "is_cortex")
# 分类特征也作为数值输入 (W_grade, lumen_level 等)
exclude_cols <- c("sample", "tubule_id", "tubule_type", "TBM_class", "cx_um", "cy_um", "is_cortex")

# 全部数值列 (含分类)
all_cols <- colnames(df)
feat_cols <- setdiff(all_cols, exclude_cols)

# 排除 NaN 太多的列 (>50% missing)
na_pct <- sapply(df[, feat_cols], function(x) mean(is.na(x)))
good_feats <- feat_cols[na_pct < 0.5]
cat(sprintf("  Features: %d total, %d after NaN filter (>50%% missing removed)\n",
            length(feat_cols), length(good_feats)))

# 移除方差为0的列
vars <- sapply(df[, good_feats], function(x) var(x, na.rm = TRUE))
good_feats <- good_feats[!is.na(vars) & vars > 1e-12]
cat(sprintf("  After zero-variance filter: %d features\n", length(good_feats)))

# 打印特征列表
cat("  Feature list:\n")
for (f in good_feats) cat(sprintf("    %s\n", f))

# ============================================================
# § 3 Subsample (每样本最多 350 管, 与 V2 保持一致)
# ============================================================
set.seed(42)
df_sub <- df %>% group_by(sample) %>%
  slice_sample(prop = 1, replace = FALSE) %>%
  slice_head(n = 350) %>%
  ungroup() %>% as.data.frame()
cat(sprintf("\n  After subsample (<=350/sample): %d tubules\n", nrow(df_sub)))

# ============================================================
# § 4 Seurat + corral CA + UMAP + slingshot
# ============================================================
cat("\n=== Building Seurat object ===\n")
mat_raw <- as.matrix(df_sub[, good_feats])
mat_raw[is.na(mat_raw)] <- 0
mat_raw[mat_raw < 0] <- 0
mat_seu <- t(mat_raw)
rownames(mat_seu) <- gsub("_", "-", good_feats)
colnames(mat_seu) <- paste0("tub_", seq_len(ncol(mat_seu)))

seu <- CreateSeuratObject(counts = mat_seu, assay = "morph",
                           meta.data = data.frame(sample = df_sub$sample,
                                                   W_grade = df_sub$W_grade,
                                                   row.names = colnames(mat_seu)))
seu <- NormalizeData(seu, normalization.method = "RC", scale.factor = 1, verbose = FALSE)
seu <- ScaleData(seu, features = rownames(seu), verbose = FALSE)
cat("  Seurat done\n")

cat("  Running corral CA...\n")
n_comp <- min(15, length(good_feats) - 1)
ca_res <- corral::corral(GetAssayData(seu, layer = "data", assay = "morph"),
                          ncomp = n_comp)
emb <- ca_res$v
rownames(emb) <- colnames(seu)
colnames(emb) <- paste0("CA_", seq_len(ncol(emb)))
seu[["ca"]] <- CreateDimReducObject(embeddings = emb, key = "CA_", assay = "morph")
cat(sprintf("  CA done: %d components\n", ncol(emb)))

cat("  Running UMAP...\n")
n_dims <- min(10, ncol(emb))
seu <- RunUMAP(seu, reduction = "ca", dims = 1:n_dims, verbose = FALSE)
umap_emb <- Embeddings(seu, "umap")
cat("  UMAP done\n")

cat("  Finding clusters (Louvain)...\n")
seu <- FindNeighbors(seu, reduction = "umap", dims = 1:2, verbose = FALSE)
seu <- FindClusters(seu, resolution = 0.15, algorithm = 1, verbose = FALSE)
cluster_lbl <- as.integer(as.character(seu$seurat_clusters))
cat(sprintf("  Clusters: %d (sizes: %s)\n", length(unique(cluster_lbl)),
            paste(table(cluster_lbl), collapse = ", ")))

cat("  Running slingshot...\n")
sce <- SingleCellExperiment::SingleCellExperiment(
  assays = list(logcounts = GetAssayData(seu, layer = "data", assay = "morph")),
  reducedDims = S4Vectors::SimpleList(UMAP = umap_emb))
SingleCellExperiment::colLabels(sce) <- factor(cluster_lbl)

# 确定起始 cluster: W0 比例最高的
w0_pct_by_cls <- tapply(df_sub$W_grade == 0, cluster_lbl, mean)
start_cls <- names(which.max(w0_pct_by_cls))
cat(sprintf("  Start cluster: %s (W0 pct = %.1f%%)\n", start_cls, w0_pct_by_cls[start_cls] * 100))
for (cl in sort(unique(cluster_lbl))) {
  n_cl <- sum(cluster_lbl == cl)
  w0p <- mean(df_sub$W_grade[cluster_lbl == cl] == 0) * 100
  w2p <- mean(df_sub$W_grade[cluster_lbl == cl] == 2) * 100
  cat(sprintf("    Cluster %s: n=%d, W0=%.0f%%, W2=%.0f%%\n", cl, n_cl, w0p, w2p))
}

ss_res <- tryCatch(
  slingshot::slingshot(sce, clusterLabels = "label", reducedDim = "UMAP",
                       start.clus = start_cls),
  error = function(e) { cat(sprintf("  slingshot FAIL: %s\n", e$message)); NULL }
)

if (!is.null(ss_res)) {
  pt <- slingshot::slingPseudotime(ss_res)
  cat(sprintf("  slingshot lineages: %d\n", ncol(pt)))
  pseudotime <- rowMeans(pt, na.rm = TRUE)
  pseudotime <- (pseudotime - min(pseudotime, na.rm = TRUE)) /
    max(pseudotime - min(pseudotime, na.rm = TRUE), na.rm = TRUE) * 100
} else {
  cat("  Fallback: cluster mean ordering\n")
  cls_x <- aggregate(mat_raw[, 1], by = list(cluster_lbl), FUN = mean)
  cls_x <- cls_x[order(cls_x$x), ]
  cluster_rank <- setNames(seq_along(cls_x$Group.1), cls_x$Group.1)
  base_score <- cluster_rank[as.character(cluster_lbl)]
  pseudotime <- (base_score - min(base_score)) / (max(base_score) - min(base_score)) * 100
}

df_sub$cluster <- cluster_lbl
df_sub$pseudotime <- pseudotime
df_sub$umap1 <- umap_emb[, 1]
df_sub$umap2 <- umap_emb[, 2]

# ============================================================
# § 5 翻转 pseudotime: 确保 W0 低, W2 高
# ============================================================
cat("\n=== Checking pseudotime direction ===\n")
w0_median <- median(df_sub$pseudotime[df_sub$W_grade == 0], na.rm = TRUE)
w2_median <- median(df_sub$pseudotime[df_sub$W_grade == 2], na.rm = TRUE)
cat(sprintf("  W0 median psd = %.1f, W2 median psd = %.1f\n", w0_median, w2_median))

if (w0_median > w2_median) {
  cat("  -> Flipping pseudotime (W0 should be low)\n")
  df_sub$pseudotime <- 100 - df_sub$pseudotime
  w0_median <- median(df_sub$pseudotime[df_sub$W_grade == 0], na.rm = TRUE)
  w2_median <- median(df_sub$pseudotime[df_sub$W_grade == 2], na.rm = TRUE)
  cat(sprintf("  After flip: W0=%.1f, W2=%.1f\n", w0_median, w2_median))
}

# 各 W grade 中位数
for (w in 0:2) {
  sub <- df_sub[df_sub$W_grade == w, ]
  cat(sprintf("  W%d: n=%d, psd median=%.1f, IQR=[%.1f, %.1f]\n",
              w, nrow(sub), median(sub$pseudotime, na.rm = TRUE),
              quantile(sub$pseudotime, 0.25, na.rm = TRUE),
              quantile(sub$pseudotime, 0.75, na.rm = TRUE)))
}

# ============================================================
# § 6 保存小管级 CSV
# ============================================================
base_seq <- as.integer(next_seq(csv_dir))
out_tub <- file.path(csv_dir, sprintf("%03d_%s-P7v3_tubule_pseudotime_newfeats.csv", base_seq, ts))
save_cols <- c("sample", "tubule_id", "cluster", "pseudotime", "umap1", "umap2",
               "W_grade", "tubule_type", "lumen_level", "brush_border", "epithelial_damage")
save_cols <- intersect(save_cols, colnames(df_sub))
write.csv(df_sub[, save_cols], out_tub, row.names = FALSE, fileEncoding = "UTF-8")
cat(sprintf("\n  Tubule pseudotime → %s (%d rows)\n", basename(out_tub), nrow(df_sub)))

# ============================================================
# § 7 Patient 级聚合
# ============================================================
cat("\n=== Patient-level aggregation ===\n")
patient_agg <- df_sub %>% group_by(sample) %>%
  summarise(
    tub_psd_v3_median = median(pseudotime, na.rm = TRUE),
    tub_psd_v3_Q1 = quantile(pseudotime, 0.25, na.rm = TRUE),
    tub_psd_v3_Q3 = quantile(pseudotime, 0.75, na.rm = TRUE),
    tub_psd_v3_IQR = quantile(pseudotime, 0.75, na.rm = TRUE) - quantile(pseudotime, 0.25, na.rm = TRUE),
    tub_psd_v3_p90 = quantile(pseudotime, 0.90, na.rm = TRUE),
    tub_psd_v3_mean = mean(pseudotime, na.rm = TRUE),
    tub_psd_v3_high_pct = mean(pseudotime > 66.67, na.rm = TRUE),
    n_tub_v3 = n(),
    W0_pct = mean(W_grade == 0, na.rm = TRUE),
    W1_pct = mean(W_grade == 1, na.rm = TRUE),
    W2_pct = mean(W_grade == 2, na.rm = TRUE),
    .groups = "drop"
  )
out_pat <- file.path(csv_dir, sprintf("%03d_%s-P7v3_patient_pseudotime_newfeats.csv", base_seq + 1, ts))
write.csv(patient_agg, out_pat, row.names = FALSE, fileEncoding = "UTF-8")
cat(sprintf("  Patient pseudotime → %s (%d patients)\n", basename(out_pat), nrow(patient_agg)))

# ============================================================
# § 8 Sanity: pseudotime vs clinical (if merged CSV available)
# ============================================================
cat("\n=== Sanity check ===\n")
merged_files <- list.files(csv_dir, pattern = "-P4v5_merged_.*\\.csv$", full.names = TRUE)
if (length(merged_files) > 0) {
  df_clin <- read.csv(tail(sort(merged_files), 1), stringsAsFactors = FALSE, check.names = FALSE)
  df_clin <- df_clin[, c("sample", "Class_All_ord", "eGFR_CKDEPI")]
  df_check <- patient_agg %>% left_join(df_clin, by = "sample")
  
  cat("  Pseudotime V3 median by Tervaert Class:\n")
  cls_map <- c("I", "IIA", "IIB", "III", "IV")
  for (cls in 1:5) {
    sub <- df_check[df_check$Class_All_ord == cls & !is.na(df_check$Class_All_ord), ]
    if (nrow(sub) > 0) {
      cat(sprintf("    Class %s (n=%d): median psd=%.1f, mean psd=%.1f\n",
                  cls_map[cls], nrow(sub), 
                  median(sub$tub_psd_v3_median, na.rm = TRUE),
                  mean(sub$tub_psd_v3_median, na.rm = TRUE)))
    }
  }
  
  # Spearman correlation
  ok <- !is.na(df_check$tub_psd_v3_median) & !is.na(df_check$Class_All_ord)
  if (sum(ok) > 5) {
    sp <- cor(df_check$tub_psd_v3_median[ok], df_check$Class_All_ord[ok], method = "spearman")
    cat(sprintf("  Spearman ρ(psd_v3 vs Class) = %.3f (n=%d)\n", sp, sum(ok)))
  }
  
  # Correlation with eGFR (should be negative)
  ok2 <- !is.na(df_check$tub_psd_v3_median) & !is.na(df_check$eGFR_CKDEPI)
  if (sum(ok2) > 5) {
    sp2 <- cor(df_check$tub_psd_v3_median[ok2], df_check$eGFR_CKDEPI[ok2], method = "spearman")
    cat(sprintf("  Spearman ρ(psd_v3 vs eGFR) = %.3f (should be negative)\n", sp2))
  }
}

# ============================================================
# § 9 Quick UMAP plot
# ============================================================
cat("\n=== Saving UMAP plot ===\n")
img_seq <- as.integer(next_seq(img_dir))
out_fig <- file.path(img_dir, sprintf("%03d_%s-tubule_pseudotime_v3_UMAP.png", img_seq, ts))

p <- ggplot(df_sub, aes(x = umap1, y = umap2, color = pseudotime)) +
  geom_point(alpha = 0.5, size = 0.5) +
  scale_color_viridis_c(option = "C") +
  labs(title = sprintf("Tubule pseudotime V3 (%d features, %d tubules)",
                        length(good_feats), nrow(df_sub)),
       subtitle = "Based on TBM JSON features | cortex + W0/W1/W2 only",
       x = "UMAP 1", y = "UMAP 2", color = "Pseudotime") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))
ggsave(out_fig, p, width = 9, height = 7, dpi = 200)
cat(sprintf("  → %s\n", basename(out_fig)))

cat("\n=== DONE ===\n")
