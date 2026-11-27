#!/usr/bin/env Rscript
# ============================================================
# 脚本: 020_P7v3_Seurat轨迹_filtered100um_V1.R
# 功能: V5 P7 V3 - 用过滤后的 P2v4f (≥100μm, 1742 gloms) 做肾小球单结构轨迹
#
# 流水线: NormalizeData(RC) → corral CA → UMAP → Louvain → slingshot
#
# 输入: P2v4f_instance_glom_filtered100um CSV (1742 gloms)
# 输出:
#   - P7v3_glom_pseudotime_V3.csv (instance-level pseudotime)
#   - P7v3_patient_pseudotime_V3.csv (patient-level aggregation)
#   - Figure: glom UMAP colored by pseudotime
#   - Figure: feature heatmap along pseudotime
#   - Figure: pseudotime vs Class boxplot
# ============================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(corral)
  library(dplyr)
  library(ggplot2)
  library(tidyr)
  library(slingshot)
  library(SingleCellExperiment)
  library(S4Vectors)
})

proj_root <- Sys.getenv("PSMP_DATA_ROOT", "./discovery_data")
csv_dir <- file.path(proj_root, "009_csv")
img_dir <- file.path(proj_root, "010_image")
log_dir <- file.path(proj_root, "013_log")

ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
next_seq <- function(dir) {
  files <- list.files(dir); nums <- as.integer(sub("^(\\d+).*$", "\\1", files))
  nums <- nums[!is.na(nums)]; sprintf("%03d", if (length(nums)) max(nums) + 1 else 1)
}
base_csv <- as.integer(next_seq(csv_dir))
out_glom_psd <- file.path(csv_dir, sprintf("%03d_%s-P7v3_glom_pseudotime_V3.csv", base_csv, ts))
out_patient_psd <- file.path(csv_dir, sprintf("%03d_%s-P7v3_patient_pseudotime_V3.csv", base_csv + 1, ts))

img_seq <- as.integer(next_seq(img_dir))
out_fig_umap <- file.path(img_dir, sprintf("%03d_%s-Fig5_glom_V3psd_UMAP_filtered.png", img_seq, ts))
out_fig_heat <- file.path(img_dir, sprintf("%03d_%s-Fig5_glom_V3psd_heatmap_filtered.png", img_seq + 1, ts))
out_fig_box  <- file.path(img_dir, sprintf("%03d_%s-Fig5_glom_V3psd_vs_Class_filtered.png", img_seq + 2, ts))

log_file <- file.path(log_dir, sprintf("%s_%s-P7v3_Seurat_filtered_V1.log", next_seq(log_dir), ts))

log_msg <- function(...) {
  m <- paste(format(Sys.time(), "%H:%M:%S"), "|", ...)
  cat(m, "\n", file = log_file, append = TRUE); cat(m, "\n")
}

log_msg("==================================================")
log_msg("V5 P7 V3 - Glom trajectory (filtered ≥100μm, n≈1742)")
log_msg("==================================================")

# ============================================================
# § 1 读过滤后的 P2v4f
# ============================================================
glom_files <- list.files(csv_dir, pattern = "-P2v4f_instance_glom_filtered100um_.*\\.csv$",
                         full.names = TRUE)
if (length(glom_files) == 0) stop("找不到 P2v4f filtered CSV!")
glom_csv <- tail(sort(glom_files), 1)
log_msg(sprintf("input: %s", basename(glom_csv)))
df_g <- read.csv(glom_csv, stringsAsFactors = FALSE, check.names = FALSE)
log_msg(sprintf("shape: %d gloms x %d cols", nrow(df_g), ncol(df_g)))

# ============================================================
# § 2 选取特征
# ============================================================
glom_features <- c(
  # 几何形态
  "glom_area_um2", "tuft_area_um2", "tuft_area_fraction", "bowman_area_um2",
  "perimeter_um", "circularity", "solidity", "eccentricity", "elongation",
  "glom_dia_outer_um", "tuft_dia_inner_um",
  "major_axis_um", "minor_axis_um",
  # 病理标签
  "mesangial_W_int",
  "n_GBM_patches", "n_FSGS_patches",
  "has_global_sclerosis", "has_segmental_sclerosis_FSGS",
  "has_crescent_strict", "has_necrosis",
  "n_intraglom_inflam_cells",
  "dist_nearest_glom_um",
  # 新 JSON 特征 (P2v4 独有)
  "has_GBM_thicken", "has_broken_capsule", "has_bowman_caps"
)
glom_features <- intersect(glom_features, colnames(df_g))
log_msg(sprintf("features used: %d", length(glom_features)))
log_msg(sprintf("  features: %s", paste(glom_features, collapse = ", ")))

# ============================================================
# § 3 构建 Seurat 对象 + 降维
# ============================================================
mat_raw <- as.matrix(df_g[, glom_features])
mat_raw[is.na(mat_raw)] <- 0
mat_raw[mat_raw < 0] <- 0  # mesangial_W_int=-1 → 0

# 去全零行
keep <- rowSums(mat_raw) > 0
df_g <- df_g[keep, ]
mat_raw <- mat_raw[keep, ]
log_msg(sprintf("non-zero instances: %d", nrow(df_g)))

mat_seu <- t(mat_raw)
rownames(mat_seu) <- gsub("_", "-", glom_features)
colnames(mat_seu) <- paste0("g", seq_len(ncol(mat_seu)))

log_msg("[Seurat] CreateSeuratObject + NormalizeData(RC) + ScaleData")
seu <- CreateSeuratObject(counts = mat_seu, assay = "morph",
                           meta.data = data.frame(sample = df_g$sample,
                                                   row.names = colnames(mat_seu)))
seu <- NormalizeData(seu, normalization.method = "RC", scale.factor = 1, verbose = FALSE)
seu <- ScaleData(seu, features = rownames(seu), verbose = FALSE)

log_msg("[corral] Pearson Residual CA")
ca_res <- corral::corral(GetAssayData(seu, layer = "data", assay = "morph"),
                          ncomp = min(15, length(glom_features) - 1))
emb <- ca_res$v
rownames(emb) <- colnames(seu)
colnames(emb) <- paste0("CA_", seq_len(ncol(emb)))
seu[["ca"]] <- CreateDimReducObject(embeddings = emb, key = "CA_", assay = "morph")

log_msg("[UMAP]")
seu <- RunUMAP(seu, reduction = "ca",
                dims = 1:min(10, ncol(Embeddings(seu, "ca"))), verbose = FALSE)
umap_emb <- Embeddings(seu, "umap")

log_msg("[Louvain]")
seu <- FindNeighbors(seu, reduction = "umap", dims = 1:2, verbose = FALSE)
seu <- FindClusters(seu, resolution = 0.5, algorithm = 1, verbose = FALSE)
cluster_lbl <- as.integer(as.character(seu$seurat_clusters))
log_msg(sprintf("  clusters: %d (sizes: %s)", length(unique(cluster_lbl)),
                paste(table(cluster_lbl), collapse = ", ")))

# ============================================================
# § 4 slingshot pseudotime
# ============================================================
log_msg("[slingshot]")
sce <- SingleCellExperiment::SingleCellExperiment(
  assays = list(logcounts = GetAssayData(seu, layer = "data", assay = "morph")),
  reducedDims = S4Vectors::SimpleList(UMAP = umap_emb))
SingleCellExperiment::colLabels(sce) <- factor(cluster_lbl)
ss_res <- tryCatch(
  slingshot::slingshot(sce, clusterLabels = "label", reducedDim = "UMAP"),
  error = function(e) { log_msg(sprintf("slingshot FAIL: %s", e$message)); NULL }
)

if (!is.null(ss_res)) {
  pt <- slingshot::slingPseudotime(ss_res)
  log_msg(sprintf("  slingshot lineages: %d", ncol(pt)))
  pseudotime <- rowMeans(pt, na.rm = TRUE)
  pseudotime <- (pseudotime - min(pseudotime, na.rm = TRUE)) /
    max(pseudotime - min(pseudotime, na.rm = TRUE), na.rm = TRUE) * 100
  log_msg(sprintf("  pseudotime range: %.1f - %.1f", min(pseudotime, na.rm=TRUE), max(pseudotime, na.rm=TRUE)))
} else {
  log_msg("[fallback] cluster mean ordering by mesangial_W_int")
  cls_W <- aggregate(mat_raw[, "mesangial-W-int"],
                     by = list(cluster = cluster_lbl), FUN = function(x) mean(x, na.rm = TRUE))
  cls_W <- cls_W[order(cls_W$x), ]
  cluster_rank <- setNames(seq_along(cls_W$cluster), cls_W$cluster)
  base_score <- cluster_rank[as.character(cluster_lbl)]
  pseudotime <- (base_score - min(base_score)) / (max(base_score) - min(base_score)) * 100
  log_msg(sprintf("  fallback range: %.1f - %.1f", min(pseudotime), max(pseudotime)))
}

df_g$cluster <- cluster_lbl
df_g$pseudotime <- pseudotime
df_g$umap1 <- umap_emb[, 1]
df_g$umap2 <- umap_emb[, 2]

# ============================================================
# § 5 保存 instance-level pseudotime
# ============================================================
save_cols <- intersect(c("sample", "glom_id", "glom_idx", "cluster", "pseudotime",
                          "umap1", "umap2", "mesangial_W_int",
                          "has_GBM_thicken", "has_global_sclerosis",
                          "has_segmental_sclerosis_FSGS", "has_crescent_strict",
                          "has_necrosis", "n_intraglom_inflam_cells",
                          "glom_area_um2", "tuft_area_um2", "tuft_area_fraction",
                          "circularity", "eccentricity", "dia_um"),
                        colnames(df_g))
write.csv(df_g[, save_cols], out_glom_psd, row.names = FALSE, fileEncoding = "UTF-8")
log_msg(sprintf("Pseudotime → %s (%d gloms)", basename(out_glom_psd), nrow(df_g)))

# ============================================================
# § 6 Patient-level aggregation
# ============================================================
log_msg("\n[Patient pseudotime V3 聚合]")
patient_g <- df_g %>% group_by(sample) %>%
  summarise(
    glom_psd_v3_median = median(pseudotime, na.rm = TRUE),
    glom_psd_v3_Q1 = quantile(pseudotime, 0.25, na.rm = TRUE),
    glom_psd_v3_Q3 = quantile(pseudotime, 0.75, na.rm = TRUE),
    glom_psd_v3_IQR = quantile(pseudotime, 0.75, na.rm = TRUE) -
                      quantile(pseudotime, 0.25, na.rm = TRUE),
    glom_psd_v3_p90 = quantile(pseudotime, 0.90, na.rm = TRUE),
    glom_psd_v3_mean = mean(pseudotime, na.rm = TRUE),
    glom_psd_v3_high_tertile_pct = mean(pseudotime > 66.67, na.rm = TRUE),
    n_glom_v3_filtered = n(),
    .groups = "drop"
  )
log_msg(sprintf("patient psd V3: %d patients × %d cols", nrow(patient_g), ncol(patient_g)))
write.csv(patient_g, out_patient_psd, row.names = FALSE, fileEncoding = "UTF-8")
log_msg(sprintf("  → %s", basename(out_patient_psd)))

# ============================================================
# § 7 Figures
# ============================================================
log_msg("\n[Figures]")

# (a) UMAP colored by pseudotime
p_umap <- ggplot(df_g, aes(x = umap1, y = umap2, color = pseudotime)) +
  geom_point(alpha = 0.5, size = 0.8) +
  scale_color_viridis_c(option = "plasma") +
  labs(x = "UMAP 1", y = "UMAP 2", color = "Pseudotime",
       title = "Glomerular trajectory (filtered ≥100μm)",
       subtitle = sprintf("n=%d glomeruli | %d features | slingshot pseudotime",
                           nrow(df_g), length(glom_features))) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))
ggsave(out_fig_umap, p_umap, width = 8, height = 6, dpi = 150)
log_msg(sprintf("  → %s", basename(out_fig_umap)))

# (b) Feature heatmap along pseudotime
df_g$psd_bucket <- cut(df_g$pseudotime, breaks = seq(0, 100, length.out = 9),
                        include.lowest = TRUE, labels = paste0("B", 1:8))
feats_heat <- c("glom_area_um2", "tuft_area_um2", "tuft_area_fraction",
                 "circularity", "eccentricity",
                 "has_GBM_thicken", "has_global_sclerosis",
                 "has_segmental_sclerosis_FSGS", "n_intraglom_inflam_cells")
feats_heat <- intersect(feats_heat, colnames(df_g))
heat_data <- df_g %>%
  group_by(psd_bucket) %>%
  summarise(across(all_of(feats_heat), \(x) mean(x, na.rm = TRUE)), .groups = "drop") %>%
  pivot_longer(cols = -psd_bucket, names_to = "feature", values_to = "value") %>%
  group_by(feature) %>%
  mutate(z = (value - mean(value, na.rm = TRUE)) / max(sd(value, na.rm = TRUE), 1e-9)) %>%
  ungroup()

p_heat <- ggplot(heat_data, aes(x = psd_bucket, y = feature, fill = z)) +
  geom_tile() +
  scale_fill_gradient2(low = "#2C7BB6", mid = "white", high = "#D7191C", midpoint = 0) +
  labs(x = "Pseudotime bucket (low → high)", y = NULL, fill = "Z-score",
       title = "Feature heatmap along V3 pseudotime (filtered ≥100μm)",
       subtitle = "Mean per bucket, z-scored") +
  theme_minimal(base_size = 10)
ggsave(out_fig_heat, p_heat, width = 9, height = 5, dpi = 150)
log_msg(sprintf("  → %s", basename(out_fig_heat)))

# (c) Pseudotime vs Class boxplot
merged_files <- list.files(csv_dir, pattern = "-P4v5_merged_.*\\.csv$", full.names = TRUE)
if (length(merged_files) == 0) {
  merged_files <- list.files(csv_dir, pattern = "-P4v4_merged.*\\.csv$", full.names = TRUE)
}
if (length(merged_files) > 0) {
  merged_csv <- tail(sort(merged_files), 1)
  log_msg(sprintf("clinical from: %s", basename(merged_csv)))
  df_clin <- read.csv(merged_csv, stringsAsFactors = FALSE, check.names = FALSE)
  df_clin_sub <- df_clin[, c("sample", "Class_All_ord")]
  
  plot_df <- patient_g %>% left_join(df_clin_sub, by = "sample") %>%
    filter(!is.na(Class_All_ord))
  
  p_box <- ggplot(plot_df, aes(x = factor(Class_All_ord,
                                labels = c("I","IIA","IIB","III","IV")),
                                y = glom_psd_v3_median,
                                fill = factor(Class_All_ord))) +
    geom_boxplot(alpha = 0.85, outlier.size = 1) +
    geom_jitter(width = 0.15, size = 1.5, alpha = 0.7) +
    scale_fill_manual(values = c("1"="#2C7BB6","2"="#ABD9E9","3"="#FFFFBF",
                                   "4"="#FDAE61","5"="#D7191C"), guide = "none") +
    labs(x = "Tervaert Class", y = "Glom Pseudotime V3 median",
         title = "V3 Pseudotime (filtered ≥100μm) vs Tervaert Class") +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold"))
  ggsave(out_fig_box, p_box, width = 7, height = 5, dpi = 150)
  log_msg(sprintf("  → %s", basename(out_fig_box)))
  
  # Spearman
  sp <- cor(plot_df$glom_psd_v3_median, plot_df$Class_All_ord, method = "spearman",
            use = "complete.obs")
  log_msg(sprintf("  Spearman ρ (V3 psd median vs Class): %.3f (n=%d)", sp, nrow(plot_df)))
} else {
  log_msg("  No merged CSV found, skip Class plot")
}

log_msg("\n==================================================")
log_msg("P7v3 DONE")
log_msg("==================================================")
