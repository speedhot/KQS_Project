# The First Code File 01_ToPeerJKQ05PaperLLM_and_Optimization.ipynb
# KQS_Project
This repository contains the implementation of a framework designed to evaluate the quality of university-level lectures. It utilizes Large Language Models (LLMs) and multi-metric optimization to compute a Knowledge Quality Score (KQS), ensuring lecture content aligns with defined learning objectives.

# 📖 Overview
The first notebook, 01_ToPeerJKQ05PaperLLM_and_Optimization.ipynb, automates the process of summarizing educational content and scoring it against ground-truth objectives. This research contributes to Agentic AI applications for SDG 4 (Quality Education).
# Key Features:
-LLM-Powered Summarization: Uses GPT-4o to condense long lecture transcripts into concise, meaningful summaries.
-Multi-Metric Evaluation: Integrates semantic similarity, concept extraction, and information theory.
-Batch Processing: Processes multiple lectures and objectives simultaneously, exporting results to Excel.
# 📐 The KQS Model 
The Knowledge Quality Score (KQS) is a composite metric calculated using:
1. Semantic Score: Calculated via BERT embeddings (MiniLM-L6-v2) and cosine similarity.
2. KG Score (Knowledge Graph): Measures concept overlap using Jaccard Similarity on noun chunks extracted via spaCy.
3. Mutual Information (MI): An entropy-based calculation measuring information gain between the source lecture and the summary.
   # The final score is derived as follows:
  1. Initially Calculated Linear Combination with Equal Weights(Normalized to a 0.0–1.0 range).
  2. Later in the next code file 02_ToPeerj1500_4500Lect_KQS_Weight_Optimization_Demo.ipynb different weighting schemes and mathematics were experimented.

   # 🛠️ Prerequisites & Setup
   To run this notebook, you will need an OpenAI API key and the following Python environment:
   # Installation of required libraries
                     !pip install openai pandas spacy sentence-transformers openpyxl
                     !python -m spacy download en_core_web_sm
# Dataset Structure 
# The code expects a zip file named KQ_FEED_Final_Meaningful_Lectures_Dataset.zip 
# with this structure:
# /lectures/: Text files of full transcripts.
# /objectives/: Text files of corresponding learning outcomes.
🚀 UsageInitialize API: Update the client object with your OpenAI API key in the second cell.
Path Configuration: Ensure the dataset zip path matches your local or Colab environment.Run All: Execute the cells to generate KQS_Batch_Results.xlsx, which contains the detailed scoring for each lecture.



# KQS Weight Optimization & Learning Analytics (Phase II)
# The Second Code File 01_ToPeerJKQ05PaperLLM_and_Optimization.ipynb
# KQS Weight Optimization & Learning Analytics PIPELINE
This repository contains the second phase of the Knowledge Quality Score (KQS) research. While the initial phase established a linear formation for lecture evaluation, this notebook implements Experimental Weight Optimization and Learning Analytics to refine how Semantic, Structural, and Informational metrics contribute to the final quality score.

# 🔬 Research Context
This module analyzes university-level lectures (ranging from 1,500 to 4,500 words) to determine how varying the importance of Semantic, Structural, and Informational metrics affects the final assessment of educational quality. This research is a core component of utilizing Agentic AI for SDG 4 (Quality Education).

# HKQS-Aligned Knowledge Quantification Score (KQS) — Colab Experiment Description

This notebook implements a **human-aligned optimization framework** for lecture-level Knowledge Quantification Score (KQS).  
The goal is to **mathematically align automated knowledge metrics with Human Expert Knowledge Quantification Scores (HKQS)** using supervised weight learning.

# Colab Notebook: KQS Weight Optimization + Feedback Analytics (Editor Submission README)

This repository contains a **Google Colab notebook** that demonstrates an end-to-end experimental workflow for:

1) **Optimizing Knowledge Quantification Score (KQS) fusion weights** against **Human Expert Knowledge Quantification Score (HKQS)**, and  
2) Running **student + faculty feedback analytics** to validate whether the computed **Final_KQS** is consistent with perceived clarity and objective alignment.

Notebook file:
- `02_ToPeerj1500_4500Lect_KQS_Weight_Optimization_Demo.ipynb`

> All file paths inside the notebook are Colab-style: `/content/...`

---

## 1. What the notebook is computing

### Core goal (HKQS-aligned optimization)
The notebook treats KQS as a **composite score** formed by fusing three lecture-level component signals:

- `Semantic_Score`
- `KG_Score` (knowledge-graph/structural signal)
- `Mutual_Info` (information-theoretic signal)

It then learns weights so that the fused score matches expert labels:

- `HKQS` (or `HumanExpert_Score(HKQS)` depending on the dataset)

---

## 2. Datasets used by the notebook (as referenced in Colab)

### A) HKQS weight optimization (initial demo)
- `/content/KQS_Batch_Results.xlsx`  
  Contains computed component columns (e.g., `Semantic_Score`, `KG_Score`, `Mutual_Info`) and `Lecture_File`.
- `/content/HKQS_Unique_Domain_Expert_Scores.csv`  
  Contains `Lecture_File` and the human expert label column.

The notebook merges these two on `Lecture_File`.

### B) Lecture-scale CSV experiments
Used in later sections to run the same fusion + analysis at different scales:
- `/content/lectures_1000.csv`
- `/content/lectures_45000.csv`

These CSVs are expected to contain:
- `Lecture_File`
- `Semantic_Score`, `KG_Score`, `Mutual_Info`
- `HKQS`
- and in some parts: `Observed_KQS` and `Final_KQS`

### C) IGI/PAS computation and plots (4500 lectures)
- `/content/Normalized 4500 OriginalVs100AugLEct_FOR_PAS_and_IGI .xlsx`
  - must contain: `Mutual_Info`
- Output written by the notebook:
  - `Normalized_4500_With_All_IGI_Factors.xlsx` (saved into `/content/`)

This output is then reused for plotting:
- `Normalized_IGI`
- `PAS`

### D) Student feedback merge + analytics
- `/content/UniqueStudent_Feedback_1500_Entries.csv`
  - replicated/expanded student feedback entries with `Lecture_id`
- `/content/300_lecture_kqs_dataset.xlsx`
  - lecture-level scores with `Lecture_id` and `Final_KQS`
- Output produced by the notebook:
  - `/content/Student_Feedback_With_KQS.xlsx`

### E) Objective alignment + clarity feedback analytics
- `/content/Student_Feedback_Objective_Aligned.xlsx`
- `/content/Faculty_Feedback_Objective_Alignment.xlsx`

Expected fields used by analytics cells include:
- `Lecture_id`
- `Final_KQS`
- `Clarity_Rating_1_to_5`
- For faculty objective-alignment plot: `Objective_Aligned`
- For faculty interpretability plot: `Understandable`
- For participation stats: `Student_ID` and `Faculty_ID`

---

## 3. Notebook sections (what happens inside)

### Step 1 — Load KQS scores and HKQS
- Reads KQS component file and HKQS expert labels.
- Cleans column names (strip spaces).
- Merges on `Lecture_File`.

**Result:** `merged_df` containing predictors + expert labels.

---

### Step 2 — Prepare training matrix
- Builds:
  - `X = [Semantic_Score, KG_Score, Mutual_Info]`
  - `Y = HKQS`

This converts the problem into supervised regression of HKQS using fused component metrics.

---

### Step 3 — Weight tuning using Adam
Two weight-optimization approaches appear:

#### (A) Adam optimization (linear fusion)
- Initializes weights `w` (3 values) and normalizes them.
- Runs epochs (e.g., 1000) minimizing MSE between:
  - `preds = X @ w`
  - `Y = HKQS`
- Tracks loss curve.

**Outcome:** optimized weights that improve agreement with HKQS.

#### (B) Softmax-constrained weights + Power Mean fusion (updated math)
The notebook also includes an “UPDATED CODE (NEW MATHS…)” section that:
- Uses **softmax(theta)** to ensure weights remain:
  - non-negative
  - sum to 1 (convex weights)
- Computes fused score using **power mean** (p-mean) fusion:

If `p != 0`:
\[
KQS_p = \Big(\sum_i w_i \cdot x_i^p \Big)^{1/p}
\]

If `p = 0`:
- Uses the geometric-mean limit form via weighted logs.

This is run on:
- 1000-lecture dataset (`lectures_1000.csv`)
- and also on larger datasets (e.g., `lectures_45000.csv`)

---

## 4. Ablation and robustness experiments included

### 4.1 p-Sensitivity ablation
- Evaluates multiple p values (e.g., `[-2, -1, 0.5, 1, 2, 4, 8]`)
- Computes MSE for each p
- Plots p vs MSE

**Purpose:** justify the chosen exponent `p` for power-mean fusion.

---

### 4.2 Component-removal ablation
Tests KQS performance when removing components:

- All components
- Without PAS / without JI / without MI (implemented as masks on features)

**Outputs:** a table of MSE and R² for each ablation case + bar plot.

---

### 4.3 Different weighting schemes comparison
The notebook defines and evaluates multiple weight selection strategies (seen in later cells), such as:
- Equal weights
- Heuristic weights
- Random simplex weights (sampling weights that sum to 1)
- Other baseline approaches

Metrics computed typically include:
- MSE, MAE, RMSE, R²

---

### 4.4 k-fold cross-validation + mirror descent
A dedicated section runs:
- 5-fold CV (`KFold(n_splits=5, shuffle=True, random_state=42)`)
- Compares prediction strategies:
  - Equal
  - Heuristic
  - OLS regression
  - Mirror descent optimization over softmax weights

**Purpose:** show generalization stability and compare optimizers.

---

## 5. Visualization sections (graphs generated)

The notebook includes multiple figure outputs, including:

- Heatmap (typically for correlations or metric comparisons)
- Bar charts (e.g., ablation MSE)
- Scatter plot: **Optimized KQS vs HKQS**
- Box plot: score distribution comparisons
- Graph comparisons for large-scale datasets (e.g., 45,000 lectures)
- IGI comparison plot
- PAS plot (original vs augmented sampling)

---

## 6. IGI computation block (Instructional Gap Index factors)

The notebook computes IGI-related columns from `Mutual_Info`:

- Estimates an entropy scale using max MI (`H_est`)
- Computes:
  - `Estimated_Entropy_RS`
  - `Normalized_MI = Mutual_Info / H_est` (clipped to [0,1])
  - `Conditional_Entropy_RS_given_GS = H_est - Mutual_Info`
  - `Normalized_Conditional_Entropy`
  - `Normalized_IGI = 1 - Normalized_MI`

It saves:
- `Normalized_4500_With_All_IGI_Factors.xlsx`

This file is then used for:
- IGI comparison plotting
- PAS plotting across:
  - first 1000 original lectures
  - 1000 randomly sampled augmented lectures

---

## 7. Student & faculty feedback analytics (what exactly happens)

### 7.1 Feedback preparation (merging KQS with student entries)
Inputs:
- `UniqueStudent_Feedback_1500_Entries.csv` (student feedback, repeated across lecture ids)
- `300_lecture_kqs_dataset.xlsx` (lecture scores with `Final_KQS`)

Process:
- Normalizes/auto-detects the `Lecture_id` column name in the score sheet
- Merges on `Lecture_id`
- Exports:
  - `Student_Feedback_With_KQS.xlsx`

**Purpose:** attach the computed lecture KQS score to each student feedback entry.

---

### 7.2 Faculty objective-alignment summary plot
Input:
- `Faculty_Feedback_Objective_Alignment.xlsx`

Generates:
- Count plot of `Objective_Aligned` (faculty opinion on whether the summary aligns with objectives)

---

### 7.3 Correlation analysis: KQS vs clarity ratings
Inputs:
- `Student_Feedback_Objective_Aligned.xlsx`
- `Faculty_Feedback_Objective_Alignment.xlsx`

Process:
- Aggregates per lecture:
  - Student mean clarity rating: `Avg(SCR)`
  - Faculty mean clarity rating: `Avg(FCR)`
  - Uses `Final_KQS` as `KQS`
- Computes and prints correlation matrix between:
  - `KQS`, `Avg(SCR)`, `Avg(FCR)`

---

### 7.4 Distribution analytics (boxplots + scatter with regression)
Generates:
- Box plot: `Final_KQS` vs Student clarity rating (1–5)
- Box plot: `Final_KQS` vs Faculty clarity rating (1–5)
- Scatter/regression plots:
  - KQS vs SCR with correlation r
  - KQS vs FCR with correlation r
  - combined overlay plot for direct comparison

---

### 7.5 Participation & summary statistics
Generates:
- Student participation:
  - total entries
  - unique `Student_ID`
  - rating counts (1–5)
  - rating distribution plot
- Faculty participation:
  - total entries
  - unique `Faculty_ID`
  - rating counts (1–5)
  - additional faculty summary charts:
    - Pie chart of `Understandable` (yes/no)
    - Histogram of clarity ratings

**Purpose:** provide an educational validation layer showing how computed KQS relates to perception-based feedback.

---

## 8. How to run in Colab (minimal)

1. Upload the notebook to Colab
2. Upload the required input files into `/content/`
3. Run cells top-to-bottom

If using Google Drive, replace `/content/...` with your Drive-mounted path.

---

## 9. Notes for  reproducibility

- The notebook is structured as a **research demonstration**: multiple dataset-scale experiments are included (1000 / 4500 / 45000).
- Sections are independent; reviewers can run:
  - HKQS weight tuning only, or
  - IGI/PAS computation only, or
  - feedback analytics only,
  depending on available files.

---

## Author
Aman Kumar
