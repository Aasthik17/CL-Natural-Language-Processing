import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.axis('off')

# Box styling
box_style = dict(boxstyle="round,pad=0.5", fc="#F0F4F8", ec="#2B4C7E", lw=1.5)
stage_style = dict(boxstyle="round,pad=0.5", fc="#DCE6F1", ec="#1B365D", lw=2)
result_style = dict(boxstyle="round,pad=0.5", fc="#E2F0D9", ec="#385723", lw=1.5)

# 1. Ingestion Box
ax.text(0.12, 0.75, "Raw English\nText Stream\n(Unannotated)", ha="center", va="center", bbox=box_style, fontsize=10, weight="bold")

# 2. Stage 1: Classification Box
ax.text(0.40, 0.75, "Stage 1:\nDistilBERT Classifier\n(AG News Fine-Tuned)\nAccuracy: 93.27%", ha="center", va="center", bbox=stage_style, fontsize=10, weight="bold")

# 3. Category Router Box
ax.text(0.70, 0.75, "Semantic Domain\nRouter\n[World, Sports, Biz, Tech]", ha="center", va="center", bbox=box_style, fontsize=10, weight="bold")

# 4. Stage 2: Translation Box
ax.text(0.70, 0.35, "Stage 2:\nMarianMT Translation\n(English to Hindi)\nBLEU: 22.53 | chrF: 39.92", ha="center", va="center", bbox=stage_style, fontsize=10, weight="bold")

# 5. Stage 3: Summarization Box
ax.text(0.40, 0.35, "Stage 3:\nmBART Summarization\n(Abstractive Hindi)\nmBERT-F1: 0.5975", ha="center", va="center", bbox=stage_style, fontsize=10, weight="bold")

# 6. Final Executive Output
ax.text(0.12, 0.35, "Categorized,\nTranslated &\nSummarized Output", ha="center", va="center", bbox=result_style, fontsize=10, weight="bold")

# Arrows
arrow_kwargs = dict(arrowstyle="->", lw=2, color="#1B365D")

# Arrow 1: Input to Classifier
ax.annotate("", xy=(0.26, 0.75), xytext=(0.20, 0.75), arrowprops=arrow_kwargs)

# Arrow 2: Classifier to Router
ax.annotate("", xy=(0.56, 0.75), xytext=(0.54, 0.75), arrowprops=arrow_kwargs)

# Arrow 3: Router to Translation (Downwards)
ax.annotate("", xy=(0.70, 0.48), xytext=(0.70, 0.65), arrowprops=arrow_kwargs)

# Arrow 4: Translation to Summarization
ax.annotate("", xy=(0.54, 0.35), xytext=(0.57, 0.35), arrowprops=arrow_kwargs)

# Arrow 5: Summarization to Final Output
ax.annotate("", xy=(0.24, 0.35), xytext=(0.26, 0.35), arrowprops=arrow_kwargs)

# Add Title
plt.title("End-to-End Multilingual Intelligence Pipeline Architecture (Supplementary Mini-Project)", fontsize=12, weight="bold", pad=20, color="#1B365D")

plt.tight_layout()
plt.savefig("report_assets/miniproject_pipeline.png", dpi=300)
plt.close()

print("Successfully generated report_assets/miniproject_pipeline.png")
