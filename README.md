# 🌍 Crisis-Fusion: Multimodal Disaster Information Classification

A deep learning project for classifying disaster-related social media posts using **text and images** from the CrisisMMD v2.0 dataset. The project evaluates transformer-based text and image models, combines their predictions using multimodal score fusion, and provides an interactive Gradio demo for humanitarian information classification.

## 📌 Overview

During natural disasters, social media can provide valuable information about affected people, infrastructure damage, rescue operations, and humanitarian needs.

However, text and images may contain different but complementary information. This project investigates whether combining both modalities improves disaster-related classification.

The project evaluates four classification tasks:

- **Task 1a — Informativeness Classification:** Identifies whether a post contains informative disaster-related content.
- **Task 1b — Disaster Type Classification:** Classifies posts into disaster categories.
- **Task 2 — Humanitarian Information Classification:** Identifies the type of humanitarian information described in a post.
- **Task 3 — Damage Severity Classification:** Classifies damage as little or no damage, mild damage, or severe damage.

## ✨ Key Features

- Text classification using BERT, RoBERTa, and BERTweet.
- Image classification using Vision Transformer (ViT).
- Equal-weight probability-level fusion of text and image predictions.
- A cross-modal attention-based fusion architecture evaluated on Task 1a.
- Task 3 experiments using class weighting, image augmentation, and oversampling.
- Interactive Gradio demo for humanitarian information classification.
- Evaluation using accuracy, macro F1, weighted F1, balanced accuracy, and other classification metrics.

## 🧠 Methodology

### 1. Unimodal Classification

Text and image models are trained and evaluated separately.

| Modality | Models |
|---|---|
| Text | BERT, RoBERTa, BERTweet |
| Image | Vision Transformer (ViT) |

### 2. Multimodal Score Fusion

The probability distributions from the text and image classifiers are combined using equal weights.

\[
P_{\text{fused}} = 0.5P_{\text{text}} + 0.5P_{\text{image}}
\]

The class with the highest fused probability is selected as the final prediction.

This approach was evaluated across all four tasks.

### 3. Cross-Modal Fusion

A separate neural architecture was evaluated on Task 1a. It uses:

- RoBERTa for text encoding.
- ViT for image encoding.
- Cross-modal attention to exchange information between text and image features.
- Multimodal self-attention to refine the combined representations.
- A feed-forward network and classification head to generate predictions.

### 4. Task 3 Improvements

Additional experiments investigated class imbalance using:

- A baseline ViT classifier.
- Class-weighted training.
- Class weighting with image augmentation.
- Class weighting with augmentation and oversampling.

## 📊 Experimental Results

The following results are from the experiments conducted in this project.

### Score Fusion vs. Best Unimodal Model

| Task | Best Unimodal Accuracy | Score Fusion Accuracy | Change |
|---|---:|---:|---:|
| Task 1a — Informativeness | 87.03% | **91.13%** | +4.11 pp |
| Task 1b — Disaster Type | 86.57% | **90.29%** | +3.72 pp |
| Task 2 — Humanitarian Information | 82.30% | **86.28%** | +3.98 pp |
| Task 3 — Damage Severity | 64.27% | 63.14% | -1.13 pp |

*pp = percentage points. The reported change is the difference in accuracy.*

Score fusion improved accuracy on three of the four tasks. Task 3 remained challenging, highlighting the importance of class imbalance and class-aware evaluation.

### Task 3 Improvement Experiments

| Experiment | Accuracy | Macro F1 |
|---|---:|---:|
| Baseline | 66.35% | 45.64% |
| Class weights | 66.73% | 55.13% |
| Class weights + augmentation | 65.78% | **56.66%** |
| Class weights + augmentation + oversampling | 53.50% | 51.49% |

Class weighting combined with augmentation achieved the highest macro F1 among these improvement experiments.

### Cross-Modal Fusion — Task 1a

The separately evaluated cross-modal fusion checkpoint achieved:

- Accuracy: **89.50%**
- Weighted F1: **89.43%**

The 50/50 score-fusion model achieved higher Task 1a accuracy in these experiments.

## 🖥️ Interactive Demo

The project includes a Gradio application for **Task 2 — Humanitarian Information Classification**.

**Demo workflow:**

1. Enter disaster-related text.
2. Upload a corresponding disaster image.
3. Run the prediction.
4. View the confidence distribution across eight humanitarian categories.

The demo uses BERTweet for text classification and ViT for image classification, combining their probability distributions using 50/50 score fusion.

### Humanitarian Categories

- Affected individuals
- Infrastructure and utility damage
- Injured or dead people
- Missing or found people
- Not humanitarian
- Other relevant information
- Rescue, volunteering or donation effort
- Vehicle damage

**Demo link:** The Gradio share URL is temporary. Start the application using the instructions below to generate a new link when needed.

## 🗂️ Project Structure

```text
crisis-fusion/
├── data/
│   └── processed/
│       ├── task1a/
│       ├── task1b/
│       ├── task2/
│       └── task3/
├── baselines.py
├── fusion_model.py
├── prepare_data.py
├── score_fusion.py
├── evaluate.py
├── evaluate_score_fusion.py
├── improve_task3.py
├── demo.py
├── demo_task2.py
├── results.csv
├── task3_improvement_results.csv
└── final_report.md
```

*The structure above lists the main project files and may not include every file or directory in the repository.*

## ⚙️ Setup and Execution

### Requirements

- Python 3.10 or a compatible Python environment.
- PyTorch.
- Hugging Face Transformers.
- scikit-learn.
- pandas and NumPy.
- Pillow.
- Gradio.
- Access to the CrisisMMD v2.0 dataset and the required trained checkpoints.

A CUDA-enabled GPU is recommended for model training and inference.

### 1. Clone the Repository

```bash
git clone https://github.com/nikhil821023-commit/crisis-fusion.git
cd crisis-fusion
```

### 2. Install Dependencies

Install the libraries required by the scripts. If a `requirements.txt` file is available, use:

```bash
pip install -r requirements.txt
```

Otherwise, install the project dependencies according to the imports and model versions used by the scripts.

### 3. Prepare the Dataset

Download CrisisMMD v2.0 from the official CrisisNLP resource:

https://crisisnlp.qcri.org/crisismmd

Place the dataset in the expected directory structure and run the data preparation script according to its configuration.

### 4. Run the Task 2 Demo

Ensure that the trained checkpoints `best_task2_bertweet.pt` and `best_task2_vit.pt` are available in the project directory.

```bash
python demo_task2.py --share
```

Open the Gradio URL printed in the terminal to use the demo.


Result Screenshot:
<img width="1600" height="762" alt="image" src="https://github.com/user-attachments/assets/b42ccb08-e3aa-4db4-b9cb-56cc53dfdbc9" />
<img width="1600" height="773" alt="image" src="https://github.com/user-attachments/assets/cc42a8b0-4db3-4e40-a8db-ff83470cab96" />
<img width="1600" height="605" alt="image" src="https://github.com/user-attachments/assets/77d64f4d-eda6-4cb4-bbd4-dc71a0ba6baa" />



### 5. Run Other Experiments

Refer to the command-line arguments and configuration in each script before running training or evaluation. Training transformer models can require substantial GPU memory and time.

## 📈 Evaluation

The project evaluates models using multiple metrics, including:

- Accuracy
- Precision and recall
- Macro F1 and weighted F1
- Balanced accuracy
- Matthews correlation coefficient
- Cohen's kappa
- ROC-AUC where applicable
- Confusion matrices and per-class metrics

Macro F1 and balanced accuracy are especially useful for understanding performance on underrepresented classes.

## ⚠️ Limitations

- Equal 50/50 fusion weights are fixed rather than learned.
- Task 3 performance is affected by class imbalance.
- The cross-modal fusion architecture was evaluated on Task 1a; it was not completed across all four tasks.
- Model predictions are not guaranteed to be correct and should not be treated as verified emergency reports.
- The Gradio share link is temporary and is not a permanent production deployment.

## 🚀 Future Work

- Learn task-specific or adaptive fusion weights.
- Extend cross-modal fusion experiments to additional tasks.
- Improve minority-class performance for damage severity classification.
- Conduct further hyperparameter tuning and ablation experiments.
- Deploy the demo through a persistent hosting service.
- Explore real-time disaster information classification.

## 📚 Dataset and References

This project uses the CrisisMMD multimodal crisis dataset.

Alam, F., Ofli, F., & Imran, M. (2018). *CrisisMMD: Multimodal Twitter Datasets from Natural Disasters.* Proceedings of the International AAAI Conference on Web and Social Media (ICWSM).

Dataset resource: https://crisisnlp.qcri.org/crisismmd

## 👨‍💻 Author

**Nikhil Kumar**

GitHub: [@nikhil821023-commit](https://github.com/nikhil821023-commit)

---

*This project is a research prototype exploring multimodal machine learning for disaster-related social media classification.*
