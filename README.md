# Addressing-Bias-in-Recruitment-Algorithms

## Problem Statement:
More and more companies are turning to recruitment algorithms to make the hiring process smoother. These algorithms review resumes, media profiles, and other personal information to assess and prioritize job applicants. However, there is a rising worry, about the fairness and issues linked to these systems. The use of recruitment algorithms can unintentionally uphold biases found in the data resulting in discrimination against certain demographics. This project aims to detect and mitigate biases in hiring tools created with several different AI models.

## Motivation:
Recruitment algorithms, when used in applications have the ability to improve the effectiveness of the hiring process. However, there are considerations that come into play when implementing them. Biased algorithms can perpetuate inequalities leading to biased hiring choices. It is essential to mitigate these biases to champion diversity and inclusivity in the workplace. Fair recruitment methods are not only a matter of morality but also contribute to organizational success by cultivating a varied workforce.

## Literature Review:
[1] Researchers have put forth strategies to address bias in hiring algorithms. One research recommended employing reweighting methods to modify the distribution of training data with the goal of lessening bias, towards minority groups.
   - **Critique & Improvement Proposed**: Although effective, these methods could be enhanced by integrating adversarial debiasing techniques, which adapt the model dynamically during training to reduce bias. I’ll implement adversarial debiasing as best I can and compute how effective it is at mitigating bias.

[2] The research paper emphasized the significance of auditing black-box models to uncover biases. Suggested utilizing models such, as decision trees to simulate the functions of intricate models.
   - **Critique & Improvement Proposed**: An area for improvement could involve creating models that merge the interpretability of decision trees, with the predictive capabilities of neural networks enabling more effective detection and reduction of biases. I have already trained a Random Forest (Decision Tree) and a Neural Network on the dataset. I can combine both models, possibly through Voting Ensemble or Stacking methods, and evaluate their performance on the dataset.

[3] Another research emphasized how adjusting hyperparameters can influence the fairness of a model. The scholars suggested incorporating fairness-aware hyperparameter optimization to find model settings that achieve a balance, between accuracy and fairness.
   - **Critique & Improvement Proposed**: While adequate, this approach can be further enhanced by integrating fairness constraints into the optimization process directly, guaranteeing that fairness is a consideration, throughout all stages of model development. I’ll use the SensitiveNets approach which introduces regularization loss that penalizes the model for being too correlated with sensitive attributes

## Methodology:

### 1. Data Collection and Preprocessing:
   - Collected and preprocessed a diverse dataset (FairCVtest) to ensure balanced representation of different demographic groups.
   - Removed personally identifiable information and irrelevant features from the dataset.

### 2. Model Training:
   - Trained the hiring tool: A Neural Network with Bi-directional LSTM and FastText for text embeddings.
   - Trained the neural network on the hiring tool and used it to generate CV embeddings from the dataset.
   - Trained three more models on the CV embeddings derived from the hybrid model: SVM (Support Vector Machine), Random Forest Classifier, and a traditional Neural Network.

### 3. Bias Detection:
   - Evaluated the performance of the hiring tool using accuracy metrics such as MAE and fairness metrics such as Top Scores, Demographic Parity, and Equality of Opportunity.
   - Identified the gender and ethnicity biases present in the hiring tool.
   - Evaluated the performance of the SVM, RF, and NN models (trained on the CV embeddings) for gender, ethnicity, and neutral labels.
   - Visualized all the detected biases with Dist, KDE, Hist, and t-SNE plots.

### 4. Bias Mitigation:
   - Explored bias mitigation techniques such as Adversarial Debiasing, Creating a Hybrid Model (combining Decision Trees and Neural Network), and the SensitiveNets approach.
   - Retrained the models and reevaluated their performance after implementing the bias mitigation techniques.

### 5. Visualization and Reporting:
   - Created visualizations to illustrate the impact of bias mitigation techniques.
   - Documented the findings.

## References:
[1] Zafar, M. B., Valera, I., Gomez Rodriguez, M., & Gummadi, K. P. (2017). Fairness Beyond Disparate Treatment & Disparate Impact: Learning Classification without Disparate Mistreatment. Proceedings of the 26th International Conference on World Wide Web, 1171-1180.

[2] Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence, 1(5), 206-215.

[3] Kearns, M., Neel, S., Roth, A., & Wu, Z. S. (2019). An Empirical Study of Rich Subgroup Fairness for Machine Learning. Proceedings of the 2019 Conference on Fairness, Accountability, and Transparency, 100-109.

## Instructions to run the project locally:
   - Create a conda env with python v3.9 using `conda create --name myenv python=3.9`
   - Install all the libraries from the provided requirements file using `pip install -r requirements.txt`
   - Download the word embeddings file here: https://www.kaggle.com/datasets/yekenot/fasttext-crawl-300d-2m. Add it to the same path the dataset is in (The word embeddings file was too big to be uploaded onto the repo even with lfs (4.2GB))
   - Make sure to change the global path variables for all the necessary files inside the notebook
   - Time to run depends on your GPU. Highly recommended to configure tensorflow to use GPU for training.
