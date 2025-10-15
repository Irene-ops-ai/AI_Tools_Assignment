# AI_Tools_Assignment
## Part 1: Theoretical Understanding
### 1. Short Answer Questions

#### Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?
- TensorFlow and PyTorch are both popular deep learning frameworks, but they differ in how they execute operations and where they are most useful.

- TensorFlow is developed by Google. It uses static computation graphs, which make it highly efficient and suitable for deployment in production (e.g., mobile or web apps).

- PyTorch, developed by Meta (Facebook), uses dynamic computation graphs, which are easier to debug and modify during training.

  I'd choose PyTorch for research and fast prototyping and TensorFlow for large-scale production and deployment.

#### Q2. Two use cases for Jupyter Notebooks in AI
  1. Model Experimentation: Jupyter allows developers to write and test small parts of code interactively, visualize results, and tune models efficiently.
  2. Documentation and Reporting: You can mix code, visualizations, and markdown notes in one file — perfect for presentations or collaborative work.
 
#### Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?
 spaCy goes beyond basic Python string operations by providing contextual language understanding.
 While basic string functions only split or search for words, spaCy can:
- Detect named entities (like brands or people).
- Perform part-of-speech tagging and dependency parsing.
This allows for more accurate and meaningful text analysis.

### 2. Comparative Analysis

| Feature                | Scikit-learn                                          | TensorFlow                                        |
| ---------------------- | ----------------------------------------------------- | ------------------------------------------------- |
| **Target Application** | Classical ML (e.g., decision trees, SVMs, regression) | Deep learning (e.g., CNNs, RNNs, neural networks) |
| **Ease of Use**        | Easier for beginners with simple syntax               | More complex setup but flexible                   |
| **Community Support**  | Strong academic & ML community                        | Very large global deep learning community         |

## Part 3: Ethics & Optimization (10%)
### 1. Ethical Considerations
- The MNIST model might underperform on handwriting styles from underrepresented groups.
- Amazon reviews may reflect biased language (e.g., gendered or cultural bias).
To reduce bias:
- Use TensorFlow Fairness Indicators to measure and visualize fairness in predictions.
- Use spaCy’s rule-based filters to control for biased words and ensure consistent labeling.

### 2. Troubleshooting Challenge
When debugging TensorFlow models:
- Check input shapes (must match model input).
- Ensure correct loss function (e.g., sparse_categorical_crossentropy for integer labels).
- Confirm data normalization (values between 0–1).



