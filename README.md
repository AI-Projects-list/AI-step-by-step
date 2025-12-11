# AI Step-by-Step Learning Guide

A comprehensive collection of interactive Jupyter notebooks covering essential topics in AI, Machine Learning, Data Science, and Big Data technologies.

## 📚 Available Guides

### 1. **Machine Learning - All Models Guide** 
📄 [ml_all_models_guide.ipynb](./ml_all_models_guide.ipynb)
- **Topics**: 45 ML models across 7 categories
- **Covers**: Regression, Classification, Clustering, Dimensionality Reduction, Ensemble Methods, Neural Networks, Anomaly Detection, Time Series, Advanced ML
- **Status**: Topics 1-2 fully implemented (Linear Regression, Polynomial Regression)

### 2. **Data Science Complete Guide** 
📄 [data_science_complete_guide.ipynb](./data_science_complete_guide.ipynb)
- **Topics**: 50 comprehensive topics across 9 categories
- **Covers**: Data Collection, EDA, Cleaning, Transformation, Visualization, Statistics, Feature Engineering, Model Evaluation, Advanced Topics
- **Status**: Topics 1-2 fully implemented (Reading CSV, Reading Excel)

### 3. **Python Complete Guide** 
📄 [python_complete_guide.ipynb](./python_complete_guide.ipynb)
- **Topics**: 60 Python topics across 10 categories
- **Covers**: Basics, Control Flow, Data Structures, Functions, OOP, File Handling, Error Handling, Modules, Advanced Features
- **Status**: Topics 1-2 fully implemented (Variables & Data Types, Operators)

### 4. **Hadoop PySpark Complete Guide** 
📄 [hadoop_pyspark_complete_guide.ipynb](./hadoop_pyspark_complete_guide.ipynb)
- **Topics**: 65 PySpark topics across 10 categories
- **Covers**: Spark Basics, DataFrames, Transformations, Joins, SQL, Data Cleaning, Performance Optimization, MLlib, Streaming, Advanced Topics
- **Status**: Topics 1-2 fully implemented (SparkSession, SparkContext)

### 5. **PyTorch Complete Guide** 
📄 [pytorch_complete_guide.ipynb](./pytorch_complete_guide.ipynb)
- **Topics**: 65 PyTorch topics across 10 categories
- **Covers**: PyTorch Fundamentals, Neural Networks, Data Handling, CNNs, RNNs, Advanced Architectures, Training Techniques, Model Optimization, Advanced Topics, Deployment
- **Status**: Topics 1-2 fully implemented (Tensors, Tensor Operations)

### 6. **Machine Learning - Detailed Models Guide** 
📄 [ml_all_models_detailed_guide.ipynb](./ml_all_models_detailed_guide.ipynb)
- **Topics**: 75 ML models across 14 categories
- **Covers**: Linear Models, Decision Trees, Boosting, SVM, Nearest Neighbors, Naive Bayes, Clustering, Dimensionality Reduction, Ensemble Methods, Neural Networks, Anomaly Detection, Time Series, Deep Learning, Advanced ML
- **Status**: Topics 1-2 fully implemented (Linear Regression, Ridge Regression)

### 7. **Context Engineering Complete Guide** 
📄 [context_engineering_complete_guide.ipynb](./context_engineering_complete_guide.ipynb)
- **Topics**: 75 prompt engineering techniques across 13 categories
- **Covers**: Fundamentals, Prompting Techniques, Advanced Strategies, Output Formatting, Context Optimization, Multi-Turn Conversation, RAG, Task-Specific, Safety & Guardrails, Context Enhancement, Meta-Prompting, Evaluation, Advanced Techniques
- **Status**: Topics 1-2 fully implemented (What is Context Engineering, Basic Prompt Structure)

### 8. **NLP Complete Guide** 
📄 [nlp_complete_guide.ipynb](./nlp_complete_guide.ipynb)
- **Topics**: 80 NLP techniques across 15 categories
- **Covers**: Text Preprocessing, Text Representation, Word Embeddings, Advanced Embeddings/Transformers, Text Classification, Sentiment Analysis, NER, POS Tagging, Text Similarity, Topic Modeling, Text Generation, Q&A, Summarization, Translation, Advanced Tasks
- **Status**: Topics 1-2 fully implemented (Tokenization, Lowercasing & Case Normalization)

### 9. **Model Context Protocol (MCP) Complete Guide** 
📄 [mcp_complete_guide.ipynb](./mcp_complete_guide.ipynb)
- **Topics**: 80 MCP concepts across 14 categories
- **Covers**: MCP Fundamentals, Core Components, Resources, Tools, Prompts, Server Implementation, Client Implementation, Advanced Features, Real-World Servers, Integration Patterns, Security & Best Practices, Testing & Debugging, Deployment, Advanced Use Cases
- **Status**: Topics 1-2 fully implemented (What is MCP, MCP Architecture Overview)

## 🎯 Learning Approach

Each topic includes:
- **📖 What**: Clear concept definition and explanation
- **🎯 Why**: Advantages and benefits
- **⏱️ When to Use**: Specific scenarios with real-world examples
- **❌ When NOT to Use**: Limitations and better alternatives
- **📊 How It Works**: Technical details and mechanisms
- **💻 Complete Script Example**: Working code with 8-11 sections
- **🌍 Real-World Applications**: Industry use cases
- **💡 Key Insights**: Best practices and tips

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/AI-Projects-list/AI-step-by-step.git
cd AI-step-by-step
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch Jupyter:
```bash
jupyter notebook
```

## 📖 Usage

Each notebook is designed for:
- **Self-paced learning**: Start from any topic
- **Reference guide**: Quick lookup for when/why to use specific techniques
- **Hands-on practice**: Run code cells directly
- **Decision-making**: Learn when to use what technique

## 🛠️ Technologies Covered

- **Python**: Core programming fundamentals
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine Learning models
- **PyTorch**: Deep Learning framework
- **PySpark**: Big Data processing
- **Matplotlib/Seaborn**: Data visualization
- **NLP**: NLTK, spaCy, Transformers (BERT, GPT)
- **LLM Integration**: Model Context Protocol (MCP)
- **Prompt Engineering**: Context engineering techniques

## 📊 Project Structure

```
AI_step_by_step/
├── ml_all_models_guide.ipynb               # 45 ML models
├── ml_all_models_detailed_guide.ipynb      # 75 ML models (detailed)
├── data_science_complete_guide.ipynb       # 50 Data Science topics
├── python_complete_guide.ipynb             # 60 Python topics
├── hadoop_pyspark_complete_guide.ipynb     # 65 PySpark topics
├── pytorch_complete_guide.ipynb            # 65 PyTorch topics
├── context_engineering_complete_guide.ipynb # 75 Context Engineering techniques
├── nlp_complete_guide.ipynb                # 80 NLP techniques
├── mcp_complete_guide.ipynb                # 80 MCP concepts
├── data_cleaning_learning.ipynb            # Data cleaning examples
├── data_wrangling_learning.ipynb           # Data wrangling examples
├── dsa_learning.ipynb                      # Data structures & algorithms
├── eda_learning.ipynb                      # Exploratory data analysis
├── javascript_learning.ipynb               # JavaScript basics
├── typescript_learning.ipynb               # TypeScript basics
├── sentiment_analysis.py                   # NLP sentiment analysis
├── text_summarization.py                   # NLP text summarization
├── pyproject.toml                          # Project configuration
└── README.md                               # This file
```

## 🎓 Learning Path

### Beginner
1. Start with **Python Complete Guide** (basics)
2. Move to **Data Science Guide** (data handling)
3. Learn **NLP Complete Guide** (text processing basics)
4. Explore **Machine Learning Guide** (simple models)

### Intermediate
1. **Context Engineering Guide** (prompt engineering for LLMs)
2. **PyTorch Guide** (deep learning)
3. **PySpark Guide** (big data)
4. **ML Detailed Guide** (advanced model understanding)
5. Advanced topics in each guide

### Advanced
1. **Model Context Protocol (MCP)** (AI integration architecture)
2. Custom implementations
3. Production deployment
4. Performance optimization
5. Distributed training
6. Agentic AI workflows

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**budis**

## 🌟 Features

✅ Comprehensive coverage of AI/ML topics  
✅ Decision-making framework (when/why)  
✅ Complete working code examples  
✅ Real-world applications  
✅ Best practices and insights  
✅ Self-contained topics  
✅ Interactive Jupyter notebooks  
✅ Production-ready patterns

## 📚 Additional Resources

- [Official Python Documentation](https://docs.python.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Learning! 🚀**
