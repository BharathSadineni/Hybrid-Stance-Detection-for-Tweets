# Stance Detection System

This repository hosts the code for a hybrid stance detection system that preprocesses data, vectorizes text, generates features, and evaluates the accuracy of stance predictions. It is designed to facilitate stance detection on tweets, allowing users to analyze content from either a predefined URL or a custom URL provided by the user.

## Getting Started

This section provides instructions to help you set up and run the project on your local machine for development and testing purposes.

### Prerequisites

Ensure you have Python installed on your system along with the following libraries, which are necessary for running the project:

### Installation

You can install all the required dependencies with the following command:

```bash
pip install emoji requests tenacity pandas numpy nltk joblib tqdm scikit-learn xgboost jmespath bs4 google
```

After installing the required libraries, clone the repository to your local machine using:

```bash
git clone [repository URL]
cd [repository directory]
```

### Usage

Execute the following command to start the stance detection system:

```bash
python coordinator.py
```

### What to Expect

#### Initial Run:
- The first execution may take up to 500 seconds as it needs to preprocess data, vectorize text, and generate features.
- The system will then calculate and display the accuracy of stance predictions on test data.
- You will be prompted to choose between running the detection on a default URL or entering a new one. Type 'yes' to use your own URL and follow the on-screen instructions to input it.
- The detected stance and its accuracy percentage will be displayed once the analysis completes.

#### Subsequent Runs:
- Any further uses of the `coordinator.py` script will be significantly quicker (about 100-200 seconds), as it only needs to vectorize new data.
- You will be prompted again to specify a URL for stance detection. Enter the desired URL when asked to proceed with the stance analysis.

