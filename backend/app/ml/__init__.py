"""Classical (non-deep) machine-learning components.

Everything in this package is trained with scikit-learn on hand-engineered
features - no neural networks, no GPU, no pretrained weights:

- symbol recognition: HOG + zoning features -> SVM / Random Forest / kNN /
  Logistic Regression (app.ml.symbol_classifier)
- mistake diagnosis: SymPy-derived features -> Random Forest / Gradient
  Boosting / Decision Tree / Logistic Regression (app.ml.mistake_classifier)

Train both models with:  python -m app.ml.train
"""
