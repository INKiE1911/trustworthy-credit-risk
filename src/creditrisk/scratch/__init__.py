"""Classic algorithms written from scratch in NumPy (Step 8), each checked against scikit-learn.

    logreg.py       logistic regression (L2 = MAP) by Newton's method, and its Bayesian
                    (Laplace) version that gives an uncertainty for every applicant
    naive_bayes.py  Gaussian Naive Bayes
    pca.py          PCA through the SVD
    lda.py          Fisher's linear discriminant (projection) and the LDA classifier
    kmeans.py       K-means with k-means++ seeding
    gmm.py          Gaussian mixture fitted with EM
    mlp.py          a 1-hidden-layer neural network with backprop and a gradient check

Only NumPy does the maths. scikit-learn's BaseEstimator is used just for the familiar
fit / predict_proba interface, so these models also work inside pipelines and the CV runner.
"""
