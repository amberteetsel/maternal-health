#### HELPER FUNCTION TO DISPLAY PRINCIPAL COMPONENT ANALYSIS RESULTS

# Dependencies
import os
import inspect

# Function

# ==========================================================
# ACTUAL INPUTS
# ==========================================================

overview_pca = """
    ### The Curse of Dimensionality

    The Curse of Dimensionality refers to various challenges and complications that arise when analyzing and organizing
    data in high-dimensional spaces. Dimensions refer to the features or attributes of data; in the context of this study,
    dimensions of the Health Rankings dataset include measures of poverty, unemployment, preventative care, maternal
    mortality, and more. As the number of features grows, the volume of the data space increases exponentially and
    available data observations become sparse. In high-dimensional spaces traditional analytical models break down because
    geometric distances (e.g. Euclidean distance) are less distinct, meaning the distance between two highly
    dissimilar data points and the distance between two highly similar points look mathematically identical. Humans
    cannot visualize beyond three dimensions (3D) so it's also difficult for analysts to conceive of and understand
    high-dimensional feature spaces.

    ### Dimensionality Reduction

    Dimensionality reduction is the process of compressing high-dimensional feature spaces into a lower-dimensional
    subspace, typically 2D or 3D, while retaining as much structural variation as possible. Dimensionality reduction
    is crucial to address the problems caused by high dimensionality:

    * Data sparsity
    * Increased computation
    * Multicollinearity
    * Overfitting
    * Performance degradation
    * Visualization challenges

    Reducing dimensionality can eliminate multicollinearity and background noise that destabilizes machine learning
    algorithms. By compressing the data, it drastically improves computational efficiency while solving the issues of data
    sparsity and data visualization by turning a multi-variable matrix into a clear, interpretable landscape in two or
    three dimensions.

    ### Principal Component Analysis

    Principal Component Analysis (PCA) is an unsupervised linear transformation technique for dimensionality reduction
    that projects data onto brand-new, uncorrelated axes known as *Principal Components*. Rather than selecting or
    deleting specific raw variables, PCA uses all original features to calculate a completely new coordinate system.

    * **Principal Component 1 (PC1):** The first geometric axis, built to point in the direction of the absolute maximum
    variance in the data.
    * **Principal Component 2 (PC2):** The second geometric axis, constructed to be completely orthogonal to PC1 and 
    capture the highest remaining variance. 

    In this way, the first few components generated should contain the vast majority of a dataset's information (variance),
    allowing analysts to discard additional components wtih minimal information loss.

    ### Eigenvalues and Eigenvectors

    In practice, PCA is based on the decomposition of a dataset's *covariance matrix*, which tracks how each feature
    moves in relation to every other feature. PCA uses linear algebra to extract critical information from this matrix:
    **Eigenvectors** and **Eigenvalues**.

    * **Eigenvectors** (The Direction): An eigenvector is a non-zero vector whose direction remains entirely unchanged
    when a linear transformation is applied. In PCA, eigenvectors represent the directions of the new principal component
    axes and are weighted/combined to build the new coordinate space.
    * **Eigenvalues** (The Magnitude): An eigenvalue is a scalar value that corresponds to a specific eigenvector. It
    measures the absolute magnitude of variance captured along that specific principal component axis. The principal
    component with the largest eigenvalue represents the axis of maximum variance, or PC1.
"""