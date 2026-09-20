# k-Nearest Neighbour

---
#### TODO
1. Verify/quantify/visualize: "1-NN has zero training error and tends to overfit noise".

---

### Roadmap
| Part | Topic                                                    | Coding-Style |
|------|----------------------------------------------------------|--------------|
| 1    | 1-NN, Voronoi, k-NN classification/regression, Minkowski | From-scratch |



 k-Nearest Neighbour (k-NN) is a **lazy, instance-based, non-parametric** learner.
 - **Lazy**: there is no training phase. "Fitting" just stores the dataset.
 - **Instance-based**: predictions come directly from stored examples, not from a learned parametric model.
 - **Non-parametric**: model complexity grows with the data (there is no fixed weight vector).

Given a query point x, the algorithm:
1. Computes the distance $d(x, x_{i})$ to every training point $x_{i}$.
2. Selects the k-closest, forming the neighbourhood $N_{k}(x)$.
3. Predicts:
   - **Classification**: majority vote, $\hat{y} = argmax_{c} \sum_{i \in N_{k}(x)} 1[y_{i} = c]$
   - **Regression**: average, $\hat{y} = \frac{\sum_{i \in N_{k}(x)} y_{i}}{k}$

The 1-NN decision boundary is the **Voronoi tesselation** of the training set, which is why 1-NN has zero training error and tends to overfit noise.
Increasing $k$ smooth the boundary.


#### Distance metrics
The Minkowski family: $d_{p}(x, z) = (\sum_{j} |x_j - z_j|^p)^{1/p}$, where $j \in [1, D]$ indexes the features of size $D$.
- $p = 1$: Manhattan
- $p = 2$: Euclidean
- $\lim_{p \to \infty}$: Chebyshev
- $\lim_{p \to 0^+}$: Hamming

The reason why the Minkowski family is used is that it generalizes many distance metrics: different values of $p$ can capture different notions of distance.

> The metric is the real "model" in k-NN.
> Different metrics encode different notions of similarity.

---
## Code Lab 1
Write k-NN from scratch and validate it against scikit-learn on Iris and visualize how $k$ changes the decision boundary.

#### Notes on implementation
- **numpy.argpartition**: returns array of indices that would partition an array into two parts: the k smallest elements and the rest. This is useful for efficiently finding the k-nearest neighbors without fully sorting the distances.
- **Counter.most_common(n)**: returns a list of the n most common elements and their counts; e.g., `Counter([1, 2, 2, 3]).most_common(1)` returns `[(2, 2)]`, which is useful for finding the majority class among the k-nearest neighbors.

#### Limitations and issues of the approach
- The sum in the distance metric treats every feature equally, so a feature with a large numerical range contributes much more to the total than the other **unless you standardize**.

---

# References
- [NumPy - numpy.argpartition](https://numpy.org/devdocs/reference/generated/numpy.argpartition.html)
- [Python Collections - Counter](https://docs.python.org/3/library/collections.html#collections.Counter)
- [Scikit-learn - train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- 