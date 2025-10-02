import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

class SimpleSVM:
    def __init__(self, C=1.0, degree=3, gamma=None, coef0=1.0, tol=1e-3, max_passes=5, max_iter=10000, seed=0):
        self.C = C
        self.degree = degree
        self.gamma = gamma  # if None, uses 1 / n_features
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)
        # learned params
        self.alphas = None
        self.b = 0.0
        self.X = None
        self.y = None
        self.K = None  # kernel matrix

    def _poly_kernel(self, X, Z):
        if self.gamma is None:  
            gamma = 1.0 / X.shape[1]
        else:
            gamma = self.gamma
        return (gamma * (X @ Z.T) + self.coef0) ** self.degree

    def fit(self, X, y):
        """
        Train binary SVM with labels y in {-1, +1} using simplified Platt SMO.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        assert set(np.unique(y)) <= {-1.0, 1.0}, "y must be in {-1, +1}"

        n_samples, n_features = X.shape
        self.X = X
        self.y = y
        self.alphas = np.zeros(n_samples)
        self.b = 0.0

        # Precompute full kernel matrix
        self.K = self._poly_kernel(X, X)

        passes = 0
        iters = 0

        def f(i):
            # decision value at training point i
            return np.sum(self.alphas * y * self.K[:, i]) + self.b

        while passes < self.max_passes and iters < self.max_iter:
            num_changed_alphas = 0
            for i in range(n_samples):
                Ei = f(i) - y[i]
                # KKT violation check
                if ((y[i]*Ei < -self.tol and self.alphas[i] < self.C) or
                    (y[i]*Ei > self.tol and self.alphas[i] > 0)):
                    # pick j != i
                    j = i
                    while j     == i:
                        j = self.rng.integers(0, n_samples)
                    Ej = f(j) - y[j]

                    ai_old = self.alphas[i]
                    aj_old = self.alphas[j]

                    # Compute L and H (box constraints)
                    if y[i] != y[j]:
                        L = max(0.0, aj_old - ai_old)
                        H = min(self.C, self.C + aj_old - ai_old)
                    else:
                        L = max(0.0, ai_old + aj_old - self.C)
                        H = min(self.C, ai_old + aj_old)
                    if L == H:
                        continue

                    # Compute eta (second derivative along the line)
                    eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        # Non-positive definite along direction, skip (rare with poly/RBF)
                        continue

                    # Update alpha_j
                    self.alphas[j] = aj_old - (y[j] * (Ei - Ej)) / eta
                    # Clip to [L, H]
                    if self.alphas[j] > H:
                        self.alphas[j] = H
                    elif self.alphas[j] < L:
                        self.alphas[j] = L

                    if abs(self.alphas[j] - aj_old) < 1e-6:
                        # No significant change
                        self.alphas[j] = aj_old
                        continue

                    # Update alpha_i (in tandem)
                    self.alphas[i] = ai_old + y[i]*y[j]*(aj_old - self.alphas[j])

                    # Compute b1, b2 and update b
                    b1 = (self.b - Ei
                          - y[i]*(self.alphas[i] - ai_old)*self.K[i, i]
                          - y[j]*(self.alphas[j] - aj_old)*self.K[i, j])
                    b2 = (self.b - Ej
                          - y[i]*(self.alphas[i] - ai_old)*self.K[i, j]
                          - y[j]*(self.alphas[j] - aj_old)*self.K[j, j])

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = 0.5 * (b1 + b2)

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
            iters += 1

        return self

    def project(self, X):
        """
        Compute decision function f(x) for arbitrary X.
        """
        X = np.asarray(X, dtype=float)
        Kx = self._poly_kernel(self.X, X)  # shape (n_train, n_test)
        return (self.alphas * self.y) @ Kx + self.b

    def predict(self, X_test):
        return np.sign(self.project(X_test))

    def support_vectors_(self): 
        return self.X[self.alphas > 1e-8], self.y[self.alphas > 1e-8], self.alphas[self.alphas > 1e-8]



print ("Train on Iris")

# Load iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]   # take first two features for visualization (sepal length, sepal width)
y = iris.target

# Convert to binary classification: setosa (0) vs. non-setosa (1,2)
y_binary = np.where(y == 0,  1, -1)  # +1 for setosa, -1 for others

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary)
# Feature scaling (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train custom SVM
svm = SimpleSVM(C=1.0, degree=3, coef0=1.0, tol=1e-3, max_passes=10, max_iter=1000, seed=42)
svm.fit(X_train, y_train)

# Predictions
y_pred = svm.predict(X_test)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print("Test Accuracy:", accuracy)
print("Test F1 Score:", f1_score(y_test, y_pred))

# Show support vectors
X_sv, y_sv, alpha_sv = svm.support_vectors_()
print("Number of support vectors:", len(X_sv))
plt.figure(figsize=(6,5))
plt.scatter(X_test[y_test==1,0], X_test[y_test==1,1], c='b', label='Setosa (+1)')
plt.scatter(X_test[y_test==-1,0], X_test[y_test==-1,1], c='r', label='Non-setosa (-1)')
plt.scatter(X_sv[:,0], X_sv[:,1], s=80, facecolors='none', edgecolors='k', linewidths=1.5, label='Support Vectors')

# Decision boundary
xmin, xmax = X_test[:,0].min()-0.5, X_test[:,0].max()+0.5
ymin, ymax = X_test[:,1].min()-0.5, X_test[:,1].max()+0.5
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
zz = svm.project(grid).reshape(xx.shape)
CS = plt.contour(xx, yy, zz, levels=[0.0], colors='k', linewidths=2)
plt.clabel(CS, inline=1, fontsize=8)

plt.xlabel("Sepal length (scaled)")
plt.ylabel("Sepal width (scaled)")
plt.title("SVM Decision Boundary on Iris (Setosa vs Non-setosa)")
plt.legend(loc="best")
plt.tight_layout()
plt.show()



   

print("Train on Breast Cancer Wisconsin")

# Load breast cancer dataset
bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target

# Binary classification: malignant (0) vs benign (1)
# We'll use +1 for  
y_binary = np.where(y == 1, 1, -1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
svm = SimpleSVM(C=1.0, degree=3, coef0=1.0, tol=1e-3, max_passes=10, max_iter=2000, seed=42)
svm.fit(X_train, y_train)

# Predictions
y_pred = svm.predict(X_test)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print("Test Accuracy (Breast Cancer Benign vs Malignant):", accuracy)
print("Test F1 Score:", f1_score(y_test, y_pred))

# Number of support vectors
X_sv, y_sv, alpha_sv = svm.support_vectors_()
print("Number of support vectors (train):", len(X_sv))
# Draw decision boundary for first two features (for visualization)
if X_train.shape[1] >= 2:
    plt.figure(figsize=(6,5))
    # Only plot using first two features
    X_plot = X_test[:, :2]
    y_plot = y_test
    X_sv_plot = X_sv[:, :2]
    plt.scatter(X_plot[y_plot==1,0], X_plot[y_plot==1,1], c='b', label='Benign (+1)')
    plt.scatter(X_plot[y_plot==-1,0], X_plot[y_plot==-1,1], c='r', label='Malignant (-1)')
    plt.scatter(X_sv_plot[:,0], X_sv_plot[:,1], s=80, facecolors='none', edgecolors='k', linewidths=1.5, label='Support Vectors')
    # Decision boundary
    xmin, xmax = X_plot[:,0].min()-0.5, X_plot[:,0].max()+0.5
    ymin, ymax = X_plot[:,1].min()-0.5, X_plot[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Pad grid to full feature size for projection
    grid_full = np.zeros((grid.shape[0], X_test.shape[1]))
    grid_full[:, :2] = grid
    zz = svm.project(grid_full).reshape(xx.shape)
    CS = plt.contour(xx, yy, zz, levels=[0.0], colors='k', linewidths=2)
    plt.clabel(CS, inline=1, fontsize=8)
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.title("SVM Decision Boundary on Breast Cancer (2 features)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()






# --- Grid search for C (from scratch) with train/val/test split ---
# Split: 60% train, 10% val, 30% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.142857, random_state=42, stratify=y_temp)  # 0.142857 ≈ 10/70

# Feature scaling (fit on train, apply to val/test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

C_values = [ 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]
best_acc = 0
best_C = None

for C in C_values:
    svm = SimpleSVM(C=C, degree=3, coef0=1.0, tol=1e-3, max_passes=10, max_iter=2000, seed=42)
    svm.fit(X_train, y_train)
    y_pred_val = svm.predict(X_val)
    acc = np.mean(y_pred_val == y_val)
    print(f"C={C}: Validation Accuracy={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_C = C

print(f"Best C: {best_C} with Validation Accuracy: {best_acc:.4f}")

# Retrain on train+val, test on test set
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.hstack([y_train, y_val])
svm = SimpleSVM(C=best_C, degree=3, coef0=1.0, tol=1e-3, max_passes=10, max_iter=2000, seed=42)
svm.fit(X_trainval, y_trainval)
y_pred_test = svm.predict(X_test)
test_acc = np.mean(y_pred_test == y_test)
print(f"Test Accuracy (with best C): {test_acc:.4f}")
print("Test F1 Score:", f1_score(y_test, y_pred_test))


