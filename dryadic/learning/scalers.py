
from sklearn.preprocessing import scale, FunctionTransformer

def center_scaling(X):
    return scale(X.T).T

center_scale = FunctionTransformer(center_scaling, validate=True)

