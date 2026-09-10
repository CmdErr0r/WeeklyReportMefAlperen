import numpy as np
import sympy as sp

np.random.seed(142)

def return_preprocessed_data():
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from pandas import read_csv

    # Extract labels
    df = read_csv('california_housing.csv')
    X = df.drop(columns=["MedHouseVal"]).values
    y = df["MedHouseVal"].values

    # Split the data for training and testing
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing
    lowerbound = np.percentile(Xtr, 0, axis=0)
    upperbound = np.percentile(Xtr, 99, axis=0)

    Xtr = np.clip(Xtr, lowerbound, upperbound)
    Xte = np.clip(Xte, lowerbound, upperbound)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    
    from IPython.display import display, Image
    corr_img_file = "./dataset_shift_report.png"
    display(Image(filename=corr_img_file))
    
    return Xtr.T, ytr, Xte.T, yte

def train_model(Xtr, ytr):
    import torch
    import torch.nn as nn
    torch.manual_seed(42)

    # Transform to the Torch Tensors
    Xtr, ytr = torch.tensor(Xtr.T, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)

    class BlackBox(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8,32),
                nn.ReLU(),
                nn.Linear(32,16),
                nn.ReLU(),
                nn.Linear(16,1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = BlackBox()
    opt   = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        model.train()

        opt.zero_grad()
        
        loss = ((model(Xtr) - ytr)**2).mean()
        loss.backward()
        
        opt.step()

        print(f"epoch:{epoch+1:3d} loss:{loss:.4f}", end="\r", flush=True)

    print(f"Extracting the model's function...", end="\r", flush=True)

    X_sym = sp.Matrix([sp.symbols(f'x{i}') for i in range(1,9)])

    layers = list(model.net.children())
    W1 = sp.Matrix(layers[0].weight.detach().numpy())
    b1 = sp.Matrix(layers[0].bias.detach().numpy())
    
    W2 = sp.Matrix(layers[2].weight.detach().numpy())
    b2 = sp.Matrix(layers[2].bias.detach().numpy())

    W3 = sp.Matrix(layers[4].weight.detach().numpy())
    b3 = sp.Matrix(layers[4].bias.detach().numpy())

    net = (W1 @ X_sym) + b1 # (32,8) @ (8,1) + (32,) = (32,1)
    net = sp.Matrix([sp.Max(val,0) for val in net])

    net = (W2 @ net) + b2
    net = sp.Matrix([sp.Max(val,0) for val in net])

    function_sym = ((W3 @ net) + b3)[0]

    function = sp.lambdify((X_sym), function_sym, modules="numpy")

    # check = np.max(function(*Xte.numpy().T) - model(Xte).detach().numpy())
    # print(f"Maximum different between model(Xte) and function(Xte) is: {check:.6f}")
    
    return function, function_sym


def function_decompose(f, X, a, b, w_type = None):    
    n_features = len(X)
    anti = lambda idx: np.setdiff1d(np.arange(n_features), idx)

    if w_type == "normal":
        # Note: If using normal, make sure a and b are set wide (e.g., -5 to 5)
        w = np.array([(1 / sp.sqrt(2 * sp.pi)) * sp.exp(-X[i]**2 / 2) for i in range(n_features)])
    elif w_type == "parabolic":
        w = np.array([(6 * (X[i] - a[i]) * (b[i] - X[i])) / ((b[i] - a[i])**3) for i in range(n_features)])
    elif w_type == "slanted":
        w = np.array([(2 * (X[i] - a[i])) / ((b[i] - a[i])**2) for i in range(n_features)])
    else:
        # Default: Uniform
        w = 1 / (b - a)

    f_mean = sp.integrate(f * w.prod(), *np.c_[X,a,b])

    fi = np.array([
        sp.integrate(f * w[anti([i])].prod(), *np.c_[X,a,b][anti([i])]) - f_mean
        for i in range(n_features)
    ])

    fij =  np.array([
        sp.integrate(f * w[anti([i,j])].prod(), *np.c_[X,a,b][anti([i,j])])
        - fi[i] - fi[j] - f_mean
        for i in range(n_features)
        for j in range(i+1, n_features)
    ])

    D = sp.integrate(f**2 * w.prod(), *np.c_[X,a,b]) - f_mean**2

    Di = np.array([
        sp.integrate(fi[i]**2 * w[[i]].prod(), *np.c_[X,a,b][[i]])
        for i in range(n_features)
    ]) / D

    get_fij = lambda i, j: int(i * n_features - (i * (i + 1)) // 2 + (j - i - 1))
    Dij = np.array([
        sp.integrate(fij[get_fij(i,j)]**2 * w[[i,j]].prod(), *np.c_[X,a,b][[i,j]])
        for i in range(n_features)
        for j in range(i+1, n_features)
    ]) / D

    return f_mean, fi, fij, D, Di, Dij

