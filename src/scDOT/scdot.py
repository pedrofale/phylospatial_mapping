import numpy as np
import torch
import torch.nn as nn
import ot

try:
    from .ddn import OptimalTransportLayer
except Exception:
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    try:
        from ddn import OptimalTransportLayer
    except Exception as e:
        raise ImportError("Could not import OptimalTransportLayer from ddn.py. Ensure `ddn.py` is in the same package or on PYTHONPATH.") from e
from deconv import nls_projgrad  # Added this import to use nls_projgrad which is used here in the original code

class scDOT(nn.Module):
    def __init__(self, sc_adata, st_adata):
        super(scDOT, self).__init__()
        self.NNLS = NNLS(sc_adata, st_adata)
        self.OT = OptimalTransportLayer(method='approx')
        self.M = nn.Parameter(torch.from_numpy(
            ot.dist(st_adata.X, sc_adata.X, metric='cosine'))
        )

    def forward(self, sc_adata, st_adata):
        NNLS_output = self.NNLS(sc_adata, st_adata)
        OT_output = self.OT(self.M)
        OT_output = OT_output/OT_output.sum(0) # col (cell) sum to 1
        return NNLS_output, OT_output

class NNLS(torch.nn.Module):
    def __init__(self, sc_adata, st_adata):
        super(NNLS, self).__init__()
        markers = st_adata.uns['markers']  #.to_numpy()
        st = st_adata.X
        self.W = nn.Parameter(torch.randn(st.shape[0], markers.shape[0]))

    def forward(self, sc_adata, st_adata):
        markers = st_adata.uns['markers']  #.to_numpy()
        st = st_adata.X
        W_nnls, _ = nls_projgrad(markers.T, st.T)
        W_nnls = W_nnls.T
        W_nnls = W_nnls/W_nnls.sum(1)[:,None]
        self.W.data.copy_(torch.tensor(W_nnls).float())
        return self.W


def train_scdot(sc_adata, st_adata, lr=1.0e-1, iters=10):
    # The code in this function is copied from the scDOT readme.md (https://github.com/namtk/scDOT/blob/main/README.md)
    model = scDOT(sc_adata, st_adata)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for i in range(iters):
        # Forward pass
        NNLS_output, OT_output = model(sc_adata, st_adata)

        # ct = torch.tensor(sc_adata.obsm['cell_type'].values.T)
        ct = torch.tensor(sc_adata.obsm['cell_type'].T)
        P_true = NNLS_output @ ct.float()  # spots by cells
        P = P_true / P_true.sum(0)  # col (cell) sum to 1
        loss_fn = torch.nn.CosineEmbeddingLoss()
        loss = loss_fn(OT_output, P_true, torch.ones(P.shape[0]))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters
        optimizer.step()

    # This segment is added to return the final mapping
    with torch.no_grad():
        NNLS_output, OT_output = model(sc_adata, st_adata)
        # ct = torch.tensor(sc_adata.obsm['cell_type'].values.T)
        ct = torch.tensor(sc_adata.obsm['cell_type'].T)
        P_true = NNLS_output @ ct.float()
        P = P_true / P_true.sum(0)

    return model, P.numpy()