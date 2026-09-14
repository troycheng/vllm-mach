"""Train-only output residual subspace; original W4 remains unchanged."""
import torch


def fit(error, train, tsha):
    assert error.shape == (17408, 5120) and error.dtype == torch.float32
    assert train.shape == (256, 5120) and train.dtype == torch.bfloat16
    y = train.float() @ error.T
    gram = (y.double() @ y.double().T).cpu()
    eigen, u = torch.linalg.eigh((gram + gram.T)*0.5)
    order = torch.argsort(eigen, descending=True)
    eigen, u = eigen[order], u[:, order]
    assert eigen[63] > 0 and torch.isfinite(eigen).all()
    basis = (y.double().T @ u[:, :64].cuda() / eigen[:64].sqrt().cuda()).float()
    factors = {}
    receipts = {}
    for rank in (32, 64):
        a = (error.T @ basis[:, :rank]).to(torch.bfloat16).contiguous()
        b = basis[:, :rank].T.to(torch.bfloat16).contiguous()
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        name = f'aware{rank}'
        factors[name] = (a, b)
        receipts[name] = {'a_sha256': tsha(a), 'b_sha256': tsha(b), 'rank': rank,
                          'extra_weight_bytes': a.nbytes+b.nbytes,
                          'train_residual_energy_in_subspace': float(eigen[:rank].sum()/eigen.sum()),
                          'scope': 'Linear output projection of original W residual, fitted on QA/code train only; no heldout/fullMLP optimization.'}
    return factors, {'factors': receipts, 'train_sha256': tsha(train),
                     'basis_orthogonality_max_abs': float((basis.T@basis-torch.eye(64, device=basis.device)).abs().max()),
                     'factorization': 'Y=Xtrain*(Wbf16-decode(W4)).T; V=top64 output singular vectors; A=Delta.T*V, B=V.T; store BF16.'}
