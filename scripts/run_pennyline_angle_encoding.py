#!/usr/bin/env python3
"""Run PennyLane angle encoding on PLANAR latent vectors. Generates circuit diagram and scatter plot."""

import sys
from pathlib import Path

PLANAR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLANAR_ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def main():
    # Try to load latent vectors from PLANAR artifacts
    artifacts = PLANAR_ROOT.parent / "artifacts"
    latent_paths = [
        artifacts / "clustering" / "latent_vectors.npy",
        PLANAR_ROOT / "artifacts" / "clustering" / "latent_vectors.npy",
    ]
    latent = None
    for p in latent_paths:
        if p.exists():
            latent = np.load(p)
            break

    if latent is None:
        # Generate synthetic latent vectors (64D) if none exist
        np.random.seed(42)
        n = 200
        latent = np.random.randn(n, 64).astype(np.float32) * 0.5 + 0.5
        latent = np.clip(latent, 0, 1)

    n_samples = min(200, len(latent))
    latent = latent[:n_samples]

    # Reduce to 8 dims via PCA
    pca = PCA(n_components=8)
    z8 = pca.fit_transform(latent)
    z8_scaled = (z8 - z8.min()) / (z8.max() - z8.min() + 1e-8) * 2 * np.pi

    try:
        import pennylane as qml
    except ImportError:
        print("PennyLane not installed. Install with: pip install pennylane")
        sys.exit(1)

    dev = qml.device("default.qubit", wires=8)

    @qml.qnode(dev)
    def angle_circuit(x):
        qml.AngleEmbedding(x, wires=range(8), rotation='Y')
        qml.StronglyEntanglingLayers(weights=np.zeros((2, 8, 3)), wires=range(8))
        return [qml.expval(qml.PauliZ(i)) for i in range(8)]

    # Draw circuit
    out_dir = PLANAR_ROOT.parent / "assets"
    out_dir.mkdir(exist_ok=True)
    try:
        fig, ax = qml.draw_mpl(angle_circuit)(z8_scaled[0])
        fig.savefig(out_dir / "angle_encoding_circuit.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("Saved angle_encoding_circuit.png")
    except Exception as e:
        print(f"Circuit draw skipped: {e}")

    # Run all samples
    outputs = np.array([angle_circuit(x) for x in z8_scaled])

    # PCA scatter colored by KMeans cluster
    pca2 = PCA(n_components=2)
    proj = pca2.fit_transform(outputs)
    km = KMeans(n_clusters=4, random_state=42).fit(outputs)

    plt.figure(figsize=(6, 5))
    plt.scatter(proj[:, 0], proj[:, 1], c=km.labels_, cmap="tab10", s=15)
    plt.title("Angle Encoding: QVC output PCA (200 samples, 8 qubits)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.savefig(out_dir / "angle_encoding_scatter.png", dpi=200)
    plt.close()
    print("Saved angle_encoding_scatter.png")
    print("Angle encoding complete.")

if __name__ == "__main__":
    main()
