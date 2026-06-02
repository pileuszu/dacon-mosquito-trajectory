import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

def main():
    print("Loading training data for fast_turning regime...")
    data_path = Path("step38_2regime_regression/train_ranker_v38_fast_turning.csv")
    if not data_path.exists():
        print(f"Error: Training file {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    # Select features and target
    X = df[['spec_par', 'spec_perp']].to_numpy()
    y = df['target'].to_numpy()
    
    # Train a simple Decision Tree to visualize the decision boundary surface
    print("Training Decision Tree classifier...")
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)
    
    # Create meshgrid to evaluate decision surface
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Define original unscaled candidate specs for visualization (subset of fast specs)
    orig_par = np.array([-0.5, 0.0, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0])
    orig_perp = np.array([-1.5, -1.0, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0, 1.5])
    
    # Grid points
    orig_points_x = []
    orig_points_y = []
    for p in orig_par:
        for n in orig_perp:
            orig_points_x.append(p)
            orig_points_y.append(n)
    orig_points_x = np.array(orig_points_x)
    orig_points_y = np.array(orig_points_y)
    
    # Scaled mismatch points (stretched by S_grid = 2.0)
    S_grid = 2.0
    scaled_points_x = orig_points_x * S_grid
    scaled_points_y = orig_points_y * S_grid
    
    # Set up styling (clean dark/gray tech aesthetic)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    for ax in axes:
        ax.set_facecolor("#1e1e1e")
        fig.patch.set_facecolor("#121212")
        ax.grid(color="#333333", linestyle="--", alpha=0.5)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        
    # Plot 1: Decoupled Feature Space (Correct Alignment)
    contour1 = axes[0].contourf(xx, yy, Z, levels=20, cmap="RdYlGn", alpha=0.45)
    axes[0].scatter(orig_points_x, orig_points_y, c="cyan", s=15, edgecolors="black", label="Unscaled Test Points", alpha=0.9)
    axes[0].set_title("Decoupled Feature Space (Correct)\nFeatures Matched with Training Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("spec_par (Acceleration Tangential)", fontsize=10)
    axes[0].set_ylabel("spec_perp (Acceleration Normal)", fontsize=10)
    axes[0].set_xlim(-1.5, 3.5)
    axes[0].set_ylim(-2.5, 2.5)
    axes[0].legend(loc="upper left", facecolor="#2a2a2a", edgecolor="none", labelcolor="white")
    
    # Plot 2: Scaled Mismatch (Broken Alignment)
    contour2 = axes[1].contourf(xx, yy, Z, levels=20, cmap="RdYlGn", alpha=0.45)
    axes[1].scatter(scaled_points_x, scaled_points_y, c="magenta", s=15, edgecolors="black", label="Incorrect Scaled Points (x 2.0)", alpha=0.9)
    axes[1].set_title("Feature Mismatch (Broken)\nStretched Features Shift Out of Learned Bounds", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("spec_par (Acceleration Tangential * S_grid)", fontsize=10)
    axes[1].set_xlim(-1.5, 3.5)
    axes[1].set_ylim(-2.5, 2.5)
    axes[1].legend(loc="upper left", facecolor="#2a2a2a", edgecolor="none", labelcolor="white")
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour1, cax=cbar_ax)
    cbar.set_label("Predicted P(Hit@1.0cm)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    
    plt.suptitle("Impact of Test-Time Feature Scaling on GBDT Decision Boundaries", color="white", fontsize=16, fontweight="bold", y=0.97)
    
    # Ensure docs directory exists
    Path("docs").mkdir(exist_ok=True)
    out_path = Path("docs/decision_boundary_distortion.png")
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Visualization saved successfully to {out_path}")

if __name__ == "__main__":
    main()
