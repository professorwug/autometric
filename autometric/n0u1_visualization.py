# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/0u1 Visualization.ipynb.

# %% auto 0
__all__ = ['plot_jacobian']

# %% ../nbs/0u1 Visualization.ipynb 1
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %% ../nbs/0u1 Visualization.ipynb 2
def plot_jacobian(X, X_embedded, U, V, S, jac, id_point):
    """
    X: torch.tensor, shape (n_points, n_dim), original data
    X_embedded: torch.tensor, shape (n_points, n_dim), embedded data
    U: torch.tensor, shape (n_points, embedding_dim, embedding_dim), left singular vectors
    V: torch.tensor, shape (n_points, ambient_dim, ambient_dim), right singular vectors
    S: torch.tensor, shape (n_points, embedding_dim), singular values
    jac: torch.tensor, shape (n_points, embedding_dim, ambient_dim), jacobian
    id_point: int, index of the point to plot

    run `U, S, V = torch.linalg.svd(jac, full_matrices=False)` to get U, S, V from jac.
    """
    j = id_point
    jac_np = jac.detach().numpy()[j, :, :]
    jacobian_np = V.detach().numpy()[j, :, :].T
    singval = S[j, :].detach().numpy()
    x0_np = X[j, :].detach().numpy()

    # Compute the singular vectors for the 3D plot
    v1 = jacobian_np[:, 0]
    v2 = jacobian_np[:, 1]
    # Compute the plane corners
    scale = 1
    corner1 = x0_np + scale * v1 + scale * v2
    corner2 = x0_np - scale * v1 + scale * v2
    corner3 = x0_np - scale * v1 - scale * v2
    corner4 = x0_np + scale * v1 - scale * v2
    # Compute the normal vector
    normalv = np.cross(v1, v2)

    # Create subplot figure with 1 row and 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}]],
        subplot_titles=('3D Visualization', '2D Projection')
    )

    # 3D scatter plot
    scatter_3d = go.Scatter3d(
        x=X.detach().numpy()[:, 0],
        y=X.detach().numpy()[:, 1],
        z=X.detach().numpy()[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue', opacity=0.3)
    )

    # Red point representing x0 in 3D
    point_3d = go.Scatter3d(
        x=[x0_np[0]],
        y=[x0_np[1]],
        z=[x0_np[2]],
        mode='markers',
        marker=dict(size=5, color='red')
    )

    # Vectors in 3D
    vectors_3d = []
    colors = ['green', 'blue']
    for i, vec in enumerate([v1, v2]):
        vectors_3d.append(go.Scatter3d(
            x=[x0_np[0], x0_np[0] + vec[0] * scale],
            y=[x0_np[1], x0_np[1] + vec[1] * scale],
            z=[x0_np[2], x0_np[2] + vec[2] * scale],
            mode='lines+text',
            line=dict(color=colors[i], width=5),
            text=[f'{singval[i]:.2e}'],
            textposition='top right'
        ))

    # Normal vector in 3D
    normal_vector_3d = go.Scatter3d(
        x=[x0_np[0], x0_np[0] + normalv[0] * scale],
        y=[x0_np[1], x0_np[1] + normalv[1] * scale],
        z=[x0_np[2], x0_np[2] + normalv[2] * scale],
        mode='lines',
        line=dict(color='red', width=5)
    )

    # Plane in 3D
    plane_3d = go.Mesh3d(
        x=[corner1[0], corner2[0], corner3[0], corner4[0]],
        y=[corner1[1], corner2[1], corner3[1], corner4[1]],
        z=[corner1[2], corner2[2], corner3[2], corner4[2]],
        color='cyan',
        opacity=0.5
    )

    # Add traces for 3D plot
    fig.add_trace(scatter_3d, row=1, col=1)
    fig.add_trace(point_3d, row=1, col=1)
    for vec in vectors_3d:
        fig.add_trace(vec, row=1, col=1)
    fig.add_trace(normal_vector_3d, row=1, col=1)
    fig.add_trace(plane_3d, row=1, col=1)

    # 2D scatter plot
    scatter_2d = go.Scatter(
        x=X_embedded.detach().numpy()[:, 0],
        y=X_embedded.detach().numpy()[:, 1],
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.3)
    )

    # Red point representing x0 in 2D
    point_2d = go.Scatter(
        x=[X_embedded[j, 0].detach().numpy()],
        y=[X_embedded[j, 1].detach().numpy()],
        mode='markers',
        marker=dict(size=5, color='red')
    )
    # 2D arrows
    u_np = U.detach().numpy()[j, :, :]  # Assuming U is defined
    us = u_np.T * singval
    scale = 0.1
    v1_2d = us[:, 0] * scale
    v2_2d = us[:, 1] * scale

    arrow_2d_v1 = go.Scatter(
        x=[X_embedded[j, 0].detach().numpy(), X_embedded[j, 0].detach().numpy() + v1_2d[0]],
        y=[X_embedded[j, 1].detach().numpy(), X_embedded[j, 1].detach().numpy() + v1_2d[1]],
        mode='lines+text',
        line=dict(color='green', width=2),
        text=[f'{singval[0]:.2e}'],
        textposition="top right"
    )

    arrow_2d_v2 = go.Scatter(
        x=[X_embedded[j, 0].detach().numpy(), X_embedded[j, 0].detach().numpy() + v2_2d[0]],
        y=[X_embedded[j, 1].detach().numpy(), X_embedded[j, 1].detach().numpy() + v2_2d[1]],
        mode='lines+text',
        line=dict(color='blue', width=2),
        text=[f'{singval[1]:.2e}'],
        textposition="bottom right"
    )

    # Add traces for 2D plot
    fig.add_trace(scatter_2d, row=1, col=2)
    fig.add_trace(point_2d, row=1, col=2)

    fig.add_trace(arrow_2d_v1, row=1, col=2)
    fig.add_trace(arrow_2d_v2, row=1, col=2)

    # Update layout for visual consistency
    fig.update_layout(
        height=600,
        width=1200,
        showlegend=False,
        title_text="Side By Side Subplots"
    )

    # Show figure
    fig.show()
    
    plt.figure(figsize=(8, 4))
    sns.heatmap(jac_np, annot=True, fmt=".2e", cmap='viridis', cbar=True)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
    print(jac_np)