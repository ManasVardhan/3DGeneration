import trimesh
import plotly.graph_objects as go
import numpy as np

filename = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/Completed/4/10_19_2025.obj"  # Change this to your file path

mesh = trimesh.load_mesh(filename)

# Extract vertices and faces
vertices = mesh.vertices
faces = mesh.faces

fig = go.Figure(data=[
    go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightblue',
        opacity=0.8,
        flatshading=True,
        lighting=dict(
            ambient=0.5,
            diffuse=0.8,
            specular=0.2,
            roughness=0.5
        ),
        lightposition=dict(
            x=100,
            y=200,
            z=300
        )
    )
])


# Update layout for better visualization
fig.update_layout(
    title=f"3D Visualization: {filename}",
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
        aspectmode='data',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    width=800,
    height=600
)

fig.show()