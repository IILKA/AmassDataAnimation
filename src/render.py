import pyrender 
import trimesh
import numpy as np 
import cv2
import time
import os

print("Pyrender version", pyrender.__version__)
print("Trimesh version", trimesh.__version__)
print("Numpy version", np.__version__)
print("OpenCV version", cv2.__version__)

#set the camera position
camera_pose = np.eye(4)
camera_pose[0,3] = 0.0
camera_pose[1,3] = 0.0
camera_pose[2,3] = -2.0
camera_pose[0,0] = 1.0
camera_pose[1,1] = 1.0
camera_pose[2,2] = -1.0
#set the basic scene
scene = pyrender.Scene()
camera = pyrender.PerspectiveCamera(yfov = np.pi/3.0, aspectRatio = 0.7)
scene.add(camera, pose = camera_pose)
light = pyrender.SpotLight(color = np.ones(3), intensity = 4.0, innerConeAngle = np.pi/16.0)
scene.add(light, pose = camera_pose)
#set the renderer 
r = pyrender.OffscreenRenderer(400,600)


from tqdm.auto import tqdm, trange
def main():
    files = os.listdir("./output/meshes")
    for cnt in trange(len(files)):
        mesh = trimesh.load_mesh(f"./output/meshes/{cnt}.obj")
        mesh = pyrender.Mesh.from_trimesh(mesh)
        mesh_node = scene.add(mesh)
        color,depth = r.render(scene)
        image_array = np.reshape(color,(600, 400, 3) )
        cv2.imwrite(f"./output/pics/{cnt}.png", image_array)
        scene.remove_node(mesh_node)

     


    
if __name__ == "__main__":
    main()
