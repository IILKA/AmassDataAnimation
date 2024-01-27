import numpy as np
import smplx
import torch
import trimesh
import pyrender
gender = "neutral"
num_betas = 10
motion_file = "./Female1Running_c3d/C2_-_Run_to_stand_stageii.npz"
model_folder = "./models"
model_type = "smplx"
motion = np.load(motion_file, allow_pickle = True)
for k, v in motion.items():
    if type(v) is float:
        print(k, v)
    else:
        print(k, v.shape)

if "betas" in motion:
    betas = motion["betas"]
else:
    betas = np.zeros((num_betas,))
num_betas = len(betas)

if "gender" in motion:
    gender = str(motion["gender"])
else:
    gender =  gender 
print(gender)
print(num_betas)
model = smplx.create(
    model_folder, 
    model_type = model_type,
    gender = gender,
    use_face_contour = False, # there is some issue
    num_betas = num_betas, 
    num_expression_coeffs = 10,
    use_pca = False, 
    ext = "npz",
)
betas, expression =torch.tensor(betas).float(), None
betas = betas.unsqueeze(0)[:, : model.num_betas]

poses = torch.tensor(motion["poses"]).float()
global_orient = poses[:, :3]

body_pose = poses[:, 3:66]
body_pose[:, :3] = 0
left_hand_pose = poses[:, 66:111]
right_hand_pose = poses[:, 111:156]


from tqdm.auto import tqdm, trange
for pose_idx in trange(body_pose.size(0)):
    pose_idx = [pose_idx]
    output = model(
        betas = betas, 
        global_orient = global_orient[pose_idx],
        body_pose = body_pose[pose_idx],
        left_hand_pose = left_hand_pose[pose_idx],
        right_hand_pose = right_hand_pose[pose_idx],
        return_verts = True,
    )
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    # rotate the mesh to face the -z axis
    for vertice in vertices:
        #rotate the mesh by 
        vertice[0], vertice[1], vertice[2] = vertice[0], vertice[2], -vertice[1]
    joints = output.joints.detach().cpu().numpy().squeeze()


    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 1.0]
    tri_mesh = trimesh.Trimesh(
        vertices, model.faces, vertex_colors = vertex_colors
        )
    output_path = f"./output/meshes/{pose_idx[0]}.obj"
    tri_mesh.export(str(output_path))
    '''
    if pose_idx[0] == 0:
        print("displaying first pose, exit window to continue processing")
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        #pyrender.Viewer(scene, use_raymond_lighting = True)
    '''












