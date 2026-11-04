"""Camera bounds and physical penetration checks used during each robot move."""
from itertools import product
import mujoco
import numpy as np

PENETRATION_TOLERANCE = 0.0002
SIGNS = np.array(list(product([-1., 1.], repeat=3)))

def contacts(model, data):
    bad = []
    for c in data.contact:
        if c.dist >= -0.0005:
            continue
        bodies = {model.body(model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)}
        if bodies == {'world', 'base'}:
            continue
        bad.append({'geoms': [model.geom(g).name for g in (c.geom1, c.geom2)], 'depth_m': float(-c.dist)})
    return bad


def corners(model,data,geom):
    return data.geom_xpos[geom]+(SIGNS*model.geom_size[geom])@data.geom_xmat[geom].reshape(3,3).T


def project(model,data,points):
    camera = model.camera('label_camera').id
    points = np.asarray(points)
    local = (points-data.cam_xpos[camera])@data.cam_xmat[camera].reshape(3,3)
    focal = 720/(2*np.tan(np.deg2rad(model.cam_fovy[camera])/2))
    depth = -local[:,2]
    return np.column_stack((480+focal*local[:,0]/depth,360-focal*local[:,1]/depth)), depth


def in_frame(model,data,points):
    pixels,depth = project(model,data,points)
    return bool(np.all(depth>0) and np.all(pixels>8) and np.all(pixels<[952,712]))


def fixture_pairs(model):
    fixture = [g for g in range(model.ngeom) if model.geom(g).name and
               (model.geom(g).name.startswith('hood_') or model.geom(g).name=='partial_cover')]
    payload = model.geom('label_box_geom').id
    slider = model.geom('partial_cover').id
    robot = [g for g in range(model.ngeom) if model.geom_group[g]==3]
    pairs = {(min(a,b),max(a,b)) for a in fixture for b in robot}
    pairs |= {(min(slider,b),max(slider,b)) for b in fixture+[payload] if slider!=b}
    for b in robot:
        # Original intentional mounting exclusions remain explicit.
        if model.body(model.geom_bodyid[b]).name not in ['gripper','moving_jaw_so101_v1']:
            pairs.add((min(payload,b),max(payload,b)))
    return sorted(pairs), fixture, payload


def geometric_penetrations(model,data,pairs):
    hits = []
    for a,b in pairs:
        radius = model.geom_rbound[a]+model.geom_rbound[b]
        if np.sum((data.geom_xpos[a]-data.geom_xpos[b])**2) > (radius+.001)**2:
            continue
        distance = float(mujoco.mj_geomDistance(model,data,a,b,.01,None))
        if distance < -PENETRATION_TOLERANCE:
            hits.append({'geoms':[model.geom(a).name,model.geom(b).name], 'penetration_m':-distance})
    return hits


def camera_geometry(model,data,fixture,payload):
    rotation=data.geom_xmat[payload].reshape(3,3)
    face=data.geom_xpos[payload]+rotation[:,2]*model.geom_size[payload,2]
    label=np.array([face+rotation@np.array([x,y,0.]) for x in [-.025,.025] for y in [-.025,.025]])
    cam=data.cam_xpos[model.camera('label_camera').id]
    cosine=float(rotation[:,2]@(cam-face)/np.linalg.norm(cam-face))
    shutter=model.geom('partial_cover').id
    screen,_=project(model,data,[data.geom_xpos[shutter],data.geom_xpos[payload],face])
    return {'box_in_frame':in_frame(model,data,corners(model,data,payload)),
            'label_in_frame':in_frame(model,data,label),
            'fixture_in_frame':all(in_frame(model,data,corners(model,data,g)) for g in fixture),
            'label_facing_cosine':cosine,
            'label_direction':'front' if cosine>=.8 else 'oblique' if cosine>.15 else 'edge' if cosine>0 else 'back',
            'shutter_screen_xy':screen[0].tolist(), 'shutter_minus_box_screen_xy':(screen[0]-screen[1]).tolist(),
            'shutter_minus_label_face_screen_xy':(screen[0]-screen[2]).tolist(),
            'shutter_local_y_m':float(data.qpos[model.jnt_qposadr[model.joint('label_shutter').id]])-.006}
