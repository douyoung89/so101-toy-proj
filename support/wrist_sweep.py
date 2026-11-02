"""Measure mechanical and camera/fixture-valid wrist intervals without changing a scene."""
import argparse
from itertools import product
import json
import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import stage2
import stage5
from ocr_memory import accepted, parse

ROOT = Path(__file__).resolve().parents[1]
STATE = mujoco.mjtState.mjSTATE_INTEGRATION
PENETRATION_TOLERANCE = .0002
SIGNS = np.array(list(product([-1.,1.], repeat=3)))


def read_json(path):
    return json.loads(path.read_text())


def inventory(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    tags = {}
    tree = ET.parse(path)
    trees = [tree]
    for include in tree.findall('include'):
        trees.append(ET.parse(path.parent/include.get('file')))
    for tree in trees:
        for element in tree.findall('actuator/*'):
            tags[element.get('name')] = element.tag
    joints = []
    for j in range(model.njnt):
        hinge = model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        start = float(data.qpos[model.jnt_qposadr[j]])
        joints.append({'name':model.joint(j).name, 'limited':bool(model.jnt_limited[j]),
                       'type':'hinge' if hinge else 'slide_or_other',
                       'range_native':model.jnt_range[j].tolist(),
                       'range_deg':np.rad2deg(model.jnt_range[j]).tolist() if hinge else None,
                       'start_native':start, 'start_deg':float(np.rad2deg(start)) if hinge else None})
    actuators = []
    for a in range(model.nu):
        joint_id = int(model.actuator_trnid[a,0])
        position_angle = (tags.get(model.actuator(a).name)=='position' and
                          model.actuator_trntype[a]==mujoco.mjtTrn.mjTRN_JOINT and
                          model.jnt_type[joint_id]==mujoco.mjtJoint.mjJNT_HINGE and model.actuator_gear[a,0]==1)
        actuators.append({'name':model.actuator(a).name, 'xml_type':tags.get(model.actuator(a).name),
                          'ctrl_limited':bool(model.actuator_ctrllimited[a]),
                          'ctrl_range_native':model.actuator_ctrlrange[a].tolist(),
                          'ctrl_units':'rad' if position_angle else 'actuator-dependent',
                          'ctrl_range_deg':np.rad2deg(model.actuator_ctrlrange[a]).tolist() if position_angle else None,
                          'gain':model.actuator_gainprm[a,:3].tolist(), 'bias':model.actuator_biasprm[a,:3].tolist()})
    return {'scene':str(path.relative_to(ROOT)), 'joints':joints, 'actuators':actuators}


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


def move_checked(model,data,goal,pairs,fixture,payload):
    """Same position interpolation/settling as stage2.move, with extra fixture diagnostics."""
    qadr,dadr,aids=stage2.ids(model)
    start=data.ctrl[aids].copy()
    elapsed=stable=0.
    first_geometry=first_contact=None
    path_frame_ok=True
    steps=0
    while elapsed<7.:
        t=min(elapsed/1.5,1.)
        data.ctrl[aids]=start+t*t*(3-2*t)*(goal-start)
        mujoco.mj_step(model,data)
        elapsed+=model.opt.timestep;steps+=1
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError('Nonfinite dynamics')
        bad=stage2.contacts(model,data)
        if bad and first_contact is None:
            first_contact={'elapsed_s':elapsed,'contacts':bad}
        if steps%5==0:  # 10 ms diagnostic sampling, no change to geometry/contact parameters.
            mujoco.mj_forward(model,data)
            if first_geometry is None:
                bad=geometric_penetrations(model,data,pairs)
                if bad:
                    first_geometry={'elapsed_s':elapsed,'pairs':bad}
            cg=camera_geometry(model,data,fixture,payload)
            path_frame_ok &= cg['box_in_frame'] and cg['label_in_frame'] and cg['fixture_in_frame']
        error=float(np.max(np.abs(data.qpos[qadr]-goal)))
        speed=float(np.max(np.abs(data.qvel[dadr])))
        stable=stable+model.opt.timestep if t==1 and error<.035 and speed<.05 else 0.
        if stable>=.3:
            mujoco.mj_forward(model,data)
            return {'settle_seconds':elapsed,'max_error_rad':error,'final_speed_rad_s':speed,
                    'first_fixture_penetration':first_geometry, 'first_enabled_contact':first_contact,
                    'path_all_parts_in_frame':bool(path_frame_ok)}
    raise RuntimeError(f'Did not settle: error={error}, speed={speed}')


def contiguous_reference_component(rows):
    ordered=sorted(rows,key=lambda r:r['relative_command_deg'])
    zero=next(i for i,r in enumerate(ordered) if abs(r['relative_command_deg'])<1e-8)
    if not ordered[zero]['experiment_valid']:
        return []
    left=right=zero
    while left>0 and ordered[left-1]['experiment_valid']:
        left-=1
    while right+1<len(ordered) and ordered[right+1]['experiment_valid']:
        right+=1
    return ordered[left:right+1]


def sweep(episode_id):
    epdir=next((ROOT/'outputs/v2_pilot/episodes').glob(f'*/{episode_id}'))
    config=read_json(epdir/'evaluation/config.json')
    ep=config['episode']
    path=epdir/'evaluation/scene.xml'
    out=ROOT/'outputs/wrist_sweep'/episode_id
    (out/'images').mkdir(parents=True,exist_ok=True)
    original=ROOT/'external/SO-ARM100/Simulation/SO101/scene_active.xml'
    inventories={'scene_active':inventory(original),'fixed_experiment_scene':inventory(path)}
    stage5.write(out/'model_inventory.json',inventories)
    for name,inv in inventories.items():
        print(name, flush=True)
        for j in inv['joints']:
            print(j['name'],'limited',j['limited'],'range_deg',j['range_deg'],'start_deg',j['start_deg'],flush=True)
        for a in inv['actuators']:
            print('actuator',a['name'],a['xml_type'],a['ctrl_range_native'],a['ctrl_units'],flush=True)
    model=mujoco.MjModel.from_xml_path(str(path))
    baseline=mujoco.MjData(model)
    index=model.jnt_qposadr[model.joint('label_shutter').id]
    baseline.qpos[index]=model.eq_data[0,0]
    mujoco.mj_forward(model,baseline)
    pairs,fixture,payload=fixture_pairs(model)
    goal=np.array(ep['design']['reference_joints_rad'])
    baseline_motion=move_checked(model,baseline,goal,pairs,fixture,payload)
    if baseline_motion['first_fixture_penetration'] or baseline_motion['first_enabled_contact']:
        raise RuntimeError('Reference pose has an unexpected penetration')
    qadr,_,aids=stage2.ids(model)
    theta0=float(baseline.qpos[qadr[4]])
    state=np.zeros(mujoco.mj_stateSize(model,STATE))
    mujoco.mj_getState(model,baseline,state,STATE)
    np.save(out/'reference_integration_state.npy',state)
    low=max(model.jnt_range[model.joint('wrist_roll').id,0],model.actuator_ctrlrange[aids[4],0])
    high=min(model.jnt_range[model.joint('wrist_roll').id,1],model.actuator_ctrlrange[aids[4],1])
    relative_low=float(np.rad2deg(low-theta0)+5)
    relative_high=float(np.rad2deg(high-theta0)-5)
    coarse=list(np.arange(np.ceil(relative_low/10)*10,np.floor(relative_high/10)*10+1,10))
    thresholds=read_json(ROOT/'outputs/stage4/thresholds.json')['thresholds']
    truth={f:int(ep['label'][f]) for f in ['protein_g','sugar_g']}
    rows=[]
    with mujoco.Renderer(model,height=720,width=960) as renderer:
        def measure(offset,phase):
            data=mujoco.MjData(model)
            mujoco.mj_setState(model,data,state,STATE)
            mujoco.mj_forward(model,data)
            before=stage5.joint_state(model,data)
            target=goal.copy();target[4]=theta0+np.deg2rad(offset)
            motion=move_checked(model,data,target,pairs,fixture,payload)
            geometry=camera_geometry(model,data,fixture,payload)
            penetration=geometric_penetrations(model,data,pairs)
            renderer.update_scene(data,camera='label_camera')
            frame=renderer.render().copy()
            image=out/'images'/f'{offset:+07.2f}.png'
            Image.fromarray(frame).save(image)
            raw=stage5.recognize(frame)
            readings=accepted(parse(raw),thresholds)
            reasons=[]
            for key in ['box_in_frame','label_in_frame','fixture_in_frame']:
                if not geometry[key]:reasons.append(key)
            if geometry['label_facing_cosine']<=.15:reasons.append('back_or_edge_label')
            if not motion['path_all_parts_in_frame']:reasons.append('parts_leave_frame_during_motion')
            if motion['first_fixture_penetration'] or penetration:reasons.append('fixture_or_robot_interpenetration')
            if motion['first_enabled_contact']:reasons.append('enabled_contact')
            row={'relative_command_deg':float(offset),'phase':phase,'absolute_command_deg':float(np.rad2deg(target[4])),
                 'actual_deg':float(np.rad2deg(data.qpos[qadr[4]])),
                 'actual_relative_deg':float(np.rad2deg(data.qpos[qadr[4]]-theta0)),
                 'before':before,'after':stage5.joint_state(model,data),'motion':motion,'geometry':geometry,
                 'endpoint_penetrations':penetration,'ocr':raw,'readings':readings,
                 'correct_reading':{f:readings[f]['value'] is not None and readings[f]['value']==truth[f] for f in truth},
                 'experiment_valid':not reasons,'invalid_reasons':reasons,'image':str(image.relative_to(out))}
            rows.append(row)
            stage5.write(out/'measurements.json',rows)
            print(f'{phase} {offset:+.0f} deg: valid={not reasons}, P={readings["protein_g"]["value"]}, S={readings["sugar_g"]["value"]}, {reasons}',flush=True)
        for offset in coarse:
            measure(float(offset),'coarse_10deg')
        component=contiguous_reference_component(rows)
        if not component:
            raise RuntimeError('Reference is outside valid interval; see measurements.json')
        minimum=component[0]['relative_command_deg']; maximum=component[-1]['relative_command_deg']
        existing={r['relative_command_deg'] for r in rows}
        fine=[]
        if minimum-10 in existing:fine.append(minimum-5)
        if maximum+10 in existing:fine.append(maximum+5)
        for offset in fine:
            measure(offset,'boundary_5deg')
        component=contiguous_reference_component(rows)
        missing_fields=[f for f in truth if not any(r['correct_reading'][f] for r in component)]
        if missing_fields:
            # A known mechanism reference can lie between coarse ticks; only then inspect its neighborhood.
            for name in ep['design']['partial_pair']:
                angle=next(p['joints']['wrist_roll'] for p in read_json(ROOT/'poses.json')['poses'] if p['name']==name)
                offset=float(round(np.rad2deg(angle-theta0)/5)*5)
                if offset not in {r['relative_command_deg'] for r in rows}:
                    measure(offset,'readability_followup_5deg')
        # If 10-degree actions miss a demonstrated readable field, measure the full
        # 5-degree action lattice rather than recommending an unusable action step.
        component=contiguous_reference_component(rows)
        coarse_field_coverage={f:any(r['correct_reading'][f] for r in component if r['phase']=='coarse_10deg') for f in truth}
        use_five=not all(coarse_field_coverage.values())
        if use_five:
            low_valid=component[0]['relative_command_deg'];high_valid=component[-1]['relative_command_deg']
            for offset in np.arange(low_valid,high_valid+1,5):
                if float(offset) not in {r['relative_command_deg'] for r in rows}:
                    measure(float(offset),'action_grid_5deg')
    rows.sort(key=lambda r:r['relative_command_deg'])
    stage5.write(out/'measurements.json',rows)
    component=contiguous_reference_component(rows)
    r0=next(r for r in rows if r['relative_command_deg']==0)
    r10=next(r for r in rows if r['relative_command_deg']==10)
    shift=(np.array(r10['geometry']['shutter_minus_box_screen_xy'])-np.array(r0['geometry']['shutter_minus_box_screen_xy'])).tolist()
    face_shift=(np.array(r10['geometry']['shutter_minus_label_face_screen_xy'])-np.array(r0['geometry']['shutter_minus_label_face_screen_xy'])).tolist()
    absolute_shift=(np.array(r10['geometry']['shutter_screen_xy'])-np.array(r0['geometry']['shutter_screen_xy'])).tolist()
    valid_low=component[0]['relative_command_deg'];valid_high=component[-1]['relative_command_deg']
    below=[r['relative_command_deg'] for r in rows if r['relative_command_deg']<valid_low]
    above=[r['relative_command_deg'] for r in rows if r['relative_command_deg']>valid_high]
    report={'episode_id':episode_id,'fixed_scene':str(path.relative_to(ROOT)),
            'reference_command_rad':float(goal[4]),'reference_actual_rad':theta0,'reference_actual_deg':float(np.rad2deg(theta0)),
            'mechanical_joint_range_deg':np.rad2deg(model.jnt_range[model.joint('wrist_roll').id]).tolist(),
            'joint_and_position_control_intersection_deg':np.rad2deg([low,high]).tolist(),
            'joint_limit_margin_deg':5,'measured_relative_candidates_deg':[r['relative_command_deg'] for r in rows],
            'experimental_relative_range_deg':[valid_low,valid_high],
            'boundary_brackets_deg':{'lower':[max(below) if below else None,valid_low],'upper':[valid_high,min(above) if above else None]},
            'initial_action_step_deg':10,'action_step_deg':5 if use_five else 10,
            'action_step_reason':'10-degree lattice missed a field that was read at an intermediate angle' if use_five else 'Both fields readable on 10-degree lattice',
            'regular_10deg_views_in_range':len([r for r in component if r['phase']=='coarse_10deg']),
            'recommended_grid_views_in_range':int(round((valid_high-valid_low)/(5 if use_five else 10)))+1,
            'correct_protein_offsets_deg':[r['relative_command_deg'] for r in component if r['correct_reading']['protein_g']],
            'correct_sugar_offsets_deg':[r['relative_command_deg'] for r in component if r['correct_reading']['sugar_g']],
            'both_correct_in_one_view_offsets_deg':[r['relative_command_deg'] for r in component if all(r['correct_reading'].values())],
            'plus10_shutter_vs_box_pixel_delta_xy':shift,
            'plus10_shutter_vs_label_face_pixel_delta_xy':face_shift,
            'plus10_shutter_absolute_pixel_delta_xy':absolute_shift,
            'shutter_local_y_change_m':r10['geometry']['shutter_local_y_m']-r0['geometry']['shutter_local_y_m'],
            'penetration_tolerance_m':PENETRATION_TOLERANCE,'path_diagnostic_interval_s':.01,
            'source_hashes':{str(p.relative_to(ROOT)):stage5.sha(p) for p in [path,path.parent/'robot.xml',epdir/'evaluation/config.json',Path(__file__)]},
            'ocr_thresholds':thresholds,'parameters_changed':False,
            'reset':'mjSTATE_INTEGRATION reference snapshot before every independent actuator move',
            'scope':'This single scene only. Geometric path diagnostics sampled every 10ms, range boundaries bracketed to 5deg; no continuous-space collision proof.'}
    stage5.write(out/'result.json',report)
    write_artifacts(out,rows,report)
    print(json.dumps(report,indent=2),flush=True)


def write_artifacts(out,rows,report):
    flat=[]
    for r in rows:
        flat.append({'relative_command_deg':r['relative_command_deg'],'absolute_command_deg':r['absolute_command_deg'],
                     'actual_deg':r['actual_deg'],'actual_relative_deg':r['actual_relative_deg'],
                     'phase':r['phase'],'experiment_valid':r['experiment_valid'],
                     **{k:r['geometry'][k] for k in ['box_in_frame','label_in_frame','fixture_in_frame','label_direction','label_facing_cosine','shutter_local_y_m']},
                     'protein_g':r['readings']['protein_g']['value'],'protein_confidence':r['readings']['protein_g']['confidence'],
                     'sugar_g':r['readings']['sugar_g']['value'],'sugar_confidence':r['readings']['sugar_g']['confidence'],
                     'reasons':';'.join(r['invalid_reasons']),'raw_ocr':' | '.join(d['text'] for d in r['ocr']),'image':r['image']})
    with (out/'angles.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
    for name,selected in [('all',rows),('experiment',[r for r in rows if r['experiment_valid']])]:
        cols=4; tile_w=400;tile_h=410
        sheet=Image.new('RGB',(cols*tile_w,((len(selected)+cols-1)//cols)*tile_h),'white')
        draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(selected):
            x=i%cols*tile_w;y=i//cols*tile_h
            with Image.open(out/r['image']) as image:
                sheet.paste(image.crop((250,140,710,580)).resize((380,330)),(x+10,y+70))
            text=f'{r["relative_command_deg"]:+.0f} deg / actual {r["actual_relative_deg"]:+.2f}\nP={r["readings"]["protein_g"]["value"]} S={r["readings"]["sugar_g"]["value"]} / {"VALID" if r["experiment_valid"] else "EXCLUDE"}\n{r["geometry"]["label_direction"]}'
            draw.multiline_text((x+10,y+3),text,font=font,fill='#17683b' if r['experiment_valid'] else '#a52323')
        sheet.save(out/f'contact_sheet_{name}.jpg')
    rows_html=''.join(f'<tr><td>{r["relative_command_deg"]:+.0f}°</td><td>{r["actual_relative_deg"]:+.3f}°</td><td>{r["experiment_valid"]}</td><td>{r["protein_g"]}</td><td>{r["sugar_g"]}</td><td>{r["reasons"]}</td><td><a href="{r["image"]}">original</a></td></tr>' for r in flat)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:24px}img{max-width:100%}td,th{padding:9px;border-bottom:1px solid #ccc}pre{white-space:pre-wrap}</style><h1>Wrist angle sweep</h1><p>Contact sheet is zoomed for inspection. Original 960×720 frames are linked below. None means no accepted OCR value.</p><pre>'+json.dumps(report,indent=2)+'</pre><h2>Selected experiment interval</h2><img src="contact_sheet_experiment.jpg"><h2>All measured angles</h2><table><tr><th>Command relative</th><th>Actual relative</th><th>Valid</th><th>Protein</th><th>Sugar</th><th>Exclusion</th><th>Image</th></tr>'+rows_html+'</table><img src="contact_sheet_all.jpg">')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',default='v2_005')
    args=parser.parse_args()
    sweep(args.episode)
