import numpy as np
import utm
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

def plot_kinematics(eul,inertia_dir,ax_3d,corners):

    tl, tr, bl, br = corners

    ax_3d.cla()
    ax_3d.plot([0,inertia_dir[0]],[0,inertia_dir[1]],[0,inertia_dir[2]], color='r')

    ax_3d.quiver(0, 0, 0, 1, 0, 0, length=1, linewidth=2, color='red')
    ax_3d.quiver(0, 0, 0, 0, 1, 0, length=1, linewidth=2, color='green')
    ax_3d.quiver(0, 0, 0, 0, 0, 1, length=1, linewidth=2, color='blue')

    ax_3d.text(tl[0], tl[1], tl[2]+0.1, "tl", color='black')
    ax_3d.text(tr[0], tr[1], tr[2]+0.1, "tr", color='black')
    ax_3d.text(bl[0], bl[1], bl[2]-0.1, "bl", color='black')
    ax_3d.text(br[0], br[1], br[2]-0.1, "br", color='black')

    ax_3d.plot([0,tl[0]],[0,tl[1]],[0,tl[2]], color='black')
    ax_3d.plot([0,tr[0]],[0,tr[1]],[0,tr[2]], color='black')
    ax_3d.plot([0,bl[0]],[0,bl[1]],[0,bl[2]], color='black')
    ax_3d.plot([0,br[0]],[0,br[1]],[0,br[2]], color='black')
    ax_3d.plot([tl[0],tr[0]],[tl[1],tr[1]],[tl[2],tr[2]], color='black')
    ax_3d.plot([tl[0],bl[0]],[tl[1],bl[1]],[tl[2],bl[2]], color='black')
    ax_3d.plot([tr[0],br[0]],[tr[1],br[1]],[tr[2],br[2]], color='black')
    ax_3d.plot([bl[0],br[0]],[bl[1],br[1]],[bl[2],br[2]], color='black')

    ax_3d.set_xlim([-1.5,1.5])
    ax_3d.set_ylim([-1.5,1.5])
    ax_3d.set_zlim([-1.5,1.5])

    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z')

    plt.pause(0.01)

def make_smooth(data):

    len = data.shape[0]
    smooth_data = data.copy()
    for i in range(1,len-1):
        prev = smooth_data[i-1,:]
        next = smooth_data[i+1,:]

        smooth_data[i,:] = (prev + next)/2

    return np.array(smooth_data)

def interpolate(drone_time, gt_time, gt_poses, synced=False):

    if not synced:
        gt_time = gt_time[:] - gt_time[0]
        drone_time = drone_time[:] - drone_time[0]

    gt_poses_new = []
    idxs = []
    for i, dtime in enumerate(drone_time):
        gt_time_diff = (gt_time < dtime).astype(np.int32)
        prev_data = gt_time_diff.sum()-1
        next_data = prev_data+1

        if prev_data<0:
            diff = np.abs(gt_time[0] - dtime)
            if diff < 0.5:
                gt_poses_new.append(gt_poses[0,:])
                idxs.append(i)

        elif next_data > gt_poses.shape[0]-1:
            break
        else:
            time_step = gt_time[next_data] - gt_time[prev_data]
            interp = gt_poses[next_data,:]*(dtime-gt_time[prev_data]) / time_step + \
                     gt_poses[prev_data,:]*(gt_time[next_data]-dtime) / time_step

            gt_poses_new.append(interp)
            idxs.append(i)

    # print(idxs)
    if synced:
        return np.array(gt_poses_new), idxs
    else:
        return np.array(gt_poses_new)

def field_to_ned(poses, angle):

    angle_rad = angle * 3.1415 / 180.0

    DCM = np.zeros((3,3))
    DCM[0,0] = np.cos(-angle_rad)
    DCM[0,1] = np.sin(-angle_rad)
    DCM[1,0] = -np.sin(-angle_rad)
    DCM[1,1] = np.cos(-angle_rad)
    DCM[2,2] = 1

    ned_poses = []
    for pos in poses:
        ned_pos = np.matmul(DCM,pos)
        ned_poses.append(ned_pos)

    ned_poses = np.array(ned_poses)
    return ned_poses

def gps_to_field(bl_loc, angle, locs):

    angle_rad = angle * 3.1415 / 180.0

    DCM = np.zeros((3,3))
    DCM[0,0] = np.cos(angle_rad)
    DCM[0,1] = np.sin(angle_rad)
    DCM[1,0] = -np.sin(angle_rad)
    DCM[1,1] = np.cos(angle_rad)
    DCM[2,2] = 1

    y_bl, x_bl, _, _ = utm.from_latlon(bl_loc[0], bl_loc[1])
    pose_bl_utm = np.array( [x_bl, y_bl, -bl_loc[2]] )

    poses = []
    for loc in locs:
        y, x, _, _ = utm.from_latlon(loc[0], loc[1])
        pose_utm = [x,y,-loc[2]]
        pose = np.matmul(DCM,pose_utm-pose_bl_utm)
        poses.append(pose)

    poses = np.array(poses)
    return poses

def gps_to_ned(ref_loc, locs):
    y_ref, x_ref, _, _ = utm.from_latlon(ref_loc[0], ref_loc[1])
    pose_ref_utm = np.array( [x_ref, y_ref, -ref_loc[2]] )

    poses_ned = []
    for loc in locs:
        y, x, _, _ = utm.from_latlon(loc[0], loc[1])
        pose_utm = [x,y,-loc[2]]
        pose_ned = pose_utm-pose_ref_utm
        poses_ned.append(pose_ned)

    poses_ned = np.array(poses_ned)
    return poses_ned

def get_dji_raw_imgs(imgsdir):

    files = [f for f in listdir(imgsdir) if isfile(join(imgsdir, f))]
    imgs = []
    for file in files:
        if file.endswith(".jpg"):
            imgs.append(file)

    names = [img.split('.')[0] for img in imgs]
    nums = [int( name[5:] ) for name in names]
    idxs = np.argsort(nums)

    imgs = np.array(imgs)[idxs]
    img_dirs = [imgsdir + "/" + img for img in imgs]

    return img_dirs

def get_dataset_imgs(imgsdir):

    files = [f for f in listdir(imgsdir) if isfile(join(imgsdir, f))]
    imgs = []
    for file in files:
        if file.endswith(".jpg"):
            imgs.append(file)

    names = [img.split('.')[0] for img in imgs]
    nums = [int( name[:] ) for name in names]
    idxs = np.argsort(nums)

    imgs = np.array(imgs)[idxs]
    img_dirs = [imgsdir + "/" + img for img in imgs]

    return img_dirs

def check_time_overlap(times_1, times_2):
    dtime_0 = times_1[0]
    dtime_1 = times_1[-1]
    gtime_0 = times_2[0]
    gtime_1 = times_2[-1]

    x = range(int(dtime_0), int(dtime_1))
    y = range(int(gtime_0), int(gtime_1))
    xs = set(x)
    res = xs.intersection(y)

    return not len(res) == 0

def make_DCM(eul):

    phi = eul[0]
    theta = eul[1]
    psi = eul[2]

    DCM = np.zeros((3,3))
    DCM[0,0] = np.cos(psi)*np.cos(theta)
    DCM[0,1] = np.sin(psi)*np.cos(theta)
    DCM[0,2] = -np.sin(theta)
    DCM[1,0] = np.cos(psi)*np.sin(theta)*np.sin(phi)-np.sin(psi)*np.cos(phi)
    DCM[1,1] = np.sin(psi)*np.sin(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi)
    DCM[1,2] = np.cos(theta)*np.sin(phi)
    DCM[2,0] = np.cos(psi)*np.sin(theta)*np.cos(phi)+np.sin(psi)*np.sin(phi)
    DCM[2,1] = np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi)
    DCM[2,2] = np.cos(theta)*np.cos(phi)

    return DCM
