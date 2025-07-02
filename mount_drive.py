import os
import shutil
import re
import subprocess

# 1. Mount Google Drive (Colab only)
# NOTE: Mount Google Drive manually in a Colab notebook cell, not in this script.
print("[INFO] Skipping Google Drive mount. Please run the following in a Colab cell if needed:")
print("from google.colab import drive\ndrive.mount('/content/drive')")

# 2. Clone the DEKR repo (removes any previous copy)
if not os.path.exists('/content/DEKR_anisotropy'):
    subprocess.run("git clone -b coco-only https://github.com/PhilipKone/DEKR_anisotropy.git", shell=True, check=True)

# 3. Remove version specifiers for torch, torchvision, numpy in requirements.txt
req_path = '/content/DEKR_anisotropy/requirements.txt'
if os.path.exists(req_path):
    with open(req_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        line = re.sub(r'^(torch|torchvision|numpy)==[^\s]+', r'\1', line)
        new_lines.append(line)
    with open(req_path, 'w') as f:
        f.writelines(new_lines)
    print("Updated requirements.txt:")
    subprocess.run(f"cat {req_path}", shell=True)
else:
    print("requirements.txt not found.")

# 4. Install requirements
subprocess.run("pip install -r /content/DEKR_anisotropy/requirements.txt", shell=True, check=True)

# 5. Unzip and arrange COCO validation images and annotations
subprocess.run("unzip /content/drive/MyDrive/HPE/coco_data.zip -d /content/", shell=True, check=True)
os.makedirs('/content/DEKR/data/coco/images', exist_ok=True)
os.makedirs('/content/DEKR/data/coco/annotations', exist_ok=True)
shutil.copytree('/content/coco_data/val2017', '/content/DEKR/data/coco/images/val2017', dirs_exist_ok=True)
shutil.copy('/content/coco_data/annotations_trainval2017/annotations/person_keypoints_val2017.json',
            '/content/DEKR_anisotropy/data/coco/annotations/person_keypoints_val2017.json')

# 6. Unzip and arrange pretrained model weights
subprocess.run('unzip "/content/drive/MyDrive/HPE/model (1).zip" -d /content/DEKR/', shell=True, check=True)
os.makedirs('/content/DEKR_anisotropy/model/pose_coco/', exist_ok=True)
shutil.copy('/content/DEKR/model/pose_coco/pose_dekr_hrnetw32_coco.pth',
            '/content/DEKR_anisotropy/model/pose_coco/pose_dekr_hrnetw32_coco.pth')

# 7. Patch deprecated np.float usage
subprocess.run('sed -i "s/np.float/float/g" /content/DEKR_anisotropy/lib/utils/rescore.py', shell=True)
subprocess.run('sed -i "s/np.float/float/g" /content/DEKR_anisotropy/lib/dataset/COCODataset.py', shell=True)

# 8. Patch out crowdposetools and CrowdPose-specific code for COCO-only
subprocess.run('sed -i "s/^from crowdposetools/# from crowdposetools/g" /content/DEKR_anisotropy/lib/utils/rescore.py', shell=True)
subprocess.run('sed -i "s/^from crowdposetools/# from crowdposetools/g" /content/DEKR_anisotropy/lib/dataset/CrowdPoseDataset.py', shell=True)
subprocess.run('sed -i "s/class CrowdRescoreEval(CrowdposeEval):/class CrowdRescoreEval:/" /content/DEKR_anisotropy/lib/utils/rescore.py', shell=True)
subprocess.run('sed -i "s/^coco_eval =.*/coco_eval = None/" /content/DEKR_anisotropy/lib/dataset/CrowdPoseDataset.py', shell=True)

# 9. Patch valid.py to skip rescoring if the model file is missing
valid_path = '/content/DEKR_anisotropy/tools/valid.py'
if os.path.exists(valid_path):
    with open(valid_path, 'r') as f:
        lines = f.readlines()
    if not any(line.strip() == 'import os' for line in lines):
        for i, line in enumerate(lines):
            if line.startswith('import') or line.startswith('from'):
                continue
            else:
                lines.insert(i, 'import os\n')
                break
    new_lines = []
    rescoring_patched = False
    for line in lines:
        if not rescoring_patched and 'scores = rescore_valid(cfg, final_poses, scores)' in line:
            # Detect indentation of the current line
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(f'{indent}if os.path.exists(cfg.RESCORE.MODEL_FILE):\n')
            new_lines.append(f'{indent}    scores = rescore_valid(cfg, final_poses, scores)\n')
            new_lines.append(f'{indent}else:\n')
            new_lines.append(f'{indent}    print("Rescore model not found, skipping rescoring.")\n')
            rescoring_patched = True
        elif rescoring_patched and 'scores = rescore_valid(cfg, final_poses, scores)' in line:
            continue
        else:
            new_lines.append(line)
    with open(valid_path, 'w') as f:
        f.writelines(new_lines)
    print("Patched valid.py: rescoring will be skipped if the model file is missing.")

print("Setup complete. You can now run validation using the DEKR scripts.")
