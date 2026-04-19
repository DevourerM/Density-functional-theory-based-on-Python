import json
import os
import subprocess
import tempfile
import sys
import io

def get_key(d, partial):
    for k in d.keys():
        if partial in k:
            return k
    return None

with open('INPUT.json', 'rb') as f:
    data = json.loads(f.read().decode('utf-8'))

numerical_settings_key = get_key(data, '数值设置')
grid_key = get_key(data[numerical_settings_key], '空间网格划分')
scf_control_key = get_key(data, 'SCF控制')
path_settings_key = get_key(data, '路径设置')

data[numerical_settings_key][grid_key]['x方向网格数'] = 4
data[numerical_settings_key][grid_key]['y方向网格数'] = 4
data[numerical_settings_key][grid_key]['z方向网格数'] = 4
data[scf_control_key]['最大迭代次数'] = 2
data[numerical_settings_key]['轨道数'] = 4
data[scf_control_key]['波函数收敛阈值'] = 1e-7
data[scf_control_key]['SCF收敛阈值'] = 1e-12

pseudopotential_path_key = get_key(data[path_settings_key], '势势文件夹路径') or list(data[path_settings_key].keys())[0]
local_pp_path = os.path.abspath('SG15-Version1p0_Pseudopotential/SG15_ONCV_v1.0_upf')
if os.path.exists(local_pp_path):
    data[path_settings_key][pseudopotential_path_key] = local_pp_path.replace('\\', '/')

with tempfile.TemporaryDirectory() as tmpdir:
    input_path = os.path.join(tmpdir, 'INPUT.json')
    log_dir = os.path.join(tmpdir, 'logs')
    output_dir = os.path.join(tmpdir, 'outputs')
    os.makedirs(log_dir)
    os.makedirs(output_dir)

    with open(input_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.abspath('src')
    
    cmd = [
        'd:/python/Density-functional-theory-based-on-Python/.venv/Scripts/python.exe',
        '-m', 'realspace_dft',
        input_path,
        '--log-dir', log_dir,
        '--output-dir', output_dir
    ]
    
    # Run without capture_output=True to avoid encoding issues during capture if needed, 
    # but here we use bytes to be safe.
    result = subprocess.run(cmd, cwd=os.getcwd(), env=env, capture_output=True)

    print(f"Return Code: {result.returncode}")
    
    stdout_text = result.stdout.decode('gbk', errors='replace')
    print("STDOUT (first 10 lines and last 10 lines):")
    stdout_lines = stdout_text.splitlines()
    if len(stdout_lines) <= 20:
        print(stdout_text)
    else:
        print('\n'.join(stdout_lines[:10]))
        print('...')
        print('\n'.join(stdout_lines[-10:]))
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr.decode('gbk', errors='replace'))

    final_density_exists = os.path.exists(os.path.join(output_dir, 'final_density.npz'))
    scf_summary_exists = os.path.exists(os.path.join(output_dir, 'scf_summary.json'))
    
    print(f"final_density.npz generated: {final_density_exists}")
    print(f"scf_summary.json generated: {scf_summary_exists}")

