import json
import os
import subprocess
import tempfile


def get_key(mapping, partial):
    for key in mapping.keys():
        if partial in key:
            return key
    return None


with open("INPUT.json", "rb") as file:
    data = json.loads(file.read().decode("utf-8"))

numerical_settings_key = get_key(data, "数值设置")
grid_key = get_key(data[numerical_settings_key], "空间网格划分")
scf_control_key = get_key(data, "SCF控制")
path_settings_key = get_key(data, "路径设置")


data[numerical_settings_key][grid_key]["x方向网格数"] = 4
data[numerical_settings_key][grid_key]["y方向网格数"] = 4
data[numerical_settings_key][grid_key]["z方向网格数"] = 4
data[scf_control_key]["最大迭代次数"] = 2
data[numerical_settings_key]["轨道数"] = 4
data[scf_control_key]["SCF收敛阈值"] = 1e-6
data[numerical_settings_key]["k点"] = [{"坐标": [0.0, 0.0, 0.0], "权重": 1.0}]
data[numerical_settings_key]["占据设置"] = {
    "方法": "FERMI_DIRAC",
    "展宽哈特里": 0.02,
    "自旋简并度": 2.0,
}

pseudopotential_path_key = get_key(data[path_settings_key], "赝势文件夹路径") or list(data[path_settings_key].keys())[0]
local_pp_path = os.path.abspath("SG15-Version1p0_Pseudopotential/SG15_ONCV_v1.0_upf")
if os.path.exists(local_pp_path):
    data[path_settings_key][pseudopotential_path_key] = local_pp_path.replace("\\", "/")

with tempfile.TemporaryDirectory() as tmpdir:
    input_path = os.path.join(tmpdir, "INPUT.json")
    log_dir = os.path.join(tmpdir, "logs")
    output_dir = os.path.join(tmpdir, "outputs")
    with open(input_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")

    cmd = [
        "d:/python/Density-functional-theory-based-on-Python/.venv/Scripts/python.exe",
        "-m",
        "realspace_dft",
        input_path,
        "--log-dir",
        log_dir,
        "--output-dir",
        output_dir,
    ]

    result = subprocess.run(cmd, cwd=os.getcwd(), env=env, capture_output=True)

    print(f"Return Code: {result.returncode}")

    stdout_text = result.stdout.decode("gbk", errors="replace")
    print("STDOUT:")
    print(stdout_text)

    if result.stderr:
        print("STDERR:")
        print(result.stderr.decode("gbk", errors="replace"))

    print(f"TensorBoard directory exists: {os.path.isdir(log_dir)}")
    print(f"final_density.txt generated: {os.path.exists(os.path.join(output_dir, 'final_density.txt'))}")
