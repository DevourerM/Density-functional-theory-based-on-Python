import json, os, solve
try:
    with open("INPUT.json", "r", encoding="utf-8") as f: config = json.load(f)
    config.update({"project_name":"diamond_repeat_smoke","show_plot":False,"show_density_evolution":False,"show_progress":False,"save_result_json":False,"save_density_npy":False,"save_density_plot":True,"visualization_repeat_count":2,"visualization_repeat_max":3,"scf_max_iterations":1})
    with open("__repeat_smoke_input.json", "w", encoding="utf-8") as f: json.dump(config, f)
    solve.main(["__repeat_smoke_input.json"])
    filename = "diamond_repeat_smoke_density_3d.png"
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"SUCCESS: {filename} exists, size={size} bytes")
    else:
        print(f"FAILURE: {filename} not found")
except Exception as e:
    import traceback
    print(f"EXCEPTION: {str(e)}")
    traceback.print_exc()
finally:
    for f in ["__repeat_smoke_input.json", "diamond_repeat_smoke_density_3d.png"]:
        if os.path.exists(f): os.remove(f)
