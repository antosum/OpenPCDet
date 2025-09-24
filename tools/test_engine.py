from pcdet_engine import PCDetEngine
from rich import print
from rich.pretty import Pretty


eng = PCDetEngine(
    cfg_file="~/projects/OpenPCDet/tools/cfgs/geminai_models/geminai_cbgs_dyn_pp_centerpoint.yaml",
    ckpt_path="~/projects/OpenPCDet/output/geminai_models/geminai_cbgs_dyn_pp_centerpoint/vital-cosmos-46/ckpt/best_model.pth",
    device="cuda",
)

print("CFG\n",Pretty(eng.cfg))
print("Data_CONFIG\n",Pretty(eng.data_config))
print("Class names \n",Pretty(eng.class_names))