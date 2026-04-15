from .mas_base import MAS
from .cot import CoT
from .agentverse import AgentVerse_HumanEval, AgentVerse_MGSM, AgentVerse_Main
from .llm_debate import LLM_Debate_Main
from .dylan import DyLAN_HumanEval, DyLAN_MATH, DyLAN_MMLU, DyLAN_Main
from .autogen import AutoGen_Main
from .camel import CAMEL_Main
from .evomac import EvoMAC_Main
# from .chatdev import ChatDev_SRDD
from .macnet import MacNet_Main, MacNet_SRDD
from .mad import MAD_Main
from .mapcoder import MapCoder_HumanEval, MapCoder_MBPP
from .self_consistency import SelfConsistency
from .mav import MAV_GPQA, MAV_HumanEval, MAV_Main, MAV_MATH, MAV_MMLU
from .selforg import SelfOrg_Main
from .selforg.selforg_no_debate import SelfOrg_NoDebate
from .selforg.selforg_random_graph import SelfOrg_RandomGraph
from .soo import SOO_Main
from .soo_centered import SOO_Centered_Main
from .soo_centered_v2 import SOO_Centered_v2_Main
from .soo_math import SOO_Math_Main
from .soo_pu import SOO_pu_Main, SOO_pu_pro_Main
from .sparc import SPARC_Math_Main
from .sparc_v2 import SPARC_v2_Math_Main
from .arena import ARENA_Math_Main
from .evo import EVO_Math_Main
from .cascade import CASCADE_Math_Main
from .soo_pu_final import SOO_pu_final_Main, SOO_pu_final_MATH, SOO_pu_final_soo_pu_Main
from .h_swarm import HSwarm_Main, HSwarm_MultiObj_Main

method2class = {
    "vanilla": MAS,
    "cot": CoT,
    "agentverse_humaneval": AgentVerse_HumanEval,
    "agentverse_mgsm": AgentVerse_MGSM,
    "agentverse": AgentVerse_Main,
    "llm_debate": LLM_Debate_Main,
    "dylan_humaneval": DyLAN_HumanEval,
    "dylan_math": DyLAN_MATH,
    "dylan_mmlu": DyLAN_MMLU,
    "dylan": DyLAN_Main,
    "autogen": AutoGen_Main,
    "camel": CAMEL_Main,
    "evomac": EvoMAC_Main,
    # "chatdev_srdd": ChatDev_SRDD,
    "macnet": MacNet_Main,
    "macnet_srdd": MacNet_SRDD,
    "mad": MAD_Main,
    "mapcoder_humaneval": MapCoder_HumanEval,
    "mapcoder_mbpp": MapCoder_MBPP,
    "self_consistency": SelfConsistency,
    "mav_gpqa": MAV_GPQA,
    "mav_humaneval": MAV_HumanEval,
    "mav_main": MAV_Main,
    "mav_math": MAV_MATH,
    "mav_mmlu": MAV_MMLU,
    "selforg": SelfOrg_Main,
    "selforg_no_debate": SelfOrg_NoDebate,
    "selforg_random_graph": SelfOrg_RandomGraph,
    "soo": SOO_Main,
    "soo_centered": SOO_Centered_Main,
    "soo_centered_v2": SOO_Centered_v2_Main,
    "soo_pu": SOO_pu_Main,
    "soo_pu_pro": SOO_pu_pro_Main,
    "soo_math": SOO_Math_Main,
    "sparc_math": SPARC_Math_Main,
    "arena_math": ARENA_Math_Main,
    "evo_math": EVO_Math_Main,
    "cascade_math": CASCADE_Math_Main,
    "sparc_v2_math": SPARC_v2_Math_Main,
    "soo_pu_final": SOO_pu_final_Main,
    "soo_pu_final_math": SOO_pu_final_MATH,
    "soo_pu_final_soo_pu": SOO_pu_final_soo_pu_Main,
    "h_swarm": HSwarm_Main,
    "h_swarm_multiobj": HSwarm_MultiObj_Main,
}

def get_method_class(method_name, dataset_name=None):
    
    # lowercase the method name
    method_name = method_name.lower()
    
    all_method_names = method2class.keys()
    matched_method_names = [sample_method_name for sample_method_name in all_method_names if method_name in sample_method_name]
    
    if len(matched_method_names) > 0:
        if dataset_name is not None:
            # lowercase the dataset name
            dataset_name = dataset_name.lower()
            # check if there are method names that contain the dataset name
            matched_method_data_names = [sample_method_name for sample_method_name in matched_method_names if sample_method_name.split('_')[-1] in dataset_name]
            if len(matched_method_data_names) > 0:
                method_name = matched_method_data_names[0]
                if len(matched_method_data_names) > 1:
                    print(f"[WARNING] Found multiple methods matching {dataset_name}: {matched_method_data_names}. Using {method_name} instead.")
    else:
        raise ValueError(f"[ERROR] No method found matching {method_name}. Please check the method name.")
    
    return method2class[method_name]