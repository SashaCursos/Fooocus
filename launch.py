import os
import sys
import ssl

print('[System ARGV] ' + str(sys.argv))

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["GRADIO_SERVER_PORT"] = "7865"

ssl._create_default_https_context = ssl._create_unverified_context


import platform
import fooocus_version

from build_launcher import build_launcher
from modules.launch_util import is_installed, run, python, run_pip, requirements_met
from modules.model_loader import load_file_from_url
from modules.config import path_checkpoints, path_loras, path_vae_approx, path_fooocus_expansion, \
    checkpoint_downloads, path_embeddings, embeddings_downloads, lora_downloads


REINSTALL_ALL = False
TRY_INSTALL_XFORMERS = False


def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")
    torch_command = os.environ.get('TORCH_COMMAND',
                                   f"pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    print(f"Python {sys.version}")
    print(f"Fooocus version: {fooocus_version.version}")

    if REINSTALL_ALL or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if TRY_INSTALL_XFORMERS:
        if REINSTALL_ALL or not is_installed("xformers"):
            xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.20')
            if platform.system() == "Windows":
                if platform.python_version().startswith("3.10"):
                    run_pip(f"install -U -I --no-deps {xformers_package}", "xformers", live=True)
                else:
                    print("Installation of xformers is not supported in this version of Python.")
                    print(
                        "You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                    if not is_installed("xformers"):
                        exit(0)
            elif platform.system() == "Linux":
                run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")

    if REINSTALL_ALL or not requirements_met(requirements_file):
        run_pip(f"install -r \"{requirements_file}\"", "requirements")

    return


vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v3.1.safetensors',
     'https://huggingface.co/lllyasviel/misc/resolve/main/xl-to-v1_interposer-v3.1.safetensors')
]


def download_models():
    for file_name, url in checkpoint_downloads.items():
        load_file_from_url(url=url, model_dir=path_checkpoints, file_name=file_name)
    for file_name, url in embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=path_embeddings, file_name=file_name)
    for file_name, url in lora_downloads.items():
        load_file_from_url(url=url, model_dir=path_loras, file_name=file_name)
    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=path_vae_approx, file_name=file_name)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=path_fooocus_expansion,
        file_name='pytorch_model.bin'
    )

#CHECKPOINTS XL (ADDED IN THE CONFIG.PY)

    #load_file_from_url(
        #url='https://huggingface.co/SashaCursos/fav_models/resolve/main/fav/sdxxxl_v30.safetensors?download=true',
        #url='https://civitai.com/api/download/models/253250',
        #model_dir=path_checkpoints,
        #file_name='SDXXXL.safetensors'
    #)


    #load_file_from_url(
        #url='https://huggingface.co/SashaCursos/fav_models/resolve/main/fav/dreamshaperXL_turboDpmppSDE.safetensors?download=true',
        #url='https://civitai.com/api/download/models/251662',
        #model_dir=path_checkpoints,
        #file_name='DreamShaperXL.safetensors'
    #)


#REFINERS1.5

    load_file_from_url(
        url='https://huggingface.co/SashaCursos/fav_models/resolve/main/fav/lazymixRealAmateur_v40.safetensors',
        #url='https://civitai.com/api/download/models/300972',
        model_dir=path_checkpoints,
        file_name='LazyMix.safetensors'
    )

    #load_file_from_url(
        #url='https://huggingface.co/SashaCursos/fav_models/resolve/main/fav/realisticVisionV60B1_v60B1VAE.safetensors',
        #url='https://civitai.com/api/download/models/245598?type=Model&format=SafeTensor&size=full&fp=fp16',
        #model_dir=path_checkpoints,
        #file_name='RealisticVision6.safetensors'
    #)

#LORAS    
    #load_file_from_url(
        #url='https://civitai.com/api/download/models/280811?type=Model&format=SafeTensor',
        #model_dir=path_loras,
        #file_name='FappXL.safetensors'
    #)
    
    #load_file_from_url(
        #url='https://civitai.com/api/download/models/280241?type=Model&format=SafeTensor',
        #model_dir=path_loras,
        #file_name='FingeringSDXL.safetensors'
    #)
    
    #load_file_from_url(
        #url='https://civitai.com/api/download/models/275493',
        #model_dir=path_loras,
        #file_name='PussyLegsTitsPose.safetensors'
    #)

    load_file_from_url(
        url='https://civitai.com/api/download/models/160240?type=Model&format=SafeTensor',
        model_dir=path_loras,
        file_name='NSFWPOVAllInOneSDXL.safetensors'
    )

    #load_file_from_url(
        #url='https://civitai.com/api/download/models/131645?type=Model&format=SafeTensor',
        #model_dir=path_loras,
        #file_name='GapeThyPussySDXL.safetensors'
    #)

    #load_file_from_url(
        #url='https://civitai.com/api/download/models/140555',
        #model_dir=path_loras,
        #file_name='POVMissionarySDXL.safetensors'
    #)

    #load_file_from_url(
        #url='https://civitai.com/api/download/models/137178',
        #model_dir=path_loras,
        #file_name='POVDoggystyleSDXL.safetensors'
    #)

    #load_file_from_url(
        #url='https://civitai.com/api/download/models/138784',
        #model_dir=path_loras,
        #file_name='POVCowgirlSDXL.safetensors'
    #)

    #load_file_from_url(
        #url='https://civitai.com/api/download/models/285434?type=Model&format=SafeTensor',
        #model_dir=path_loras,
        #file_name='SpreadAssSDXL.safetensors'
    #)

    #load_file_from_url(
        #url='https://civitai.com/api/download/models/141832',
        #model_dir=path_loras,
        #file_name='POVReverseCowgirlSDXL.safetensors'
    #)

    #load_file_from_url(
        #url='https://civitai.com/api/download/models/135068',
        #model_dir=path_loras,
        #file_name='POVBlowjobSDXL.safetensors'
    #)   


    load_file_from_url(
        url='https://civitai.com/api/download/models/132739?type=Model&format=SafeTensor',
        model_dir=path_loras,
        file_name='Onoff.safetensors'
    ) 

    #load_file_from_url(
        #url='https://civitai.com/api/download/models/288679?type=Model&format=SafeTensor',
        #model_dir=path_loras,
        #file_name='iPhoneMirrorSelfie.safetensors'
    #) 

    #load_file_from_url(
        #url='https://civitai.com/api/download/models/197172',
        #model_dir=path_loras,
        #file_name='BrieLarsonSDXL.safetensors'
    #)

    #load_file_from_url(
        #url='https://civitai.com/api/download/models/195753',
        #model_dir=path_loras,
        #file_name='HaileeSteinfeldSDXL.safetensors'
    #)

    #load_file_from_url(
        #url='https://civitai.com/api/download/models/243854',
        #model_dir=path_loras,
        #file_name='ElizabethOlsenSDXL.safetensors'
    #)

    #load_file_from_url(
        #url='https://civitai.com/api/download/models/129681',
        #model_dir=path_loras,
        #file_name='ScarlettJohanssonSDXL.safetensors'
    #)

    #load_file_from_url(
        #url='https://civitai.com/api/download/models/200333',
        #model_dir=path_loras,
        #file_name='MarisaTomeiSDXL.safetensors'
    #)

    #load_file_from_url(
        #url='https://civitai.com/api/download/models/264789?type=Model&format=SafeTensor',
        #model_dir=path_loras,
        #file_name='LegsBehindHeadSDXL.safetensors'
    #)

    #load_file_from_url(
        #url='https://civitai.com/api/download/models/191924?type=Model&format=SafeTensor',
        #model_dir=path_loras,
        #file_name='TitsqueezeSDXL.safetensors'
    #)
    
    
#EMBEDDINGS
    
    #load_file_from_url(
        #url='https://civitai.com/api/download/models/166373?type=Model&format=SafeTensor',
        #model_dir=path_embeddings,
        #file_name='UltimateTextEmbeddingsSDXLPack.safetensors'
    #)
    
    

    return


def ini_args():
    from args_manager import args
    return args


prepare_environment()
build_launcher()
args = ini_args()


if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)


download_models()

from webui import *
