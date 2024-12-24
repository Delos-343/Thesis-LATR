from mmcv.utils import Config
# from mmengine.config import Config
from utils.utils import *
from experiments.ddp import *
from experiments.runner import *
from argparse import Namespace

def test_openlane_lite():
    args = Namespace(
        cfg_options=None,
        config="config/release_iccv/latr_1000_baseline_lite.py",
        distributed=False,  # Set to False for single GPU
        gpu=0,
        local_rank=0,
        nodes=1,
        use_slurm=False,
        world_size=1
    )
    
    # Set your desired values for args here
    args.config = "config/release_iccv/latr_1000_baseline_lite.py"
    # cfg_options_list = ['evaluate=true', 'eval_ckpt=pretrained_models/openlane_lite.pth']
    cfg_options_list = ['evaluate=true', 'eval_ckpt=pretrained_models/openlane_lite.pth']
    # Convert the list of options to a dictionary
    cfg_options_dict = {}
    for option in cfg_options_list:
        key, value = option.split('=')
        cfg_options_dict[key] = value
    
    args.cfg_options = cfg_options_dict
    # Load configuration from the specified file
    cfg = Config.fromfile(args.config)
    
    # Merge any additional configuration options provided through --cfg-options
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    ddp_init(args)
    cfg.merge_from_dict(vars(args))
    # Create a runner and set the configuration
    runner = Runner(cfg)
    
    # Test Openlane (lite version)
    runner.eval()

def train_openlane_lite():
    args = Namespace(
        cfg_options=None,
        config="config/release_iccv/latr_1000_baseline_lite.py",
        distributed=False,
        gpu=0,
        local_rank=0,
        nodes=1,
        use_slurm=False,
        world_size=1
    )

    # Set your desired values for args here
    args.config = "config/release_iccv/latr_1000_baseline_lite.py"
    cfg_options_list = []  # You can add training-specific options here

    # Convert the list of options to a dictionary
    cfg_options_dict = {}
    for option in cfg_options_list:
        key, value = option.split('=')
        cfg_options_dict[key] = value

    args.cfg_options = cfg_options_dict
    # Load configuration from the specified file
    cfg = Config.fromfile(args.config)

    # Merge any additional configuration options provided through --cfg-options
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
       
    ddp_init(args)
    cfg.merge_from_dict(vars(args))
    # Create a runner and set the configuration
    runner = Runner(cfg)

    # Train Openlane (lite version)
    runner.train()
    
def train_openlane_base():
    args = Namespace(
        cfg_options=None,
        config="config/release_iccv/latr_1000_baseline.py",
        distributed=False,
        gpu=0,
        local_rank=0,
        nodes=1,
        use_slurm=False,
        world_size=1
    )

    # Set your desired values for args here
    args.config = "config/release_iccv/latr_1000_baseline.py"
    cfg_options_list = []  # You can add training-specific options here

    # Convert the list of options to a dictionary
    cfg_options_dict = {}
    for option in cfg_options_list:
        key, value = option.split('=')
        cfg_options_dict[key] = value

    args.cfg_options = cfg_options_dict
    # Load configuration from the specified file
    cfg = Config.fromfile(args.config)

    # Merge any additional configuration options provided through --cfg-options
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
       
    ddp_init(args)
    cfg.merge_from_dict(vars(args))
    # Create a runner and set the configuration
    runner = Runner(cfg)

    # Train Openlane (lite version)
    runner.train()
    
if __name__ == '__main__':
    # train_openlane_base()
    train_openlane_lite()
    # test_openlane_lite()