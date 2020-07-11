from yacs.config import CfgNode as CN


_C = CN()

# needed by trainer
_C.name = 'default'
_C.checkpoints_path = '../data/checkpoints/'
_C.results_path = '../data/results/'
_C.learning_rate = 1e-3
_C.weight_decay = 0.0
_C.momentum = 0.0
_C.optim = 'RMSprop'
_C.schedule = [40, 60]
_C.gamma = 0.1
_C.resume = False 

# needed by train()
_C.batch_size = 4
_C.num_threads = 4
_C.num_epoch = 100
_C.freq_plot = 10
_C.freq_save = 100
_C.freq_eval = 100
_C.freq_vis = 100


# --- netG options ---
_C.netG = CN()
_C.netG.mean = (0.5, 0.5, 0.5)
_C.netG.std = (0.5, 0.5, 0.5)
_C.netG.ckpt_path = ''
_C.netG.projection = 'orthogonal' 

# --- netG:backbone options ---
_C.netG.backbone = CN()
_C.netG.backbone.IMF = 'PIFuHGFilters'

# --- netG:normalizer options ---
_C.netG.normalizer = CN()
_C.netG.normalizer.IMF = 'PIFuNomalizer'
_C.netG.normalizer.soft_onehot = False
_C.netG.normalizer.soft_dim = 64

# --- netG:head options ---
_C.netG.head = CN()
_C.netG.head.IMF = 'PIFuNetGMLP'

# --- netG:loss options ---
_C.netG.loss = CN()
_C.netG.loss.IMF = 'MSE'


# --- netC options ---
_C.netC = CN()
_C.netC.mean = (0.5, 0.5, 0.5)
_C.netC.std = (0.5, 0.5, 0.5)
_C.netC.ckpt_path = ''
_C.netC.projection = 'orthogonal' 

# --- netC:backbone options ---
_C.netC.backbone = CN()
_C.netC.backbone.IMF = 'PIFuResBlkFilters'

# --- netC:normalizer options ---
_C.netC.normalizer = CN()
_C.netC.normalizer.IMF = 'PIFuNomalizer'
_C.netC.normalizer.soft_onehot = False
_C.netC.normalizer.soft_dim = 64

# --- netC:head options ---
_C.netC.head = CN()
_C.netC.head.IMF = 'PIFuNetCMLP'

# --- netC:loss options ---
_C.netC.loss = CN()
_C.netC.loss.IMF = 'L1'


# --- dataset options ---
_C.dataset = CN()
_C.dataset.aug_bri = 0.4
_C.dataset.aug_con = 0.4
_C.dataset.aug_sat = 0.4
_C.dataset.aug_hue = 0.0
_C.dataset.blur = 1.0
_C.dataset.num_sample_geo = 5000
_C.dataset.num_sample_color = 0
_C.dataset.sigma_geo = 0.05
_C.dataset.sigma_color = 0.001
_C.dataset.pre_load = False


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`

# cfg = get_cfg_defaults()
# cfg.merge_from_file('../configs/example.yaml')

# # Now override from a list (opts could come from the command line)
# opts = ['dataset.root', '../data/XXXX', 'learning_rate', '1e-2']
# cfg.merge_from_list(opts)

