from algorithms.alix import ALIX
from algorithms.non_naive_rad2 import NonNaiveRAD2
from algorithms.rad2 import RAD2
from algorithms.rad_alix import RAD_ALIX
from algorithms.rad_shift import RAD_shift
from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.transfer import Transfer
from algorithms.curriculum_learning.AugCL import AugCL
from algorithms.non_naive_rad import NonNaiveRAD
from algorithms.curriculum_learning.curriculum_FTL import Curriculum_FTL
from algorithms.curriculum_learning.curriculum_single import CurriculumSingle
from algorithms.curriculum_learning.curriculum_fresh import CurriculumFresh

algorithm = {
    "sac": SAC,
    "rad": RAD,
    "curl": CURL,
    "pad": PAD,
    "soda": SODA,
    "drq": DrQ,
    "svea": SVEA,
    "transfer": Transfer,
    "non_naive_rad": NonNaiveRAD,
    "augcl": AugCL,
    "curriculum_FTL": Curriculum_FTL,
    "curriculum_single": CurriculumSingle,
    "2x_fresh": CurriculumFresh,
    "rad2": RAD2,
    "non_naive_rad2": NonNaiveRAD2,
    "alix": ALIX,
    "rad_alix": RAD_ALIX,
    "rad_shift": RAD_shift,
}


def make_agent(obs_shape, action_shape, args):
    return algorithm[args.algorithm](obs_shape, action_shape, args)
