from algorithms.alix import ALIX
from algorithms.drq3 import DrQ3
from algorithms.drq4 import DrQ4
from algorithms.drq5 import DrQ5
from algorithms.drq6 import DrQ6
from algorithms.non_naive_rad2 import NonNaiveRAD2
from algorithms.rad2 import RAD2
from algorithms.rad_alix import RAD_ALIX
from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.svea2 import SVEA2
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
    "svea2": SVEA2,
    "drq3": DrQ3,
    "drq4": DrQ4,
    "drq5": DrQ5,
    "drq6": DrQ6,
}


def make_agent(obs_shape, action_shape, args):
    return algorithm[args.algorithm](obs_shape, action_shape, args)
