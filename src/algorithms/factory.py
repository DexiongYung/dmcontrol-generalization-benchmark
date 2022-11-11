from algorithms.alix import ALIX
from algorithms.curriculum_learning.augcl4 import AugCL4
from algorithms.curriculum_learning.secant import SECANT
from algorithms.data_mix import Data_Mix
from algorithms.non_naive_drq import Non_Naive_DrQ
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
from algorithms.svea2 import SVEA2
from algorithms.transfer import Transfer
from algorithms.curriculum_learning.curriculum import Curriculum
from algorithms.curriculum_learning.curriculum_double import Curriculum_Double
from algorithms.non_naive_rad import NonNaiveRAD
from algorithms.sac_policy_eval import SAC_policy_eval
from algorithms.SARSA_policy_eval import SARSA_policy_eval
from algorithms.curriculum_learning.curriculum_FTL import Curriculum_FTL
from algorithms.curriculum_learning.curriculum_resample import Curriculum_Resample
from algorithms.curriculum_learning.curriculum_2x_nu_actor import Curriculum_2x_Nu_Actor
from algorithms.curriculum_learning.curriculum_2x_opt import Curriculum_2x_Opt
from algorithms.curriculum_learning.curriculum_bb import Curriculum_BB
from algorithms.curriculum_learning.curriculum_single import CurriculumSingle
from algorithms.curriculum_learning.curriculum_fresh import CurriculumFresh
from algorithms.curriculum_learning.augcl2 import AugCL2

algorithm = {
    "sac": SAC,
    "rad": RAD,
    "curl": CURL,
    "pad": PAD,
    "soda": SODA,
    "drq": DrQ,
    "svea": SVEA,
    "transfer": Transfer,
    "curriculum": Curriculum,
    "non_naive_rad": NonNaiveRAD,
    "2x_curriculum": Curriculum_Double,
    "sac_policy_eval": SAC_policy_eval,
    "sarsa_policy_eval": SARSA_policy_eval,
    "curriculum_FTL": Curriculum_FTL,
    "curriculum_resample": Curriculum_Resample,
    "2x_curriculum_nu_actor": Curriculum_2x_Nu_Actor,
    "2x_curriculum_opt": Curriculum_2x_Opt,
    "2x_curriculum_bb": Curriculum_BB,
    "curriculum_single": CurriculumSingle,
    "2x_fresh": CurriculumFresh,
    "secant": SECANT,
    "augcl2": AugCL2,
    "svea2": SVEA2,
    "rad2": RAD2,
    "non_naive_rad2": NonNaiveRAD2,
    "alix": ALIX,
    "rad_alix": RAD_ALIX,
    "rad_shift": RAD_shift,
    "non_naive_drq": Non_Naive_DrQ,
    "augcl4": AugCL4,
}


def make_agent(obs_shape, action_shape, args):
    return algorithm[args.algorithm](obs_shape, action_shape, args)
