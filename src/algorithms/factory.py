from algorithms.drq_no_next_obs import DrQ_No_Next_Obs
from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.transfer import Transfer
from algorithms.drq_rad import DrQ_RAD

algorithm = {
    "sac": SAC,
    "rad": RAD,
    "curl": CURL,
    "pad": PAD,
    "soda": SODA,
    "drq": DrQ,
    "svea": SVEA,
    "transfer": Transfer,
    "drq_rad": DrQ_RAD,
    "drq_no_next_obs": DrQ_No_Next_Obs,
}

transfer_algorithm = {}


def make_agent(obs_shape, action_shape, args):
    return algorithm[args.algorithm](obs_shape, action_shape, args)


def make_transfer_agent(obs_shape, action_shape, args):
    pass
    # return transfer_algorithm[args.]
