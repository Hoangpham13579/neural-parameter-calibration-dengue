from enum import IntEnum

# Agent kinds and their order used in the SEIRD+ model


class Compartments(IntEnum):    
    CumulativeHumanInfected = 0
    HumanInfected           = 1
    MosquitoExposed         = 2
    HumanRecovered          = 3
    HumanSusceptible        = 4
    HumanExposed            = 5
    MosquitoSusceptible     = 6
    MosquitoInfected        = 7