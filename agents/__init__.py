from agents.trl import TRLAgent
from agents.crl import CRLAgent
from agents.qrl import QRLAgent
from agents.gcfbc import GCFBCAgent
from agents.gciql import GCIQLAgent

agents = dict(
    trl=TRLAgent,
    crl=CRLAgent,
    qrl=QRLAgent,
    gcfbc=GCFBCAgent,
    gciql=GCIQLAgent,
)
