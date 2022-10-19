from reservoirpy.nodes import Reservoir, Ridge, IPReservoir, FORCE, LMS, RLS, Input
from reservoirpy.utils import verbosity
verbosity(0)

def createESN(**params):
    ridge = params['ridge']
    del params['ridge']
    params['units'] = int(params['units'])
    reservoir = Reservoir(**params)
    readout = Ridge(output_dim=10, ridge=ridge)
    esn = reservoir >> readout
    return esn

def createIPESN(**params):
    readout = Ridge(output_dim=10, ridge=params['ridge'])
    del params['ridge']
    params['units'] = int(params['units'])
    reservoir = IPReservoir(**params)
    esn = reservoir >> readout
    return esn

def createESNForce(**params):
    readout = FORCE(output_dim=10, alpha=params['alpha'])
    del params['alpha']
    params['units'] = int(params['units'])
    reservoir = Reservoir(**params)
    esn = reservoir >> readout
    return esn

def createIPESNForce(**params):
    readout = FORCE(output_dim=10, alpha=params['alpha'])
    del params['alpha']
    params['units'] = int(params['units'])
    reservoir = IPReservoir(**params)
    esn = reservoir >> readout
    return esn

def createESNLms(**params):
    readout = LMS(output_dim=10, alpha=params['alpha'])
    del params['alpha']
    params['units'] = int(params['units'])
    reservoir = Reservoir(**params)
    esn = reservoir >> readout
    return esn
    
def createESNRls(**params):
    readout = RLS(output_dim=10, alpha=params['alpha'])
    del params['alpha']
    params['units'] = int(params['units'])
    reservoir = Reservoir(**params)
    esn = reservoir >> readout
    return esn

def createIPESNRls(**params):
    readout = RLS(output_dim=10, alpha=params['alpha'])
    del params['alpha']
    params['units'] = int(params['units'])
    reservoir = IPReservoir(**params)
    esn = reservoir >> readout
    return esn

def createReservoir(**params):
    params['units'] = int(params['units'])
    reservoir = Reservoir(**params)
    return reservoir

def createIPReservoir(**params):
    params['units'] = int(params['units'])
    reservoir = IPReservoir(**params)
    return reservoir