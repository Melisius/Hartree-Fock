import numpy as np
import copy
import time
from slowquant.molecularintegrals.runMIcython import IntegralsNumba
from slowquant.molecularintegrals.MIpython import nucrep

##CALC OF INTEGRALS
def runIntegrals(molecule, basis, settings, results):
    # Nuclear-nuclear repulsion
    VNN = np.zeros(1)
    VNN[0] = nucrep(molecule)
    
    # Reform basisset information
    
    # basisidx [number of primitives, start index in basisfloat and basisint]
    # basisfloat array of float values for basis, N, zeta, c, x, y, z 
    # basisint array of integer values for basis, l, m, n
    basisidx   = np.zeros((len(basis),2))
    startidx = 0
    for i in range(0, len(basisidx)):
        basisidx[i,0] = basis[i][4]
        basisidx[i,1] = startidx
        startidx     += basisidx[i,0]
    basisidx = basisidx.astype(np.int64)
    
    basisfloat = np.zeros((np.sum(basisidx[:,0]),6))
    basisint   = np.zeros((np.sum(basisidx[:,0]),3))
    
    idxfi = 0
    for i in range(0, len(basisidx)):
        for j in range(0, basis[i][4]):
            basisfloat[idxfi,0] = basis[i][5][j][0]
            basisfloat[idxfi,1] = basis[i][5][j][1]
            basisfloat[idxfi,2] = basis[i][5][j][2]
            basisfloat[idxfi,3] = basis[i][1]
            basisfloat[idxfi,4] = basis[i][2]
            basisfloat[idxfi,5] = basis[i][3]
            
            basisint[idxfi,0]  = basis[i][5][j][3]
            basisint[idxfi,1]  = basis[i][5][j][4]
            basisint[idxfi,2]  = basis[i][5][j][5]
            
            idxfi += 1
    
    basisint = basisint.astype(np.int64)
    basisidx = basisidx.astype(np.int64)
    
    Na, S, T, ERI = IntegralsNumba(basisidx, basisfloat, basisint, molecule)

    results['VNN'] = VNN
    results['VNe'] = Na
    results['S']   = S
    results['Te']  = T
    results['Vee'] = ERI
    return results

