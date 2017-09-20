import numpy as np
import time
import sys
import slowquant.basissets.BasisSet as BS
import slowquant.molecularintegrals.runMolecularIntegrals as MI
import slowquant.hartreefock.runHartreeFock as HF   
import slowquant.properties.runProperties as prop
import slowquant.mollerplesset.runMPn as MP
import slowquant.qfit.Qfit as QF
import slowquant.geometryoptimization.GeometryOptimization as GO
import slowquant.configurationinteraction.runCI as CI
#import slowquant .coupledcluster.runCC as CC
import slowquant.bomd.runBOMD as MD

def run(moleculename, settingsname):
    settings = np.genfromtxt('slowquant/Standardsettings.csv', delimiter = ';', dtype='str')
    set = {}
    for i in range(len(settings)):
        set.update({settings[i][0]:settings[i][1]})
    
    molecule = np.genfromtxt(str(moleculename), delimiter=';')
    results = {}
    
    output = open('out.txt', 'w')
    output.write('User specified settings: \n')
    settings = np.genfromtxt(str(settingsname), delimiter = ';', dtype='str')
    for i in range(len(settings)):
        set[settings[i][0]] = settings[i][1]
        output.write('    '+str(settings[i][0])+'    '+str(settings[i][1])+'\n')
    output.write('\n \n')

    output.write('Inputfile: \n')
    for i in range(0, len(molecule)):
        for j in range(0, 4):
            output.write("   {: 12.8e}".format(molecule[i,j]))
            output.write("\t \t")
        output.write('\n')
    output.write('\n \n')
    output.close()
    
    if set['Initial method'] == 'BOMD':
        results = MD.runBOMD(molecule, set, results)
    
    elif set['Initial method'] == 'UHF':
        basis = BS.bassiset(molecule, set)
        start = time.time()
        results = MI.runIntegrals(molecule, basis, set, results)
        print(time.time()-start, 'INTEGRALS')
        
        start = time.time()
        results = HF.runHartreeFock(molecule, set, results)
        print(time.time()-start, 'UHF')
    
    elif set['Initial method'] == 'HF':
        if set['GeoOpt'] == 'Yes':
            molecule, results = GO.runGO(molecule, set, results)
        
        basis = BS.bassiset(molecule, set)
        
        start = time.time()
        results = MI.runIntegrals(molecule, basis, set, results)
        print(time.time()-start, 'INTEGRALS')
        
        start = time.time()
        results = HF.runHartreeFock(molecule, set, results)
        print(time.time()-start, 'HF')
        
        start = time.time()
        results = prop.runprop(basis, molecule, set, results)
        print(time.time()-start, 'PROPERTIES')
        
        start = time.time()
        results = MP.runMPn(molecule, results, set)
        print(time.time()-start, 'Perturbation')
        
        start = time.time()
        results = QF.runQfit(basis, molecule, set, results)
        print(time.time()-start, 'QFIT')
        
        start = time.time()
        results = CI.runCI(set, results, molecule)
        print(time.time()-start, 'CI')
        
        start = time.time()
        #results = CC.runCC(molecule, set, results)
        print(time.time()-start, 'CC')
        
    return results

    
if __name__ == "__main__":
    total = time.time()
    mol = str(sys.argv[1])
    set = str(sys.argv[2])
    results = run(mol, set)
    print(time.time() - total, 'Execution time')
