import numpy as np
from slowquant.molecularintegrals.MIpython import 
from numba import jit, float64, int32
from numba.types import Tuple

@jit(tuple((float64[:,:], float64[:,:], float64[:,:], float64[:,:,:,:]))(int32[:,:],float64[:,:],int32[:,:],float64[:,:]),nopython=True,cache=True)
def runIntegrals(basisidx, basisfloat, basisint, input):
    # Array to store R values, only created once if created here
    R1buffer = np.zeros((4*np.max(basisint)+1,4*np.max(basisint)+1,4*np.max(basisint)+1))
    Rbuffer = np.zeros((4*np.max(basisint)+1,4*np.max(basisint)+1,4*np.max(basisint)+1,3*4*np.max(basisint)+1))
    Na = np.zeros((len(basisidx),len(basisidx)))
    S = np.zeros((len(basisidx),len(basisidx)))
    T = np.zeros((len(basisidx),len(basisidx)))
    ERI = np.zeros((len(basisidx),len(basisidx),len(basisidx),len(basisidx)))
    # Making array to save E
    E1arr = np.zeros((len(basisidx),len(basisidx),np.max(basisidx[:,0]),np.max(basisidx[:,0]),np.max(basisint[:,0:3])*2+1))
    E2arr = np.zeros((len(basisidx),len(basisidx),np.max(basisidx[:,0]),np.max(basisidx[:,0]),np.max(basisint[:,0:3])*2+1))
    E3arr = np.zeros((len(basisidx),len(basisidx),np.max(basisidx[:,0]),np.max(basisidx[:,0]),np.max(basisint[:,0:3])*2+1))
    # basisidx [number of primitives, start index in basisfloat and basisint]
    # basisfloat array of float values for basis, N, zeta, c, x, y, z 
    # basisint array of integer values for basis, l, m, n, atomidx
    
    for k in range(0, len(basisidx)):
        for l in range(k, len(basisidx)):
            calc  = 0.0
            calc2 = 0.0
            calc3 = 0.0
            for i in range(basisidx[k,1],basisidx[k,1]+basisidx[k,0]):
                for j in range(basisidx[l,1],basisidx[l,1]+basisidx[l,0]):
                    a  = basisfloat[i,1]
                    b  = basisfloat[j,1]
                    Ax = basisfloat[i,3]
                    Ay = basisfloat[i,4]
                    Az = basisfloat[i,5]
                    Bx = basisfloat[j,3]
                    By = basisfloat[j,4]
                    Bz = basisfloat[j,5]
                    l1 = basisint[i,0]
                    l2 = basisint[j,0]
                    m1 = basisint[i,1]
                    m2 = basisint[j,1]
                    n1 = basisint[i,2]
                    n2 = basisint[j,2]
                    N1 = basisfloat[i,0]
                    N2 = basisfloat[j,0]
                    c1 = basisfloat[i,2]
                    c2 = basisfloat[j,2]
                    
                    #E1, E2, E3, p, P is also used in ERI calculation, make smart later
                    p   = a+b
                    Px  = (a*Ax+b*Bx)/p
                    Py  = (a*Ay+b*By)/p
                    Pz  = (a*Az+b*Bz)/p
                    
                    Ex = np.zeros(l1+l2+1)
                    Ey = np.zeros(m1+m2+1)
                    Ez = np.zeros(n1+n2+1)
                    
                    for t in range(l1+l2+1):
                        Ex[t] = E1arr[k,l,i-basisidx[k,1],j-basisidx[l,1],t] = E(l1,l2,t,Ax-Bx,a,b,Px-Ax,Px-Bx,Ax-Bx)
                    for u in range(m1+m2+1):
                        Ey[u] = E2arr[k,l,i-basisidx[k,1],j-basisidx[l,1],u] = E(m1,m2,u,Ay-By,a,b,Py-Ay,Py-By,Ay-By)
                    for v in range(n1+n2+1):
                        Ez[v] = E3arr[k,l,i-basisidx[k,1],j-basisidx[l,1],v] = E(n1,n2,v,Az-Bz,a,b,Pz-Az,Pz-Bz,Az-Bz)

                    for atom in range(1, len(input)):
                        Zc = input[atom,0]
                        Cx = input[atom,1]
                        Cy = input[atom,2]
                        Cz = input[atom,3]
                        RPC = ((Px-Cx)**2+(Py-Cy)**2+(Pz-Cz)**2)**0.5
                        R1 = R(l1+l2, m1+m2, n1+n2, Cx, Cy, Cz, Px, Py, Pz, p, R1buffer, Rbuffer)

                        calc += elnuc(p, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, Zc, Ex, Ey, Ez, R1)
                    calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2)
                    calc2 += calct
                    calc3 += calct2
                        
            Na[k,l] = Na[l,k] = calc
            S[k,l]  = S[l,k]  = calc3
            T[k,l]  = T[l,k]  = calc2
    #END OF one electron integrals

    # Run ERI
    for mu in range(0, len(basisidx)):
        for nu in range(mu, len(basisidx)):
            munu = mu*(mu+1)//2+nu
            for lam in range(0, len(basisidx)):
                for sig in range(lam, len(basisidx)):
                    lamsig = lam*(lam+1)//2+sig
                    if munu >= lamsig:
                        calc = 0.0
                        for i in range(basisidx[mu,1],basisidx[mu,1]+basisidx[mu,0]):
                            N1 = basisfloat[i,0]
                            a  = basisfloat[i,1]
                            c1 = basisfloat[i,2]
                            Ax = basisfloat[i,3]
                            Ay = basisfloat[i,4]
                            Az = basisfloat[i,5]
                            l1 = basisint[i,0]
                            m1 = basisint[i,1]
                            n1 = basisint[i,2]
                            for j in range(basisidx[nu,1],basisidx[nu,1]+basisidx[nu,0]):
                                N2 = basisfloat[j,0]
                                b  = basisfloat[j,1]
                                c2 = basisfloat[j,2]
                                Bx = basisfloat[j,3]
                                By = basisfloat[j,4]
                                Bz = basisfloat[j,5]
                                l2 = basisint[j,0]
                                m2 = basisint[j,1]
                                n2 = basisint[j,2]

                                p   = a+b
                                Px  = (a*Ax+b*Bx)/p
                                Py  = (a*Ay+b*By)/p
                                Pz  = (a*Az+b*Bz)/p
                                
                                E1 = E1arr[mu,nu,i-basisidx[mu,1],j-basisidx[nu,1]]
                                E2 = E2arr[mu,nu,i-basisidx[mu,1],j-basisidx[nu,1]]
                                E3 = E3arr[mu,nu,i-basisidx[mu,1],j-basisidx[nu,1]]

                                for k in range(basisidx[lam,1],basisidx[lam,1]+basisidx[lam,0]):
                                    N3 = basisfloat[k,0]
                                    c  = basisfloat[k,1]
                                    c3 = basisfloat[k,2]
                                    Cx = basisfloat[k,3]
                                    Cy = basisfloat[k,4]
                                    Cz = basisfloat[k,5]
                                    l3 = basisint[k,0]
                                    m3 = basisint[k,1]
                                    n3 = basisint[k,2]
                                    for l in range(basisidx[sig,1],basisidx[sig,1]+basisidx[sig,0]):
                                        N4 = basisfloat[l,0]
                                        d  = basisfloat[l,1]
                                        c4 = basisfloat[l,2]
                                        Dx = basisfloat[l,3]
                                        Dy = basisfloat[l,4]
                                        Dz = basisfloat[l,5]
                                        l4 = basisint[l,0]
                                        m4 = basisint[l,1]
                                        n4 = basisint[l,2]
                                                                                    
                                        q   = c+d
                                        Qx  = (c*Cx+d*Dx)/q
                                        Qy  = (c*Cy+d*Dy)/q
                                        Qz  = (c*Cz+d*Dz)/q

                                        E4 = E1arr[lam,sig,k-basisidx[lam,1],l-basisidx[sig,1]]
                                        E5 = E2arr[lam,sig,k-basisidx[lam,1],l-basisidx[sig,1]]
                                        E6 = E3arr[lam,sig,k-basisidx[lam,1],l-basisidx[sig,1]]
                                        
                                        alpha = p*q/(p+q)
                                        
                                        R1 = R(l1+l2+l3+l4, m1+m2+m3+m4, n1+n2+n3+n4, Qx, Qy, Qz, Px, Py, Pz, alpha, R1buffer, Rbuffer)
                                        calc += elelrep(p,q,l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, N4, c1, c2, c3, c4, E1, E2, E3, E4, E5, E6, R1)
                            
                        ERI[mu,nu,lam,sig] = ERI[nu,mu,lam,sig] = ERI[mu,nu,sig,lam] = ERI[nu,mu,sig,lam] = ERI[lam,sig,mu,nu] = ERI[sig,lam,mu,nu] = ERI[lam,sig,nu,mu] = ERI[sig,lam,nu,mu] = calc            
                                
    #END OF run ERI
    return Na, S, T, ERI