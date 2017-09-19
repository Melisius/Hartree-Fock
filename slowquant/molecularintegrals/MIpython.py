import numpy as np
import math
import scipy.misc as scm
from numba import jit, float64, int32
from numba.types import Tuple

##INTEGRAL FUNCTIONS
def nucrep(input):
    #Classical nucleus nucleus repulsion
    Vnn = 0
    for i in range(1, len(input)):
        for j in range(1, len(input)):
            if i < j:
                Vnn += (input[i][0]*input[j][0])/(((input[i][1]-input[j][1])**2+(input[i][2]-input[j][2])**2+(input[i][3]-input[j][3])**2))**0.5
    return Vnn


def Nrun(basisset):
    # Normalize primitive functions
    for i in range(len(basisset)):
        for j in range(len(basisset[i][5])):
            a = basisset[i][5][j][1]
            l = basisset[i][5][j][3]
            m = basisset[i][5][j][4]
            n = basisset[i][5][j][5]
            
            part1 = (2.0/math.pi)**(3.0/4.0)
            part2 = 2.0**(l+m+n) * a**((2.0*l+2.0*m+2.0*n+3.0)/(4.0))
            part3 = math.sqrt(scm.factorial2(int(2*l-1))*scm.factorial2(int(2*m-1))*scm.factorial2(int(2*n-1)))
            basisset[i][5][j][0] = part1 * ((part2)/(part3))
    """
    # Normalize contractions
    for k in range(len(basisset)):
        if len(basisset[k][5]) != 1:
            l = basisset[k][5][0][3]
            m = basisset[k][5][0][4]
            n = basisset[k][5][0][5]
            L = l+m+n
            factor = (np.pi**(3.0/2.0)*scm.factorial2(int(2*l-1))*scm.factorial2(int(2*m-1))*scm.factorial2(int(2*n-1)))/(2.0**L)
            sum = 0
            for i in range(len(basisset[k][5])):
                for j in range(len(basisset[k][5])):
                    alphai = basisset[k][5][i][1]
                    alphaj = basisset[k][5][j][1]
                    ai     = basisset[k][5][i][2]*basisset[k][5][i][0]
                    aj     = basisset[k][5][j][2]*basisset[k][5][j][0]
                    
                    sum += ai*aj/((alphai+alphaj)**(L+3.0/2.0))
            
            Nc = (factor*sum)**(-1.0/2.0)
            for i in range(len(basisset[k][5])):
                basisset[k][5][i][0] *= Nc
    """
    return basisset


@jit(flaot64[:,:,:](int32, int32, int32, float64, float64, float64, float64, float64, float64, float64, float64[:,:,:], float64[:,:,:,:], int32),nopython=True,cache=True)
def R(l1l2, m1m2, n1n2, Cx, Cy, Cz, Px, Py, Pz, p, R1, Rbuffer, check=0):
    cdef double RPC, PCx, PCy, PCz, val
    cdef int t, u, v, n, exclude_from_n
    # check = 0, normal calculation. 
    # check = 1, derivative calculation
    
    PCx = Px-Cx
    PCy = Py-Cy
    PCz = Pz-Cz
    RPC = ((PCx)**2+(PCy)**2+(PCz)**2)**0.5
    if check == 0:
        for t in range(0, l1l2+1):
            for u in range(0, m1m2+1):
                for v in range(0, n1n2+1):
                    # Check the range of n, to ensure no redundent n are calculated
                    if t == u == 0:
                        exclude_from_n = v
                    elif t == 0:
                        exclude_from_n = n1n2 + u
                    else:
                        exclude_from_n = n1n2 + m1m2 + t
                    for n in range(0, l1l2+m1m2+n1n2+1-exclude_from_n):
                        val = 0.0
                        if t == u == v == 0:
                            Rbuffer[t,u,v,n] = (-2.0*p)**n*boys(n,p*RPC*RPC)
                        else:
                            if t == u == 0:
                                if v > 1:
                                    val += (v-1)*Rbuffer[t,u,v-2,n+1]
                                val += PCz*Rbuffer[t,u,v-1,n+1]  
                            elif t == 0:
                                if u > 1:
                                    val += (u-1)*Rbuffer[t,u-2,v,n+1]
                                val += PCy*Rbuffer[t,u-1,v,n+1]
                            else:
                                if t > 1:
                                    val += (t-1)*Rbuffer[t-2,u,v,n+1]
                                val += PCx*Rbuffer[t-1,u,v,n+1]
                            Rbuffer[t,u,v,n] = val
                            
                        if n == 0:
                            R1[t,u,v] = Rbuffer[t,u,v,n]
                            
                            
    elif check == 1:
        # For the first derivative +1 is needed in t, u and v
        # First the "normal" Rs are calculated
        for t in range(0, l1l2+1):
            for u in range(0, m1m2+1):
                for v in range(0, n1n2+1):
                    # Check the range of n, to ensure no redundent n are calculated
                    if t == u == 0:
                        exclude_from_n = v
                    elif t == 0:
                        exclude_from_n = n1n2 + u
                    else:
                        exclude_from_n = n1n2 + m1m2 + t
                    # +1 in n because of derivative    
                    for n in range(0, l1l2+m1m2+n1n2+1+1-exclude_from_n):
                        val = 0.0
                        if t == u == v == 0:
                            Rbuffer[t,u,v,n] = (-2.0*p)**n*boys(n,p*RPC*RPC)
                        else:
                            if t == u == 0:
                                if v > 1:
                                    val += (v-1)*Rbuffer[t,u,v-2,n+1]
                                val += PCz*Rbuffer[t,u,v-1,n+1]  
                            elif t == 0:
                                if u > 1:
                                    val += (u-1)*Rbuffer[t,u-2,v,n+1]
                                val += PCy*Rbuffer[t,u-1,v,n+1]
                            else:
                                if t > 1:
                                    val += (t-1)*Rbuffer[t-2,u,v,n+1]
                                val += PCx*Rbuffer[t-1,u,v,n+1]
                            Rbuffer[t,u,v,n] = val

                        if n == 0:
                            R1[t,u,v] = Rbuffer[t,u,v,n]
        
        # The next three blocks of code, calculates the
        # +1 incriments in the different angularmoment directions.
        # only one direction is +1 at a time.
        # eg. no need to calc R(t+1,u+1,v+1)
        # but only; R(t+1,u,v), R(t,u,v+1), R(t,u+,v)
        v = n1n2+1
        for t in range(0, l1l2+1):
            for u in range(0, m1m2+1):
                # Check the range of n, to ensure no redundent n are calculated
                if t == u == 0:
                    exclude_from_n = v
                elif t == 0:
                    exclude_from_n = n1n2 + 1 + u
                else:
                    exclude_from_n = n1n2 + 1 + m1m2 + t
                # +1 in n because of derivative    
                for n in range(0, l1l2+m1m2+n1n2+1+1-exclude_from_n):
                    val = 0.0
                    if t == u == 0:
                        if v > 1:
                            val += (v-1)*Rbuffer[t,u,v-2,n+1]
                        val += PCz*Rbuffer[t,u,v-1,n+1]  
                    elif t == 0:
                        if u > 1:
                            val += (u-1)*Rbuffer[t,u-2,v,n+1]
                        val += PCy*Rbuffer[t,u-1,v,n+1]
                    else:
                        if t > 1:
                            val += (t-1)*Rbuffer[t-2,u,v,n+1]
                        val += PCx*Rbuffer[t-1,u,v,n+1]
                    Rbuffer[t,u,v,n] = val

                    if n == 0:
                        R1[t,u,v] = Rbuffer[t,u,v,n]
        
        u = m1m2+1
        for t in range(0, l1l2+1):
            for v in range(0, n1n2+1):
                # Check the range of n, to ensure no redundent n are calculated
                if t == 0:
                    exclude_from_n = n1n2 + u
                else:
                    exclude_from_n = n1n2 + m1m2 + 1 + t
                # +1 in n because of derivative    
                for n in range(0, l1l2+m1m2+n1n2+1+1-exclude_from_n):
                    val = 0.0
                    if t == 0:
                        if u > 1:
                            val += (u-1)*Rbuffer[t,u-2,v,n+1]
                        val += PCy*Rbuffer[t,u-1,v,n+1]
                    else:
                        if t > 1:
                            val += (t-1)*Rbuffer[t-2,u,v,n+1]
                        val += PCx*Rbuffer[t-1,u,v,n+1]
                    Rbuffer[t,u,v,n] = val

                    if n == 0:
                        R1[t,u,v] = Rbuffer[t,u,v,n]

        t = l1l2+1
        for u in range(0, m1m2+1):
            for v in range(0, n1n2+1):
                # Check the range of n, to ensure no redundent n are calculated
                exclude_from_n = n1n2 + m1m2 + t
                # +1 in n because of derivative    
                for n in range(0, l1l2+m1m2+n1n2+1+1-exclude_from_n):
                    val = 0.0
                    if t > 1:
                        val += (t-1)*Rbuffer[t-2,u,v,n+1]
                    val += PCx*Rbuffer[t-1,u,v,n+1]
                    Rbuffer[t,u,v,n] = val

                    if n == 0:
                        R1[t,u,v] = Rbuffer[t,u,v,n]
        
    return R1
    
    
def boys(double m,double T):
    return hyp1f1(m+0.5,m+1.5,-T)/(2.0*m+1.0) 


@jit(float64(float64, float64, int32, int32, int32, int32, int32, int32, int32, int32, int32, int32, int32, int32, float64, float64, float64, float64, float64, float64, float64, float64, float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:]),nopython=True,cache=True)
def elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, N4, c1, c2, c3, c4, E1, E2, E3, E4, E5, E6, Rpre):
    
    pi = 3.141592653589793238462643383279

    N = N1*N2*N3*N4*c1*c2*c3*c4
    
    val = 0.0
    for tau in range(l3+l4+1):
        for nu in range(m3+m4+1):
            for phi in range(n3+n4+1):
                factor = (-1.0)**(tau+nu+phi)*E4[tau]*E5[nu]*E6[phi]
                for t in range(l1+l2+1):
                    for u in range(m1+m2+1):
                        for v in range(n1+n2+1):
                            val += E1[t]*E2[u]*E3[v]*Rpre[t+tau,u+nu,v+phi]*factor

    val *= 2.0*pi**2.5/(p*q*(p+q)**0.5) 
    return val*N


@jit(float64(int32,int32,int32,float64,float64,float64,float64,float64,float64),nopython=True,cache=True)
def E(i, j, t, Qx, a, b, XPA, XPB, XAB):
    #McMurchie-Davidson scheme, 9.5.6 and 9.5.7 Helgaker

    p = a + b
    q = a*b/p
    if (t < 0) or (t > (i + j)):
        return 0.0
    elif i == j == t == 0:
        return np.exp(-q*Qx*Qx)
    elif j == 0:
        return (1.0/(2.0*p))*E(i-1,j,t-1,Qx,a,b,XPA,XPB,XAB) + XPA*E(i-1,j,t,Qx,a,b,XPA,XPB,XAB) + (t+1.0)*E(i-1,j,t+1,Qx,a,b,XPA,XPB,XAB)
    else:
        return (1.0/(2.0*p))*E(i,j-1,t-1,Qx,a,b,XPA,XPB,XAB) + XPB*E(i,j-1,t,Qx,a,b,XPA,XPB,XAB) + (t+1.0)*E(i,j-1,t+1,Qx,a,b,XPA,XPB,XAB)    


@jit(float64(float64, int32, int32, int32, int32, int32, int32, float64, float64, float64, float64, float64, float64[:], float64[:], float64[:], float64[:,:,:]),nopython=True,cache=True)
def elnuc(p, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, Zc, Ex, Ey, Ez, R1):
    #McMurchie-Davidson scheme    
    double pi = 3.141592653589793238462643383279
    
    N = N1*N2*c1*c2

    val = 0.0
    for t in range(0, l1+l2+1):
        for u in range(0, m1+m2+1):
            for v in range(0, n1+n2+1):
                val += Ex[t]*Ey[u]*Ez[v]*R1[t,u,v]*Zc

    return -val*2.0*pi/p*N


@jit(Tuple((float64,float64,float64))(float64, float64, float64, float64, float64, float64, float64, float64, int32, int32, int32, int32, int32, int32, float64, float64, float64, float64),nopython=True,cache=True)
def Kin(a, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb, na, nb, N1, N2, c1, c2):
    #Obara-Saika scheme, 9.3.40 and 9.3.41 Helgaker
    # Calculates electronic kinetic energy and overlap integrals
    #    at the same time
    cdef double p, N, Px, Py, Pz, XPA, YPA, ZPA, XPB, YPB, ZPB
    cdef double [:,:] Tijx, Tijy, Tijz, Sx, Sy, Sz
    cdef int i, j
    
    p = a + b
    N = N1*N2*c1*c2
    Px = (a*Ax+b*Bx)/p
    Py = (a*Ay+b*By)/p
    Pz = (a*Az+b*Bz)/p
    XPA = Px - Ax
    YPA = Py - Ay
    ZPA = Pz - Az
    XPB = Px - Bx
    YPB = Py - By
    ZPB = Pz - Bz
    
    Tijx = np.zeros((la+2,lb+2))
    Tijy = np.zeros((ma+2,mb+2))
    Tijz = np.zeros((na+2,nb+2))
    Sx = Overlap(a, b, la, lb, Ax, Bx)
    Sy = Overlap(a, b, ma, mb, Ay, By)
    Sz = Overlap(a, b, na, nb, Az, Bz)
    Tijx[0,0] = (a-2.0*a**2*(XPA**2+1.0/(2.0*p)))*Sx[0,0]
    Tijy[0,0] = (a-2.0*a**2*(YPA**2+1.0/(2.0*p)))*Sy[0,0]
    Tijz[0,0] = (a-2.0*a**2*(ZPA**2+1.0/(2.0*p)))*Sz[0,0]
    
    for i in range(0, la+1):
        for j in range(0, lb+1):
            Tijx[i+1,j] = XPA*Tijx[i,j] + 1.0/(2.0*p)*(i*Tijx[i-1,j]+j*Tijx[i,j-1]) + b/p*(2.0*a*Sx[i+1,j] - i*Sx[i-1,j])
            Tijx[i,j+1] = XPB*Tijx[i,j] + 1.0/(2.0*p)*(i*Tijx[i-1,j]+j*Tijx[i,j-1]) + a/p*(2.0*b*Sx[i,j+1] - j*Sx[i,j-1])
    
    for i in range(0, ma+1):
        for j in range(0, mb+1):
            Tijy[i+1,j] = YPA*Tijy[i,j] + 1.0/(2.0*p)*(i*Tijy[i-1,j]+j*Tijy[i,j-1]) + b/p*(2.0*a*Sy[i+1,j] - i*Sy[i-1,j])
            Tijy[i,j+1] = YPB*Tijy[i,j] + 1.0/(2.0*p)*(i*Tijy[i-1,j]+j*Tijy[i,j-1]) + a/p*(2.0*b*Sy[i,j+1] - j*Sy[i,j-1])
    
    for i in range(0, na+1):
        for j in range(0, nb+1):
            Tijz[i+1,j] = ZPA*Tijz[i,j] + 1.0/(2.0*p)*(i*Tijz[i-1,j]+j*Tijz[i,j-1]) + b/p*(2.0*a*Sz[i+1,j] - i*Sz[i-1,j])
            Tijz[i,j+1] = ZPB*Tijz[i,j] + 1.0/(2.0*p)*(i*Tijz[i-1,j]+j*Tijz[i,j-1]) + a/p*(2.0*b*Sz[i,j+1] - j*Sz[i,j-1])
    
    return (Tijx[la, lb]*Sy[ma,mb]*Sz[na,nb]+Tijy[ma, mb]*Sx[la,lb]*Sz[na,nb]+Tijz[na, nb]*Sy[ma,mb]*Sx[la,lb])*N, Sx[la, lb]*Sy[ma, mb]*Sz[na, nb]*N


@jit(float64[:,:](float64,float64,int32,int32,float64,float64),nopython=True,cache=True)
def Overlap(a, b, la, lb, Ax, Bx):
    #Obara-Saika scheme, 9.3.8 and 9.3.9 Helgaker
    #Used in Kin integral!
    pi = 3.141592653589793238462643383279

    p = a + b
    u = a*b/p
    
    Px = (a*Ax+b*Bx)/p
    
    S00 = (pi/p)**0.5 * np.exp(-u*(Ax-Bx)**2)
    
    Sij = np.zeros((la+2,lb+2))
    Sij[0,0] = S00
    
    
    for i in range(0, la+1):
        for j in range(0, lb+1):
            Sij[i+1,j] = (Px-Ax)*Sij[i,j] + 1.0/(2.0*p) * (i*Sij[i-1,j] + j*Sij[i,j-1])
            Sij[i,j+1] = (Px-Bx)*Sij[i,j] + 1.0/(2.0*p) * (i*Sij[i-1,j] + j*Sij[i,j-1])
    
    return Sij



