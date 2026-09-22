#!/usr/bin/env python3.6
path_GYSELA='dn_n_Anna.h5'#path for GYSELA file
path_Working='fw2dfdtd/' #path for the working directory
from numba import jit #library for calculation in C
import numpy as np
import sys
sys.path.append(path_Working)
import math
import cmath
import os
import glob
import h5py as h5
import copy
import time
import random as rand
from scipy.interpolate import griddata
from scipy.signal import medfilt
from scipy.signal import savgol_filter
from scipy.signal import chirp, sweep_poly
import multiprocessing as mp
c = 3.E8
ep0 = 8.85418782E-12
mu0 = 1.25663706E-6
e = 1.602E-19
me = 9.109E-31 
f0=60E9 
T=0 
w0 = 2*math.pi*f0
dt = 1/(80.*f0) 
dx = 2*c*dt 
dy = 2*c*dt
B_ampl=2.5*1.65  
ne_mode=1
plasma_size_x=0.1 
plasma_size_y=0.07 
x_start=2.15 
NX = math.ceil(plasma_size_x/dx) 
NY = math.ceil(2*plasma_size_y/dy)
Xpml = 100 
Ypml = 100
v_f=25e15/1 
v_w=2*math.pi*v_f
tmax =  10
sourcei = Xpml+5
sourcej = int(NY/2.)
waist = int(0.025/dx) 
sourcewidth = int(0.0324/dx)
n_max=2e19
comment='test_fixed_60GHz_2e9ne_10pts_from_source_15000'
def init(t,p,sigma,ww,tstart):
    return math.exp(-p**2/sigma**2)*cmath.exp(1j*(w0)*t)*(math.tanh(t/dt/10. - 3) + 1)/2./math.sqrt(mu0/ep0)
class loadHDF5():
    def __init__(self, filename):
        fh5       = h5.File(filename, 'r')
        var_in_h5 = list(fh5.keys())
        self.keys = var_in_h5

        for str_var in var_in_h5:
            str_var_new = str_var.replace('%','_')
            var_tmp     = fh5[str_var]
            self.__dict__[str_var_new] = var_tmp.value
        fh5.close()                

class grid:
    def __init__(self, length, width, xpml, ypml):
        
        self.length = length
        self.width = width
        self.xpml = xpml
        self.ypml = ypml
        
        self.Er = np.full((length, width+1), 0. + 0j)
        self.Ey = np.full((length+1, width), 0. + 0j)
        self.Jr = np.full((length, width+1), 0. + 0j)
        self.Jy = np.full((length+1, width), 0. + 0j)
        self.Hz = np.full((length, width), 0. + 0j)
        self.Hz_source = []
        self.res = []

        self.Hzx = np.full((length, width), 0. + 0j)
        self.Hzy = np.full((length, width), 0. + 0j)
        self.sxpml = np.full((length, width), 0.)
        self.ssxpml = np.full((length, width), 0.)
        self.sypml = np.full((length, width), 0.)
        self.ssypml = np.full((length, width), 0.)

        self.density = np.full((length, width), 0.)
        self.B0 = np.full((length, width), 0.)
        self.wc = np.array([[e*self.B0[j,i]/me for i in range(width)]for j in range(length)])
        self.wp = np.sqrt(self.density*e**2/(ep0*me))

    
    def clear(self):
        self.Er = np.full((self.length, self.width+1), 0. + 0j)
        self.Ey = np.full((self.length+1, self.width), 0. + 0j)
        self.Jr = np.full((self.length, self.width+1), 0. + 0j)
        self.Jy = np.full((self.length+1, self.width), 0. + 0j)
        self.Hz = np.full((self.length, self.width), 0. + 0j)
        self.Hzx = np.full((self.length, self.width), 0. + 0j)
        self.Hzy = np.full((self.length, self.width), 0. + 0j)
        self.res = []
        self.Hz_source = []
        
    def next_step(self, t, sourcei, sourcej, source_width,f):
        w = 2*math.pi*f
        self.res.append(self.Ey[sourcei+10,sourcej])
        width = self.width
        length = self.length
        xpml = self.xpml
        ypml = self.ypml

        for j in range(sourcej - source_width-1, sourcej + source_width + 1):       
            self.Hz[sourcei, j] += init(t, j-sourcej,waist,w,tsweepstart)       
        self.Hz_source.append(self.Hz[sourcei, sourcej])
        self.Er,self.Ey,self.Hz,self.Hzx,self.Hzy,self.Jr,self.Jy=next_step_numba(width, length, xpml, ypml,dt, mu0,ep0,dx,dy,self.sxpml,self.ssxpml,self.sypml,self.ssypml,self.Er,self.Ey,self.Hz,self.Hzx,self.Hzy,self.Jr,self.Jy,self.wc,self.wp)      
@jit("Tuple((complex128[:,:],complex128[:,:],complex128[:,:],complex128[:,:],complex128[:,:],complex128[:,:],complex128[:,:]))(int64,int64,int64,int64,float64,float64,float64,float64,float64,float64[:,:],float64[:,:],float64[:,:],float64[:,:],complex128[:,:],complex128[:,:],complex128[:,:],complex128[:,:],complex128[:,:],complex128[:,:],complex128[:,:],float64[:,:],float64[:,:])",nopython=True,cache=True)
def next_step_numba(width, length, xpml, ypml,dt, mu0,ep0,dx,dy,sxpml,ssxpml,sypml,ssypml,Er,Ey,Hz,Hzx,Hzy,Jr,Jy,wc,wp):
     
    Er_2=Er.copy()
    Ey_2=Ey.copy()
    Hz_2=Hz.copy()
    Hzx_2=Hzx.copy()
    Hzy_2=Hzy.copy()
    Jr_2=Jr.copy()
    Jy_2=Jy.copy()
    
    Er_2[:,0]=0
    Er_2[:,width]=0
    Ey_2[0,:]=0
    Ey_2[length,:]=0  

    for j in range(width):
        for i in range(length):

            if i <= xpml-1 or i >= length - xpml or j <= ypml-1 or j >= width - ypml : 
                Hzx_2[i,j] = math.exp(-ssxpml[i,j]*dt/mu0)*Hzx_2[i,j] - dt/(mu0*dx)*(Ey[i+1,j] - Ey[i,j])
                Hzy_2[i,j] = math.exp(-ssypml[i,j]*dt/mu0)*Hzy_2[i,j] + dt/(dy*mu0)*(Er[i,j+1] - Er[i,j])        
                   
            else :
                Jr_2[i,j] = (1-wc[i,j]**2*dt**2/4.)/(1 + wc[i,j]**2*dt**2/4.)*Jr[i,j] - wc[i,j]*dt/(1+wc[i,j]**2*dt**2/4.)*(Jy[i,j] + Jy[i,j-1] + Jy[i+1,j] + Jy[i+1, j-1])/4. + ep0*wp[i,j]**2*dt/(1+wc[i,j]**2*dt**2/4.)*(Er[i,j]-wc[i,j]*dt/2.*(Ey[i+1,j] + Ey[i,j]+ Ey[i, j-1] + Ey[i+1, j-1])/4.)
                Jy_2[i,j] = (1-wc[i,j]**2*dt**2/4.)/(1 + wc[i,j]**2*dt**2/4.)*Jy[i,j]+ wc[i,j]*dt/(1+wc[i,j]**2*dt**2/4.)*(Jr[i,j+1] + Jr[i-1,j+1]+ Jr[i, j] + Jr[i-1, j])/4. + ep0*wp[i,j]**2*dt/(1+wc[i,j]**2*dt**2/4.)*(Ey[i,j]+wc[i,j]*dt/2.*(Er[i,j+1] + Er[i-1,j+1]+ Er[i, j] + Er[i-1, j])/4.)
                Hz_2[i,j] = Hz_2[i,j] + dt/(mu0*dy)*(Er[i, j+1] - Er[i, j]) - dt/(mu0*dx)*(Ey[i+1,j] - Ey[i, j])


    for j in range(width+1):
        for i in range(length):
            if j != width:
                if (i<xpml or j <ypml or i > length - xpml-1 or j > width - ypml) :
                    Er_2[i,j] = math.exp(-sypml[i,j]*dt/ep0)*Er_2[i,j] + dt/(ep0*dy)*(Hzx_2[i,j] -Hzx_2[i,j-1] + Hzy_2[i,j] -Hzy_2[i,j-1])- dt/ep0*Jr_2[i,j]
                    
                elif j == ypml :
                    Er_2[i,j] = math.exp(-sypml[i,j]*dt/ep0)*Er_2[i,j] + dt/(ep0*dy)*(Hz_2[i,j]-Hzx_2[i,j-1] - Hzy_2[i,j-1] )- dt/ep0*Jr_2[i,j]
                    
                elif j == width - ypml:
                    Er_2[i,j] = math.exp(-sypml[i,j]*dt/ep0)*Er_2[i,j] + dt/(ep0*dy)*(Hzx_2[i,j] + Hzy_2[i,j] - Hz_2[i,j-1]) - dt/ep0*Jr_2[i,j]
                    
                else :
                    Er_2[i,j] = Er_2[i,j] + dt/(dy*ep0)*(Hz_2[i,j]- Hz_2[i,j-1]) - dt/ep0*Jr_2[i,j]
                
            else : 
                Er_2[i,j] = math.exp(-sypml[i,j-1]*dt/ep0)*Er_2[i,j] + dt/(ep0*dy)*(-Hzx_2[i,j-1] - Hzy_2[i,j-1])- dt/ep0*Jr_2[i,j]
        
    
        
    for j in range(width):
        for i in range(length+1): 
            if i != length :
                    
                if (i<xpml or j < ypml or i > length - xpml or j > width - ypml - 1) :
                    Ey_2[i,j] = math.exp(-sxpml[i,j]*dt/ep0)*Ey_2[i,j] - dt/(ep0*dx)*(Hzx_2[i,j] -Hzx_2[i-1,j] + Hzy_2[i,j] - Hzy_2[i-1,j])- dt/ep0*Jy_2[i,j]
                    
                elif i == xpml : 
                    Ey_2[i,j] = math.exp(-sxpml[i,j]*dt/ep0)*Ey_2[i,j] - dt/(ep0*dx)*(Hz_2[i,j]-Hzx_2[i-1,j] - Hzy_2[i-1,j]) - dt/ep0*Jy_2[i,j]
                    
                elif i == length - xpml :
                    Ey_2[i,j] = math.exp(-sxpml[i,j]*dt/ep0)*Ey_2[i,j] - dt/(ep0*dx)*(Hzx_2[i,j] + Hzy_2[i,j] - Hz_2[i-1,j]) - dt/ep0*Jy_2[i,j]
                    
                else : 
                    Ey_2[i,j] = Ey_2[i,j] - dt/(ep0*dx)*(Hz_2[i, j] -Hz_2[i-1,j]) - dt/ep0*Jy_2[i,j]
                        
            else : 
                Ey_2[i,j] = math.exp(-sxpml[i-1,j]*dt/ep0)*Ey_2[i,j] - dt/(ep0*dx)*(-Hzx_2[i-1,j] - Hzy_2[i-1,j]) - dt/ep0*Jy_2[i,j]
    
    return Er_2,Ey_2,Hz_2,Hzx_2,Hzy_2,Jr_2,Jy_2
           
def PML(simul, sigmam, n):
    for j in range(simul.width):
        for i in range(simul.xpml):
            simul.sxpml[i,j] = sigmam*((simul.xpml - i)/float(simul.xpml))**(n)
            simul.ssxpml[i,j] = (mu0/(ep0))*simul.sxpml[i,j]
            
    for j in range(simul.width):
        for i in range(simul.length - simul.xpml -1 , simul.length):
            simul.sxpml[i,j] =sigmam*((i - simul.length + simul.xpml + 1)/float(simul.xpml))**(n)
            simul.ssxpml[i,j] =(mu0/(ep0))*simul.sxpml[i,j]
            
    for i in range(simul.length):
        for j in range(simul.ypml):
            simul.sypml[i,j] = sigmam*((simul.ypml - j)/float(simul.ypml))**(n)
            simul.ssypml[i,j] =(mu0/(ep0))*simul.sypml[i,j]
            
    for i in range(simul.length):
        for j in range(simul.width - simul.ypml -1, simul.width):
            simul.sypml[i,j] = sigmam*((j - simul.width + simul.ypml +1)/float(simul.ypml))**(n)
            simul.ssypml[i,j] = (mu0/(ep0))*simul.sypml[i,j]

def wp(density):
    return np.array([[math.sqrt(max(0.00001, density[j,i]*e**2/(ep0*me))) for i in range(NY)]for j in range(NX)])

def N2(w, wc, wp):
    return np.array([[(1 - (wp[i,j]/w)**2*((w**2 - wp[i,j]**2)/(w**2-wc[i,j]**2 - wp[i,j]**2))) for j in range(NY)] for i in range(NX)])

def Nx(w,wc,wp):
    Nx = []
    N = N2(w,wc,wp)
    for i in range(NX):
        if N[i][50]>0:
            Nx.append(math.sqrt(N[i][50]))
        else :
            Nx.append(0)
    return Nx

def COL(w,wc,wp):
    
    cut=[0 for i in range(NY)]
    N = N2(w,wc,wp)
    for i in range(NY):
        for j in range(NX-1) :
            if N[j,i]*N[j+1,i]<0 :
                cut[i] = j- N[j,i]/float(N[j+1,i] - N[j,i])
                break
    return cut

def phi_WKB(i,w,wc,wp, dx):
    wkb = 0
    Nx0 = Nx(w,wc,wp)
    for k in range(31, int(i)+1):
        wkb += dx*Nx0[k]
    wkb = wkb*2*w/c - math.pi/2
    wkb += (i - int(i))*dx*Nx0[int(i)]*2*w/c
    return wkb

def correction(density, nec, ic, di):
    A0 = nec - ((density[int(ic)+1] - density[int(ic)])*(ic - int(ic)) + density[int(ic)])
    a0 = A0
    a2 = -2*A0/float(di)**2
    a4 = A0/float(di)**4
    for i in range(int(ic) - di, min(int(ic) + di+1, NX)):
        density[i] += a0 + a2*(i-ic)**2 + a4*(i-ic)**4

def interp(col,i0, width):
    x1 = i0-width
    x2 = i0 + width
    y1 = (col[int(x1)+1] - col[int(x1)])*(x1-int(x1)) + col[int(x1)]
    y2 = (col[int(x2)+1] - col[int(x2)])*(x2-int(x2)) + col[int(x2)]
    yy1 = col[int(x1)+1] - col[int(x1)]
    yy2 = col[int(x2)+1] - col[int(x2)]
    M = np.array([[1, x1, x1**2, x1**3], [1,x2,x2**2, x2**3], [0,1,2*x1,3*x1**2], [0,1,2*x2,3*x2**2]])
    B = np.array([[y1], [y2], [yy1], [yy2]])
    A = np.dot(np.linalg.inv(M),B)
    for i in range(int(x1),int(x2)+1):
        col[i] = A[0] + A[1]*(i) + A[2]*(i)**2 + A[3]*(i)**3

def fluct_col(length, height, width):
    col = [(i%length)*height/length for i in range(1200)]
    for i in range(1,1200):
        if i%length == 0 :
            interp(col, i, width)
    return col

def mod_fluct(i,t,v):
    return fluct[int((i+v*t)%NY)]

def sin_fluct(a,i,L,t,T):
    return a*math.sin(i*2*math.pi/float(L) - 2*math.pi/float(T)*t)

def nec0(dens, w):
    col = COL(w, wc, wp(dens))[100]
    nec = (dens[int(col)+1,100] - dens[int(col),100])*(col - int(col)) + dens[int(col),100]
    return nec

def trace_wx(density0, wc, NX):
    ne_7=density0[:,0]
    wc_7=np.flipud(wc[:,0])
    wp_7=np.sqrt(np.abs(ne_7*e**2/(ep0*me)))
    wxx=[]
    for j in range(0,NX):
        wxx.append(0.5*(wc_7[j]+math.sqrt(wc_7[j]**2+4*wp_7[j]**2)))
    wxx=np.array(wxx)
    return wxx

def codeFDTD(gys_n_with_dn, gys_points, gys_xx_lin, gys_yy_lin, NX, NY, Xpml, Ypml, B0, wc, x_start, plasma_size_x, tmax, dt,  sourcei, sourcej, sourcewidth,f0, T, path_Working):
    results = []
    Hz_s=[]
    t=T
    w0 = 2*math.pi*f0
    n_max=2e19
    gys_temp=gys_n_with_dn[0:513,0:512,t]
    gys_dn_val=np.reshape(gys_temp,512*513)
    gys_grid_z0 = griddata(gys_points, gys_dn_val, (gys_xx_lin, gys_yy_lin), method='linear')
    density0 = gys_grid_z0.T*1e19
    if ne_mode==0:
        density0=0*density0
    if ne_mode==2:
        density0=0*density0
        pos_0=NX-240  
        for i in range(NY):
            density0[0:pos_0,i]=np.linspace(n_max,0,pos_0)
            for j in range(pos_0-10,pos_0+10):
                density0[j-1,i]=n_max*(j-pos_0-10)**2/(pos_0-Xpml)/40
    simul0 = grid(NX,NY,Xpml,Ypml)
    simul0.wc = wc
    simul0.clear()
    simul0.density = np.flipud(density0)
    simul0.wp = wp(simul0.density)
    PML(simul0, 10. , 4.)

    for k in range(tmax) :
        simul0.next_step(k*dt, sourcei, sourcej, sourcewidth,f0)
        Hz_s.append(simul0.Hz[sourcei,sourcej])
    sourceEy=simul0.res
    phi1 = []
    ind_phi_stable=30 #500
    for l in range(tmax):
        phi1.append(cmath.phase(sourceEy[l]*cmath.exp(-1j*w0*l*dt)-sourceEy[ind_phi_stable]*cmath.exp(-1j*w0*ind_phi_stable*dt)))
    results.append(simul0.res)
    np.save(path_Working+'results/test_2019/sourceEy' + str(f0/1e9) +comment+'.npy', results[t])
    np.save(path_Working+'results/test_2019/phase' + str(f0/1e9) +comment+'.npy', phi1)
    np.save(path_Working+'results/test_2019/Ey' + str(f0/1e9) +comment+'.npy', simul0.Ey)
    return 0
#end of function

B0 = np.array([[B_ampl/float(x_start+NX*dx-j*dx) for i in range(NY)] for j in range(NX)])
wc = np.array([[e*B0[j,i]/me for i in range(NY)]for j in range(NX)])

A=30
print('Done')
gys_data = loadHDF5(path_GYSELA)
G_size_rho=512
G_size_theta=513
G_size_t=170
gys_rho=gys_data.rho
gys_theta=np.linspace(0, 2*math.pi,G_size_theta) 
gys_dn=gys_data.dn_on_n_trth
gys_n=gys_data.dens
gys_n[437:467]=np.linspace(gys_n[437],gys_n[467],30)
for ll in range(400,512):
    gys_n[ll]=gys_n[ll]*0.5*(1-math.tanh((ll-420)/1e1))
gys_x= np.array([[0.0000 for i in range(G_size_rho)] for j in range(G_size_theta)])
gys_y= np.array([[0.0000 for i in range(G_size_rho)] for j in range(G_size_theta)])
for ix in range(G_size_rho):
    for iy in range(G_size_theta):
        gys_x[iy][ix]=1.65+gys_rho[ix]*0.55*np.cos(gys_theta[iy])
        gys_y[iy][ix]=gys_rho[ix]*0.55*np.sin(gys_theta[iy])
gys_dn1=np.swapaxes(gys_dn,0,2)
gys_dn1=np.swapaxes(gys_dn1,0,1)
gys_dens=np.resize(gys_n,(G_size_theta,G_size_rho))
gys_n_with_dn=np.array([[[0.0000 for i in range(G_size_theta)] for j in range(G_size_rho)] for k in range(G_size_theta)])
gys_dn1[gys_dn1>10]=10
for ix in range(G_size_t):
   gys_n_with_dn[:,:,ix]=gys_dens*(1+gys_dn1[:,:,ix]/1e2)
gys_x1=np.reshape(gys_x, G_size_rho*G_size_theta)
gys_y1=np.reshape(gys_y, G_size_rho*G_size_theta)
gys_points= np.array([[0.0000 for i in range(2)] for j in range(G_size_rho*G_size_theta)])
gys_points[:,0]=gys_x1
gys_points[:,1]=gys_y1

gys_x_lin=np.linspace(x_start,x_start+plasma_size_x,NX)
gys_y_lin=np.linspace(-plasma_size_y,plasma_size_y,NY)
gys_xx_lin, gys_yy_lin = np.meshgrid(gys_x_lin, gys_y_lin)

gys_temp=gys_n_with_dn[0:513,0:512,T]
gys_dn_val=np.reshape(gys_temp,512*513)
gys_grid_z0 = griddata(gys_points, gys_dn_val, (gys_xx_lin, gys_yy_lin), method='nearest')

# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())
print('Done')
# Step 2: `pool.apply` the `howmany_within_range()`
frequencies = [60 + i*5 for i in range(4)]
outputs=[pool.apply(codeFDTD, args=(gys_n_with_dn, gys_points, gys_xx_lin, gys_yy_lin, NX, NY, Xpml, Ypml, B0, wc, x_start, plasma_size_x, tmax, dt,  sourcei, sourcej, sourcewidth,ff, T, path_Working)) for ff in frequencies]

# Step 3: Don't forget to close
pool.close()  
