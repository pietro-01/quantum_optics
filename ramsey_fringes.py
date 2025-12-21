### RAMSAY FRINGES ###

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const

## MY DATA ##

rabi_0=2*np.pi*14
tau=np.pi/2/rabi_0
T=0.3
bloch_0=np.array([0,0,-1])
dets=np.linspace(-300,300,601)

## UTILITY FUNCTIONS ##

def normalize(v) :
    if np.linalg.norm(v)!=0:
        v=v/np.linalg.norm(v)
    return v

def angle(rabi, time) :
    return np.linalg.norm(rabi)*time

def evo_step(bloch,rabi,time):
    rabi_norm=normalize(rabi)
    ang=angle(rabi,time)
    terms=[]
    terms.append(bloch*np.cos(ang))
    terms.append(np.cross(rabi_norm,bloch)*np.sin(ang))
    terms.append(rabi_norm*np.inner(rabi_norm,bloch)*(1-np.cos(ang)))
    bloch_final=sum(terms)
    return bloch_final

def evo_tot(bloch_0,rabi_0,tau,T,det):
    rabi_int=np.array([rabi_0,0,det])
    rabi_free=np.array([0,0,det])
    bloch_1=evo_step(bloch_0,rabi_int,tau)
    bloch_2=evo_step(bloch_1,rabi_free,T)
    bloch_3=evo_step(bloch_2,rabi_int,tau)
    return bloch_3

def prob_inversion(bloch):
    return (bloch[2]+1)/2

##RAMSAY FRINGES, ONE VELOCITY ##

inv_prob=np.zeros(len(dets))
for i in range(len(dets)):
    inv_prob[i]=prob_inversion(evo_tot(bloch_0,rabi_0,tau,T,dets[i]))

fig, ax = plt.subplots()
ax.plot(dets,inv_prob)
ax.set(xlabel='detuning (Hz)', ylabel='inversion probability', title='Ramsay fringes')
ax.grid()

## RAMSAY FRINGES, MAXWELL BOLTZMANN ##

vel_center=const.g*T/2 # velocità centrale è data dal tempo medio di attraversamento 
vel_max=vel_center*5 # stabilisco in maniera arbitraria un massimo 
mass = 2.2e-25 # la massa del cesio 
temp_low = 10e-6 # temperatura 
temp_high = 1e-3

class MaxwellBoltzmann():
    """maxwell blotzmann distribution"""
    def __init__(self,temp,vel_center,mass):
        self.prefactor=np.sqrt(mass/(2*np.pi*const.k*temp)) 
        self.partial_exponent=-mass/(2*const.k*temp)
        self.vel_center=vel_center

    def prob(self,vel):
        exponent=self.partial_exponent*((vel-self.vel_center)**2)
        probability_density=self.prefactor*np.exp(exponent)
        return probability_density



vel_vec=np.linspace(0,vel_max,301) #vettore delle velocità per le quali farò integrazione 
vel_d=vel_vec[1]-vel_vec[0] # delta di velocità

# Low temperature #

my_distribution_low = MaxwellBoltzmann(temp_low,vel_center,mass)
inv_prob_mbd_low=[] # lista dove salvo le inversion probabilities per bassa temperatura

for det in dets:
    element=0 #inizializzo un counter 
    for vel in vel_vec: #faccio una integrazione discreta dato un certo det
        T_free=2*vel/const.g #il tempo della free evolution dipende dalla velocità 
        bloch_final=evo_tot(bloch_0, rabi_0, tau, T_free, det)
        element+=prob_inversion(bloch_final)*my_distribution_low.prob(vel)
    inv_prob_mbd_low.append(element*vel_d) #ricorda di moltiplicare per vel_d

fig2, ax2 = plt.subplots()
ax2.plot(dets,inv_prob_mbd_low)
ax2.set(xlabel='detuning (Hz)', ylabel='inversion probability', title='Ramsay fringes with MBD 10uK')
ax2.grid()

# High temperature #

my_distribution_high = MaxwellBoltzmann(temp_high,vel_center,mass)
inv_prob_mbd_high=[] # lista dove salvo le inversion probabilities per bassa temperatura

for det in dets:
    element=0 #inizializzo un counter 
    for vel in vel_vec: #faccio una integrazione discreta dato un certo det
        T_free=2*vel/const.g #il tempo della free evolution dipende dalla velocità 
        bloch_final=evo_tot(bloch_0, rabi_0, tau, T_free, det)
        element+=prob_inversion(bloch_final)*my_distribution_high.prob(vel)
    inv_prob_mbd_high.append(element*vel_d) #ricorda di moltiplicare per vel_d

fig3, ax3 = plt.subplots()
ax3.plot(dets,inv_prob_mbd_high)
ax3.set(xlabel='detuning (Hz)', ylabel='inversion probability', title='Ramsay fringes with MBD 1mK')
ax3.grid()

plt.show()
