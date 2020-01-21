import numpy as np
import pylab as py
import scipy.signal as ss
import scipy.fftpack as sf

def czas(T = 1.0, Fs = 128.0):
    dt = 1.0/Fs
    t = np.arange(0,T,dt)
    return t
 
def gauss(t0 = 0.3, sigma = 0.02, T = 1.0, Fs = 128.0):
    t = czas(T,Fs)
    s = np.exp(-((t-t0)/(sigma))**2/2)
    return s
 
def g2(t0 = 0.3, sigma = 0.02, T = 1.0, Fs = 128.0):
    t = czas(T,Fs)
    s = (-(t-t0)/(sigma))*np.exp(-((t-t0)/(sigma))**2/2)
    return s
 
def g3(t0 = 0.3, sigma = 0.02, T = 1.0, Fs = 128.0):
    t = czas(T,Fs)
    s = np.exp(-((t-t0)/(sigma))**2/2)
    s[t<t0] = 0
    return s
 
def gabor(t0 = 0.5, sigma = 0.1, T = 1.0, f=10, phi = 0, Fs = 128.0):
    t = czas(T,Fs)
    s = np.exp(-((t-t0)/(sigma))**2/2) * np.cos(2*np.pi*f*(t-t0) + phi)
    return s
 
def sin(f = 10.0, T = 1.0, Fs = 128.0, phi =0 ):
    '''sin o zadanej cz?sto?ci (w Hz), d?ugo?ci, fazie i cz?sto?ci próbkowania
    Domy?lnie wytwarzany jest sygna? reprezentuj?cy 
    1 sekund? sinusa o cz?sto?ci 1Hz i zerowej fazie próbkowanego 128 Hz
    '''
    t = czas(T,Fs)
    s = np.sin(2*np.pi*f*t + phi)
    return s
 
def chirp(f0,fk,T,Fs):
    t = czas(T,Fs)
    f  = f0 + (fk-f0)/2.0/(T)*t
    s  = np.cos(2*np.pi*t*f)
    return s
 
def cwt(x, MinF,MaxF,Fs,w=7.0,df=1.0,plot = True):
    '''w - parametr falki Morleta,
      wiaze sie z jej czestoscia centralna i skala w nastepujacy sposob:
      f = 2*s*w / T
      gdzie: s-skala,  T-dlugosc sygnalu w sek.'''
    T= len(x)/Fs
    M = len(x)
    t = np.arange(0,T,1./Fs)
    freqs = np.arange(MinF,MaxF,df)
    P = np.zeros((len(freqs),M))
    X = sf.fft(x)
    for i,f in enumerate(freqs):
        s = T*f/(2*w)
        psi = sf.fft(ss.morlet(M, w=w, s=s, complete=True))
        psi /= np.sqrt(np.sum(psi*psi.conj()))    
        tmp = np.fft.fftshift(sf.ifft(X*psi))
        P[i,:] = (tmp*tmp.conj()).real
 
    if plot:
        py.imshow(P,aspect='auto',origin='lower',extent=(0,T,MinF, MaxF))
        py.show()
    return P,freqs,t
 
def wvd(x, Fs, plot=True):
    samples = len(x)
    N = samples / 2
    z = np.zeros(samples)
    xh = ss.hilbert(x)
    x_period_h = np.concatenate((z,xh,z));
 
    t = range(0, samples, 1)  # czas w samplach
    tfr = np.zeros((samples , samples), dtype=complex)
    for ti in t:
        for tau in range(-samples//2,samples//2):
            tfr[samples//2 + tau, ti] =  x_period_h[samples+ti +  tau] * x_period_h[samples+ti - tau].conj() 
    tfr = np.fft.fftshift(tfr,axes = 0)
    Tfr = np.fft.fft(tfr, samples, axis=0)/samples
    ts = np.array(t, dtype=float) / (float(Fs))
    f = np.linspace(0, Fs / 2, N)
    if plot:
        py.imshow( Tfr.real, interpolation='nearest', extent=[0, ts[-1], 0, f[-1]], origin='lower', aspect='auto')
        py.show()
    return Tfr, ts, f
