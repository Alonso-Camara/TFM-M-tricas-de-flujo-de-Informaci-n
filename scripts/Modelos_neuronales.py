# -*- coding: utf-8 -*-
"""
Modelos neuronales sintéticos para la validación de resultados

"""
from __future__ import annotations
from typing import Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt


def hh_multirate(
    T: float = 400,
    dt: float = 0.01,
    mu1: float = 0.0,
    theta1: float = 1.0,
    sigma1: float = 3.0,
    mu2: float = 0.0,
    theta2: float = 1.0,
    sigma2: float = 2.0,
    cte1: float = 4.0,
    cte2: float = 4.0,
    g_syn: float = 3.0,
    E_syn: float = 10.0,
    alpha: float = 100,
    beta: float = 50,
    Tmax: float = 1e-2,
    V_th: float = -40.0,
    k: float = 0.5,
    tau: float = 1.0,
    tau_e: float  = 2.7,
    std_e: float  = 0.03,
    tau_i: float  = 10.5,
    std_i: float  = 0.006,
    ge0: float = 0.0121,
    gi0: float = 0.0573,
    use_OU_I1: bool = True,
    use_OU_I2: bool = False,
    plot: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    
    """ 
        Presynaptic Neuron 1 connected via a chemical synapse (continuous neurotransmitter release model) to postsynaptic Neuron 2. The Hodgkin–Huxley (HH)            dynamics follow a multirate scheme with a fast step for the membrane potential and a slow step for the ion channels. Neuron 1 is embedded in a complex         system through the point-conductance model (I_syn_sistem), and both neurons receive external input (I1 e I2) that can be Gaussian or Ornstein–                 Uhlenbeck (OU).
    
        Parameters
    ----------
    T :
        Total simulation time. T = 400

    dt :
        Time step.

    mu1, theta1, sigma1 :
        Parameters of noise I1 for neuron 1 (theta1 only for OU).

    mu2, theta2 sigma2 :
        Parameters of noise I2 for neuron 2  (theta2 only for OU).

    cte1, cte2 :
        Constant input currents added to each neuron.

    g_syn :
        Synaptic strength. A key parameter used to regulate the level of influence Neuron 1 exerts over Neuron 2.

    E_syn :
        Synaptic reversal potential. Determines whether the synapse is excitatory or inhibitory and to what degree. The higher the value (more positive), the          more excitatory the synapse becomes.

    alpha, beta :
        Synaptic kinetics parameters. Alpha regulates the binding rate of receptors to the neurotransmitter (how fast the channels open) and beta regulates            the channel closing rate (how fast the signal decays) 

    Tmax :
        Maximum neurotransmitter release. Increasing this value generally results in a higher synaptic current.

    V_th :
        Threshold for neurotransmitter release. A lower value makes the presynaptic neuron more sensitive.

    k :
        Slope of the sigmoid activation. A smaller k results in a steeper (more abrupt) slope, leading to higher neurotransmitter release for the same                  presynaptic potential.
    tau:
        Delay parameter in the transfer of information between neurons.

    tau_e, std_e:
        Parameters of Ornstein-Uhlenbeck noise for ge (excitatory conductance of I_syn_sistem)

    tau_i, std_i:
        Parameters of Ornstein-Uhlenbeck noise for gi (inhibitory conductance of I_syn_sistem)

    ge0, gi0:
        Synaptic strength of the excitatory/inhibitory conductance of I_syn_sistem
        
    use_OU_I1, use_OU_I2:
        If True, Ii noise is OU. If false, then the noise Ii is Gaussian.
        
    plot :
        If True, display the simulation plots. If False, no plots are shown.

    Returns
    -------
    V1 :
        Membrane potential of neuron 1.

    V2 :
        Membrane potential of neuron 2 (with synaptic coupling).
    """

    #np.random.seed(42)

    # ======================
    # PARÁMETROS HH
    # ======================
    C = 1.0
    gNa, gK, gL = 120.0, 36.0, 0.3
    ENa, EK, EL = 50.0, -77.0, -54.4

    # ======================
    # FUNCIONES HH
    # ======================
    def get_rates(V):
        am = 0.1*(V+40)/(1 - np.exp(-(V+40)/10))
        bm = 4*np.exp(-(V+65)/18)
        ah = 0.07*np.exp(-(V+65)/20)
        bh = 1/(1 + np.exp(-(V+35)/10))
        an = 0.01*(V+55)/(1 - np.exp(-(V+55)/10))
        bn = 0.125*np.exp(-(V+65)/80)
        return am, bm, ah, bh, an, bn
    

    def dVdt(V,m,h,n,I,I_syn):
        I_ion = gNa*(m**3)*h*(V-ENa) + gK*(n**4)*(V-EK) + gL*(V-EL)
        return (I - I_syn - I_ion)/C
        

    # ======================
    # TIEMPO
    # ======================
    dt_lento = dt*10
    V_steps = int(T / dt)
    delay_steps = int(tau / dt)
    canales_steps = int(T / dt_lento)
    t_eval = np.linspace(0, T, V_steps)

    # ======================
    # RUIDOS (vectorizado)
    # ======================
    if use_OU_I1:
        I1 = np.zeros(V_steps)
        exp_term = np.exp(-theta1 * dt)
        std_term = sigma1 * np.sqrt((1 - np.exp(-2 * theta1 * dt)) / (2 * theta1))
        for i in range(V_steps - 1):
            xi = np.random.randn()  
            I1[i+1] = I1[i] * exp_term + mu1 * (1 - exp_term) + std_term * xi        # solución exacta proceso OU
        I1 += cte1
    else:
        I1 = cte1 + mu1 + sigma1 * np.random.randn(V_steps)
        
    if use_OU_I2:
        I2 = np.zeros(V_steps)
        exp_term = np.exp(-theta2 * dt)
        std_term = sigma2 * np.sqrt((1 - np.exp(-2 * theta2 * dt)) / (2 * theta2))
        for i in range(V_steps - 1):
            xi = np.random.randn()  # N(0,1)
            I2[i+1] = I2[i] * exp_term + mu2 * (1 - exp_term) + std_term * xi        # solución exacta proceso OU
        I2 += cte2              
    else:
        I2 = cte2 + mu2 + sigma2 * np.random.randn(V_steps) 
    

    # ======================
    # SINAPSIS (entorno --> neurona 1)
    # ======================
    
    # (Valores típicos del paper Destexhe et al.)
    # Excitatorio
    #ge0 = 0.0121   (uS)
    Ee = 0.0      # (mV)
    
    # Inhibitorio
    # gi0 = 0.0573   (uS)
    Ei = -75.0    #  (mV)
    
    ge = np.zeros(V_steps)
    gi = np.zeros(V_steps)
    ge[0], gi[0] = ge0, gi0
    
    # ge
    exp_e = np.exp(-dt / tau_e)
    std_term_e = std_e * np.sqrt(1 - np.exp(-2 * dt / tau_e))
    
    # gi
    exp_i = np.exp(-dt / tau_i)
    std_term_i = std_i * np.sqrt(1 - np.exp(-2 * dt / tau_i))


    # ======================
    # SINAPSIS (neurona 1 --> neurona 2)
    # ======================
    def T_release(V_pre):
        return Tmax / (1 + np.exp(-(V_pre - V_th)/k))

    # ======================
    # INICIALIZACIÓN
    # ======================
    V1, m1, h1, n1 = np.zeros(V_steps), np.zeros(canales_steps), np.zeros(canales_steps), np.zeros(canales_steps)
    V2, m2, h2, n2 = np.zeros(V_steps), np.zeros(canales_steps), np.zeros(canales_steps), np.zeros(canales_steps)
    V2_no, m2n, h2n, n2n = np.zeros(V_steps), np.zeros(canales_steps), np.zeros(canales_steps), np.zeros(canales_steps)
    r = np.zeros(V_steps)

    V1[0] = V2[0] = V2_no[0] = -65.0
    am, bm, ah, bh, an, bn = get_rates(V1[0])
    m1[0] = m2[0] = m2n[0] = am/(am+bm)
    h1[0] = h2[0] = h2n[0] = ah/(ah+bh)
    n1[0] = n2[0] = n2n[0] = an/(an+bn)

    # ======================
    # BUCLE DE INTEGRACIÓN
    # Canales iónicos mediante integración exponencial y voltaje con Euler explícito 
    # ======================

    # para la primera iteración
    V1_med = V1[0]
    V2_med = V2[0]
    V2_no_med = V2_no[0]
    
    for j in range(canales_steps):

        # Calculamos el índice del paso siguiente controlando no desbordar arrays
        next_j = j + 1 if j + 1 < canales_steps else j      

        # ---- Canales iónicos en tiempo dt_lento=0.1 usando V_med----

            # Neurona 1 HH 
        am1, bm1, ah1, bh1, an1, bn1 = get_rates(V1_med)   
        tau_m = 1.0 / (am1 + bm1)  
        tau_h = 1.0 / (ah1 + bh1)   
        tau_n = 1.0 / (an1 + bn1)
        m_inf = am1 / (am1 + bm1)  
        h_inf = ah1 / (ah1 + bh1)      
        n_inf = an1 / (an1 + bn1)
        #Canales ionicos mediante integración exponencial
        m1[next_j] = m_inf + (m1[j] - m_inf) * np.exp(-dt_lento / tau_m) 
        h1[next_j] = h_inf + (h1[j] - h_inf) * np.exp(-dt_lento / tau_h)                     
        n1[next_j] = n_inf + (n1[j] - n_inf) * np.exp(-dt_lento / tau_n)

            # Neurona 2 (CON sinapsis) 
        am2, bm2, ah2, bh2, an2, bn2 = get_rates(V2_med)
        tau_m = 1.0 / (am2 + bm2)  
        tau_h = 1.0 / (ah2 + bh2)    
        tau_n = 1.0 / (an2 + bn2)
        m_inf = am2 / (am2 + bm2)  
        h_inf = ah2 / (ah2 + bh2)     
        n_inf = an2 / (an2 + bn2)
        #Canales ionicos mediante integración exponencial
        m2[next_j] = m_inf + (m2[j] - m_inf) * np.exp(-dt_lento  / tau_m) 
        h2[next_j] = h_inf + (h2[j] - h_inf) * np.exp(-dt_lento  / tau_h)                    
        n2[next_j] = n_inf + (n2[j] - n_inf) * np.exp(-dt_lento  / tau_n)
        
            # Neurona 2 (SIN sinapsis) 
        am2n, bm2n, ah2n, bh2n, an2n, bn2n = get_rates(V2_no_med)
        tau_m = 1.0 / (am2n + bm2n)  
        tau_h = 1.0 / (ah2n + bh2n)    
        tau_n = 1.0 / (an2n + bn2n)
        m_inf = am2n / (am2n + bm2n)  
        h_inf = ah2n / (ah2n + bh2n)      
        n_inf = an2n / (an2n + bn2n)
        #Canales ionicos mediante integración exponencial
        m2n[next_j] = m_inf + (m2n[j] - m_inf) * np.exp(-dt_lento  / tau_m) 
        h2n[next_j] = h_inf + (h2n[j] - h_inf) * np.exp(-dt_lento  / tau_h)                  
        n2n[next_j] = n_inf + (n2n[j] - n_inf) * np.exp(-dt_lento  / tau_n)  


        # ---- Potencial V en tiempo dt=0.01 ------
        for k in range (10): 
            i = j * 10 + k    # índice global
            if i >= V_steps -1:
                break

            # valores de la interpolación lineal de los canales para actualizar el potencial
            theta = (k+1)/10
            m1_interp = (1-theta)*m1[j] + theta*m1[next_j]
            h1_interp = (1-theta)*h1[j] + theta*h1[next_j]
            n1_interp = (1-theta)*n1[j] + theta*n1[next_j]
            m2_interp = (1-theta)*m2[j] + theta*m2[next_j]
            h2_interp = (1-theta)*h2[j] + theta*h2[next_j]
            n2_interp = (1-theta)*n2[j] + theta*n2[next_j]
            m2n_interp = (1-theta)*m2n[j] + theta*m2n[next_j]
            h2n_interp = (1-theta)*h2n[j] + theta*h2n[next_j]
            n2n_interp = (1-theta)*n2n[j] + theta*n2n[next_j]
            
        
            # ---- Corriente entrada neurona 1 desde el sistema ----
            # g excitatoria
            ge[i+1] = ge0 + (ge[i] - ge0) * exp_e + std_term_e * np.random.randn()
            # g inhibitoria                                                                        # Proceso OU (solución exacta)
            gi[i+1] = gi0 + (gi[i] - gi0) * exp_i + std_term_i * np.random.randn()
            # Evitar conductancias negativas
            ge[i+1] = max(ge[i+1], 0)
            gi[i+1] = max(gi[i+1], 0)
    
            I_syn_sistem = ge[i] * (V1[i] - Ee) + gi[i] * (V1[i] - Ei)    # Isyn = ge(t)*(V - Ee) + gi(t)*(V - Ei)
    
            # ---- RK4 para el voltaje  -----
            # Neurona 1
            k1 = dVdt(V1[i], m1_interp, h1_interp, n1_interp, I1[i], I_syn_sistem)
            k2 = dVdt(V1[i]+ 0.5*dt*k1, m1_interp, h1_interp, n1_interp, I1[i], I_syn_sistem)
            k3 = dVdt(V1[i]+ 0.5*dt*k2, m1_interp, h1_interp, n1_interp, I1[i], I_syn_sistem)
            k4 = dVdt(V1[i]+ dt*k3, m1_interp, h1_interp, n1_interp, I1[i], I_syn_sistem)
            V1[i+1] = V1[i] + dt*(k1+2*k2+2*k3+k4)/6 

            # ---- Sinapsis con delay neurona 1 --> neurona 2
            if i - delay_steps >= 0:
                V_pre_delayed = V1[i - delay_steps]
            else:
                V_pre_delayed = V1[0] 
            Tt = T_release(V_pre_delayed)
            r[i+1] = r[i] + (alpha*Tt*(1 - r[i]) - beta*r[i]) * dt
            I_syn = g_syn * r[i] * (V2[i] - E_syn)

            #  Neurona 2 (CON sinapsis)    
            k1 = dVdt(V2[i], m2_interp, h2_interp, n2_interp, I2[i], I_syn)
            k2 = dVdt(V2[i]+ 0.5*dt*k1, m2_interp, h2_interp, n2_interp, I2[i], I_syn)
            k3 = dVdt(V2[i]+ 0.5*dt*k2, m2_interp, h2_interp, n2_interp, I2[i], I_syn)
            k4 = dVdt(V2[i]+ dt*k3, m2_interp, h2_interp, n2_interp, I2[i], I_syn)
            V2[i+1] = V2[i] + dt*(k1+2*k2+2*k3+k4)/6 
    
            #  Neurona 2 (SIN sinapsis)     
            k1 = dVdt(V2_no[i], m2n_interp, h2n_interp, n2n_interp, I2[i], 0)
            k2 = dVdt(V2_no[i]+ 0.5*dt*k1, m2n_interp, h2n_interp, n2n_interp, I2[i], 0)
            k3 = dVdt(V2_no[i]+ 0.5*dt*k2, m2n_interp, h2n_interp, n2n_interp, I2[i], 0)
            k4 = dVdt(V2_no[i]+ dt*k3, m2n_interp, h2n_interp, n2n_interp, I2[i], 0)
            V2_no[i+1] = V2_no[i] + dt*(k1+2*k2+2*k3+k4)/6  

            
        # Valores medios del Volatje para la siguiente actualización de los canales
        end_idx = min((j + 1) * 10, V_steps)
        V1_med = np.mean(V1[j*10 : end_idx])
        V2_med = np.mean(V2[j*10 : end_idx])
        V2_no_med = np.mean(V2_no[j*10 : end_idx])


    # ======================
    # PLOTS
    # ======================
    if plot:
        plt.figure(figsize=(12,9))
    
        ax1 = plt.subplot(3,1,1)
        ax1.plot(t_eval, V1)
        ax1.set_title("Neurona presináptica (OU)")
        ax1b = ax1.twinx()
        ax1b.plot(t_eval, I1, alpha=0.3)
    
        ax2 = plt.subplot(3,1,2)
        ax2.plot(t_eval, V2_no)
        ax2.set_title("Postsináptica SIN sinapsis")
        ax2b = ax2.twinx()
        ax2b.plot(t_eval, I2, alpha=0.3)
    
        ax3 = plt.subplot(3,1,3)
        ax3.plot(t_eval, V2)
        ax3.set_title("Postsináptica CON sinapsis")
        ax3b = ax3.twinx()
        ax3b.plot(t_eval, I2, alpha=0.3)
    
        plt.tight_layout()
        plt.show()

    return V1, V2





def izhikevich(
    T: float = 1000,
    dt: float = 0.1,
    a1: float = 0.01,
    b1: float = 0.26,
    c1: float = -65.0,
    d1: float = 2.0,
    a2: float = 0.02,
    b2: float = 0.25,
    c2: float = -65.0,
    d2: float = 6.0,
    I1_const: float = 4.5,
    I2_const: float = 1.0,
    g_syn: float = 0.1,
    E_syn: float = -80.0,
    alpha: float = 5.0,
    beta: float = 0.1,
    Tmax: float = 1,
    V_th: float = 0.0,
    tmax: float = 30.0,
    tau_delay: float = 0.0,
    plot: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Simulación de dos neuronas Izhikevich acopladas mediante sinapsis química (modelo de neurotransmisores de liberación por pulso fijo)

    Neurona 1:
        Izhikevich + corriente constante I1

    Neurona 2:
        Izhikevich + corriente constante I2 + corriente sináptica

    Returns
    -------
    V1 : potencial membrana neurona 1
    V2 : potencial membrana neurona 2
    """

    # ======================
    # TIEMPO
    # ======================
    n_steps = int(T / dt)
    t_eval = np.arange(0, T, dt)
    delay_steps = int(tau_delay / dt)
    
    # ======================
    # CORRIENTES CONSTANTES
    # ======================
    I1 = np.ones(n_steps) * I1_const
    I2 = np.ones(n_steps) * I2_const

    # ======================
    # VARIABLES IZHIKEVICH
    # ======================
    V1 = np.zeros(n_steps)
    U1 = np.zeros(n_steps)
    V2 = np.zeros(n_steps)
    U2 = np.zeros(n_steps)

    # ======================
    # CONDICIONES INICIALES
    # ======================
    V1[0] = -65
    U1[0] = b1 * V1[0]

    V2[0] = -65
    U2[0] = b2 * V2[0]
    
    # ======================
    # SINAPSIS
    # ======================
    r = np.zeros(n_steps)
    release_steps = int(tmax/dt)
    release_counter = 0
    V_pre_previous = V1[0]
    
    # ======================
    # BUCLE SIMULACIÓN
    # ======================
    for i in range(n_steps-1):
        
        # NEURONA 1
        v = V1[i]
        u = U1[i]

        dv = (0.04*v**2+5*v+140-u+I1[i]) * dt
        du = (a1*(b1*v-u)) * dt

        V1[i+1] = v + dv
        U1[i+1] = u + du

        if V1[i+1] >= 30:
            V1[i] = 30.0
            V1[i+1] = c1
            U1[i+1] += d1

        # SINAPSIS
        if i-delay_steps >= 0:
            V_pre = V1[i-delay_steps]
        else:
            V_pre = V1[0]

        if V_pre_previous < V_th and V_pre >= V_th:
            release_counter = release_steps
        if release_counter > 0:
            Tt = Tmax
            release_counter -= 1
        else:
            Tt = 0.0
            
        V_pre_previous = V_pre

        # dinámica receptores
        r[i+1] = (r[i]+(alpha*Tt*(1-r[i])- beta*r[i])*dt)
        I_syn = g_syn*r[i+1]*(E_syn - V2[i])

        # NEURONA 2
        v = V2[i]
        u = U2[i]
        dv = (0.04*v**2+5*v+140-u+I2[i]-I_syn) * dt
        du = (a2*(b2*v-u)) * dt

        V2[i+1] = v + dv
        U2[i+1] = u + du

        if V2[i+1] >= 30:
            V2[i] = 30.0
            V2[i+1] = c2
            U2[i+1] += d2

    # ======================
    # PLOTS
    # ======================
    if plot:
        plt.figure(figsize=(12,8))
        ax1 = plt.subplot(2,1,1)
        ax1.plot(t_eval, V1)
        ax1.set_title("Neurona 1 Izhikevich (presináptica)")
        ax1.set_ylabel("V (mV)")
        ax1b = ax1.twinx()
        ax1b.plot(t_eval, I1, alpha=0.3)
        ax1b.set_ylabel("I1")
        ax2 = plt.subplot(2,1,2)
        ax2.plot(t_eval, V2)
        ax2.set_title("Neurona 2 Izhikevich + sinapsis")
        ax2.set_xlabel("Tiempo")
        ax2.set_ylabel("V (mV)")
        ax2b = ax2.twinx()
        ax2b.plot(t_eval, I2, alpha=0.3)
        ax2b.set_ylabel("I2")
        plt.tight_layout()
        plt.show()


    return V1, V2