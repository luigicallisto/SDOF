import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configurazione della pagina Streamlit
st.set_page_config(page_title="Analisi Dinamica Iwan-Mroz", layout="wide")

# =========================================================================
# FUNZIONI CORE DEL MODELLO
# =========================================================================

def calibraIM(N, alfa_cut, tau_f, G0):
    n_step = N
    gamma_f = tau_f / G0 / (1 - alfa_cut)
    d_gamma = gamma_f / n_step
    gamma = np.arange(n_step + 1) * d_gamma
    
    tau = np.zeros(n_step + 1)
    denom = gamma * alfa_cut + gamma_f * (1 - alfa_cut)
    tau[1:] = gamma[1:] * tau_f / denom[1:]
    
    G = np.zeros(n_step + 1)
    G[1:] = tau[1:] / gamma[1:]
    G[0] = G0
    
    H_mat = np.zeros(n_step + 1)
    R = np.zeros(n_step + 1)
    H_mat[0] = G[0]
    
    for i in range(1, n_step):
        H_mat[i] = (G[i+1] * gamma[i+1] - G[i] * gamma[i]) / (gamma[i+1] - gamma[i])
        R[i] = G[i] * gamma[i]
        
    R[n_step] = tau_f
    return R, H_mat


def calc_tau_IM(Q, dq, H_mat, R, Qalfa):
    N_size = len(R)
    flag = 0
    x = np.sign(dq)
    i = 0
    
    while flag == 0 and i < N_size:
        dQ = H_mat[i] * dq
        if abs(Q + dQ - Qalfa[i]) <= R[i]:
            Q = Q + dQ   
            if abs(Q - Qalfa[i]) < R[i] or i == N_size - 1:
                i_update = i - 1
            else:
                i_update = i
            for j in range(i_update + 1):
                Qalfa[j] = Q - x * R[j]
            flag = 1
        else:
            dQ = Qalfa[i] + x * R[i] - Q 
            Q = Q + dQ
            dq = dq - dQ / H_mat[i]
            i = i + 1
            
    if abs(Q) > R[N_size - 1]:
        Q = np.sign(Q) * R[N_size - 1]
        
    if i >= N_size:
        D_val = 0.0
    else:
        D_val = H_mat[i] 
        
    return Q, Qalfa, D_val


def acc_mod(k, n_div):
    n_len = len(k)
    k_mod = np.zeros(n_len * n_div)
    for j in range(n_len):
        k_mod[(j + 1) * n_div - 1] = k[j]
        
    for j in range(n_len):
        a1 = j * n_div - 1
        a2 = (j + 1) * n_div - 1
        for jj in range(1, n_div):
            if j == 0:
                k_mod[jj - 1] = (k_mod[a2] / n_div) * jj
            else:
                k_mod[j * n_div + jj - 1] = k_mod[a1] + (k_mod[a2] - k_mod[a1]) / n_div * jj
    return k_mod


# =========================================================================
# INTERFACCIA UTENTE (STREAMLIT SIDEBAR)
# =========================================================================
st.title("Seismic response of NON-DISPLACING earth retaining structures (Iwan - Mroz model)")
st.title("Luigi Callisto, Sapienza University of Rome, Italy")

st.sidebar.header("INPUT PARAMETERS") 
H = st.sidebar.number_input("Excavatiom height (m)", value=3.5, step=0.1)
kC = st.sidebar.number_input("Critical seismic coefficient", value=0.2, step=0.01)
alfa = st.sidebar.number_input("Hyperbola cut-off parameter Alfa", min_value=0.0, max_value=0.99, value=0.8, step=0.01)
sC = st.sidebar.number_input("Normalised displacement at system capacity sC", value=0.02, step=0.01, format="%.3f")
csi_ur = st.sidebar.number_input("Damping ratio", value=0.0001, step=0.0001, format="%.5f")
ampl = st.sidebar.number_input("Accelerogram amplification factor", value=1.0, step=0.1)
segno = st.sidebar.selectbox("Accelerogram sign", [1, -1], index=0)
dt_input = st.sidebar.number_input("Accelerogram sampling time (s)", value=0.02, step=0.0001, format="%.3f")
n_div = st.sidebar.number_input("No. of subdivisions in dt", value=10, min_value=1, step=1)
N = st.sidebar.number_input("No. of I-M rheological elements", value=200, min_value=10, step=10)

g = 9.81

# =========================================================================
# CARICAMENTO FILE
# =========================================================================
uploaded_file = st.sidebar.file_uploader(
    "Upload accelerogram file (xlx, csv, txt format)",
    type=["xlsx", "csv", "txt"]
)

if uploaded_file is not None:
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".xlsx"):
            FFF = pd.read_excel(uploaded_file, header=None).values

        elif filename.endswith(".csv"):
            FFF = pd.read_csv(uploaded_file, header=None).values

        elif filename.endswith(".txt"):
            FFF = pd.read_csv(
                uploaded_file,
                header=None,
                sep=r"\s+"
            ).values
        # Lettura dei dati
        #FFF = pd.read_excel(uploaded_file, header=None).to_numpy()
        n = FFF.shape[0]
        dt_acc = dt_input
        a_eq = FFF[:, 0] * ampl * segno
        amax = np.max(np.abs(a_eq))
        
        # Calcolo Periodo Iniziale
        T0 = 2 * np.pi * np.sqrt(sC * H * (1 - alfa) / (kC * g))
        
        # Mostra informazioni preliminari
        st.success("File loaded")
        st.metric("Initial natural period $T_0$", f"{T0:.3f} s")
        
        # Sotto-divisione Accelerogramma
        if n_div > 1:
            a_base = acc_mod(a_eq * g, n_div)
            n = n * n_div
            dt = dt_acc / n_div
        else:
            a_base = acc_mod(a_eq * g, 1)
            dt = dt_acc

        kc_vect = np.ones(n) * kC
        time = np.arange(n) * dt

        # Inizializzazione Vettori
        D0 = kC / (sC * (1 - alfa))
        smorz = csi_ur
        k_alfa_iniz = np.zeros(N)
        
        D = np.ones(n) * D0
        u = np.zeros(n)
        s = np.zeros(n)
        vel = np.zeros(n)
        accel = np.zeros(n)
        kh = np.zeros(n)

        # Calibrazione Modello
        R1, H1 = calibraIM(N, alfa, kC, D0)
        R_IM = R1[1:N+1]
        H_IM = H1[0:N]

        k_alfa = k_alfa_iniz.copy()
        D[0] = D0

        # Avanzamento della barra di calcolo su Streamlit
        # prog_bar = st.progress(0, text="Elaborazione dell'analisi dinamica...")

        # Analisi Dinamica (Linear Acceleration Method)
        for i in range(1, n - 1):
            m_sist = 1.0
            k_sist = m_sist * D[i-1] * g / H
            c_sist = 2 * smorz * np.sqrt(k_sist * m_sist)
            
            Kd = k_sist + 3 * c_sist / dt + 6 * m_sist / dt**2
            DP = -m_sist * (a_base[i] - a_base[i-1])
            DPd = DP \
                + m_sist * (6 / dt * vel[i-1] + 3 * accel[i-1]) \
                + c_sist * (3 * vel[i-1] + dt / 2 * accel[i-1])
            
            Du = DPd / Kd
            Dv = 3 / dt * Du - 3 * vel[i-1] - dt / 2 * accel[i-1]
            
            u[i] = u[i-1] + Du
            vel[i] = vel[i-1] + Dv
            s[i] = u[i] / H
            
            kh[i], k_alfa, D[i] = calc_tau_IM(kh[i-1], Du / H, H_IM, R_IM, k_alfa)
            
            if D[i] <= 1e-8:
                D[i] = 1e-8
                
            accel[i] = accel[i-1] + (
                -m_sist * (a_base[i] - a_base[i-1])
                - c_sist * (vel[i] - vel[i-1])
                - k_sist * (u[i] - u[i-1])
            ) / m_sist
            
            # Aggiorna la barra di avanzamento ogni tanto per non rallentare
            # if i % (max(1, n // 20)) == 0:
                # prog_bar.progress(i / (n - 1), text="Elaborazione in corso...")
                
        # prog_bar.empty()

        # Post-Processing e contrazione dei vettori
        acc_tot = -(accel + a_base)
        time_plot = time[::n_div]
        acc_tot_plot = acc_tot[::n_div]
        kh_plot = kh[::n_div]
        kc_plot = kc_vect[::n_div]
        dx_plot = u[::n_div]
        a_eq_plot = a_eq
       # =========================================================================
        # OUTPUT DEI RISULTATI
        # =========================================================================
        
        col1, col2, = st.columns(2)
        
        # Sfruttiamo col1 per il Periodo Iniziale se calcolato in precedenza
                   
        # Usiamo dx_plot per coerenza con le variabili dei grafici
        col1.metric("Final displacement", f"{dx_plot[-1]:.3f} m")
        
        if abs(np.min(kh_plot)) > np.max(kh_plot):
            max_acc = np.min(kh_plot)
        else:
            max_acc = np.max(kh_plot)
        col2.metric("Max Acceleration", f"{max_acc:.3f} g")

        # --- GRAFICI RIDIMENSIONATI ---
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        fig, axs = plt.subplots(
        2, 2,
        figsize=(8, 5),
        gridspec_kw={'width_ratios': [2, 1]}  # prima colonna più larga
        )

        # --- Sotto-grafico [0,0]: Accelerazioni nel tempo
        axs[0,0].plot(time_plot, kh_plot, label="kH (Response)", linewidth=1)
        axs[0,0].plot(time_plot, -a_eq_plot, label="-a_base (Input)", alpha=0.7, linewidth=1)
        axs[0,0].plot(time_plot, kc_plot, label="kC", alpha=0.7, linewidth=1)

        axs[0,0].set_ylabel("a (g)",fontsize=8)
        axs[0,0].legend(fontsize='small')
        axs[0,0].grid(True)
        
        # --- Sotto-grafico [0,1]: Ciclo Isteretico / Piano A-D
        axs[0,1].plot(dx_plot, kh_plot, color='orange', linewidth=1.2)

        axs[0,1].set_xlabel("dx (m)",fontsize=8)
        axs[0,1].set_ylabel("kH",fontsize=8)
        axs[0,1].grid(True)
        
        # --- Sotto-grafico [1,0]: Spostamento nel tempo
        axs[1,0].plot(time_plot, dx_plot, color='green', linewidth=1)
        
        axs[1,0].set_xlabel("time (s)",fontsize=8)
        axs[1,0].set_ylabel("dx (m)",fontsize=8)
        axs[1,0].grid(True)
        
        # --- Sotto-grafico [1,1]: Spazio vuoto nascosto
        axs[1,1].axis('off')
        
        plt.tight_layout()

        # Visualizzazione allineata al container di Streamlit
        st.pyplot(fig, use_container_width=True)

        # --- DATAFRAME RISULTATI ---
        results_df = pd.DataFrame({
            "time_s": time_plot,
            "kH_g": kh_plot,         # Corretto il nome della variabile in minuscolo
            "a_base_g": -a_eq_plot,
            "kC_g": kc_plot,
            "dx_m": dx_plot
        })

        # Conversione CSV
        csv = results_df.to_csv(index=False).encode('utf-8')

        # Pulsante download
        st.write("") # Un po' di spazio verticale prima del pulsante
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="seismic_response_results.csv",
            mime="text/csv"
        )
    except Exception as e:
        # Questo blocco DEVE stare qui per chiudere il "try:" iniziato sopra!
        st.error(f"There was an error {e}")
        
else:
    st.info("Upload accelerogram from sidebar: xlsx, txt, or csv format, a single column of data expressed in g")