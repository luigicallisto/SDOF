import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d  
import scipy.io as sio                 
import os
import urllib.request
from scipy.io import loadmat
import requests

# Configurazione della pagina Streamlit
st.set_page_config(page_title="NL Static analysis", layout="wide")

# =============================================================================
# FUNZIONI DI SUPPORTO
# =============================================================================

@st.cache_data # Scarica e tiene in memoria i file una volta sola per tutti gli utenti
def get_single_spectrum(name_spectra):
    # CORRETTO: Adesso è un dizionario chiave-valore valido per la ricerca
    file_mapping = {
        "SpettriN_5.mat": "https://www.dropbox.com/scl/fi/la33lpwojhwxvbqp00090/SpettriN_5.mat?rlkey=r5a0r3yxwhwnqg1pda1t9d3ke&st=y0p2ywqi&dl=1",
        "SpettriN_10.mat": "https://www.dropbox.com/scl/fi/ddnsldkhm644chety98n5/SpettriN_10.mat?rlkey=livs5fw39eidlcgp70gt4v54h&st=414j5vu1&dl=1",
        "SpettriN_15.mat": "https://www.dropbox.com/scl/fi/jfpg3gkvk44v8oi7nl0p0/SpettriN_15.mat?rlkey=a5rndfk3b7dj0xvi8zf0gnbw0&st=o22ea3ur&dl=1",
        "SpettriN_20.mat": "https://www.dropbox.com/scl/fi/od27dxjy0jxhayathaf5q/SpettriN_20.mat?rlkey=2326j6cqwuykkg38831lfrt7i&st=rqirbpu2&dl=1"
    }
    
    if name_spectra not in file_mapping:
        # Se lo smorzamento è fuori scala, restituisce matrici vuote di default
        return {"N_cyDBMean": np.zeros((1000, 21)), "N_cyDBMstd": np.zeros((1000, 21))}
        
    url = file_mapping[name_spectra]
    
    # Se sul server esiste un file vecchio corrotto (pesa pochi KB), lo eliminiamo
    if os.path.exists(name_spectra) and os.path.getsize(name_spectra) < 100000:
        os.remove(name_spectra)
    
    # Se il file non è presente sul server di Streamlit, lo scarica al volo
    if not os.path.exists(name_spectra):
        with st.spinner(f"Caricamento database sismico ({name_spectra}) da Dropbox..."):
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, name_spectra)
            
    # Carica in memoria il file richiesto
    return loadmat(name_spectra)

def intersect(x1, y1, x2, y2):
    x_common = np.unique(np.concatenate([x1, x2]))
    y1_interp = np.interp(x_common, x1, y1)
    y2_interp = np.interp(x_common, x2, y2)
    diff = y1_interp - y2_interp
    idx = np.where(np.diff(np.sign(diff)))[0]
    
    if len(idx) > 0:
        i = idx[0]
        x_int = x_common[i] - diff[i] * (x_common[i+1] - x_common[i]) / (diff[i+1] - diff[i])
        y_int = np.interp(x_int, x1, y1)
        return float(x_int), float(y_int)
    return 0.0, 0.0

def FN_SpectrumNorm(T, ag, Fo, Tc, csi, Cat, ST):
    T = np.asarray(T)
    eta = (10 / (5 + csi * 100)) ** 0.5
    if eta < 0.55: eta = 0.55
        
    if Cat == 'B': Smin, Smax, cost, slope, molt, esp = 1.0, 1.2, 1.4, 0.4, 1.1, -0.2
    elif Cat == 'C': Smin, Smax, cost, slope, molt, esp = 1.0, 1.5, 1.7, 0.6, 1.05, -0.33
    elif Cat == 'D': Smin, Smax, cost, slope, molt, esp = 0.9, 1.8, 2.4, 1.5, 1.25, -0.5
    elif Cat == 'E': Smin, Smax, cost, slope, molt, esp = 1.0, 1.6, 2.0, 1.1, 1.15, -0.4
    else: Smin, Smax, cost, slope, molt, esp = 1.0, 1.2, 1.0, 0.0, 1.0, 0.0

    S = np.clip(cost - slope * Fo * ag, Smin, Smax)
    Cc = molt * (Tc ** esp)
    Tc_mod = Tc * Cc
    Tb = Tc_mod / 3
    Td = 4 * ag + 1.6
    
    condlist = [(T >= 0) & (T < Tb), (T >= Tb) & (T < Tc_mod), (T >= Tc_mod) & (T < Td), (T >= Td)]
    funclist = [
        lambda t: ag * S * ST * eta * Fo * (t / Tb + 1 / (eta * Fo) * (1 - t / Tb)),
        lambda t: ag * S * ST * eta * Fo,
        lambda t: ag * S * ST * eta * Fo * (Tc_mod / t),
        lambda t: ag * S * ST * eta * Fo * (Tc_mod * Td / (t ** 2))
    ]
    return np.piecewise(T, condlist, funclist)

def FN_SpectrumEC8_1(T, ag, Type, GroundType, ST, csi):
    T = np.asarray(T)
    S_mat = np.array([[1.0, 1.0], [1.2, 1.35], [1.15, 1.5], [1.35, 1.8], [1.4, 1.6]])
    TB_mat = np.array([[0.15, 0.05], [0.15, 0.05], [0.2, 0.1], [0.2, 0.1], [0.15, 0.05]])
    TC_mat = np.array([[0.4, 0.25], [0.5, 0.25], [0.6, 0.25], [0.8, 0.3], [0.5, 0.25]])
    TD_mat = np.array([[2.0, 1.2], [2.0, 1.2], [2.0, 1.2], [2.0, 1.2], [2.0, 1.2]])
    
    ground_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    row = ground_mapping.get(GroundType.upper(), 0)
    col = int(Type) - 1
    
    S, TB, TC, TD = S_mat[row, col], TB_mat[row, col], TC_mat[row, col], TD_mat[row, col]
    eta = max(0.55, (10 / (5 + csi * 100)) ** 0.5)
    
    condlist = [(T >= 0) & (T < TB), (T >= TB) & (T < TC), (T >= TC) & (T < TD), (T >= TD)]
    funclist = [
        lambda t: ag * S * (1 + t / TB * (2.5 * eta - 1)),
        lambda t: ag * S * eta * 2.5,
        lambda t: ag * S * eta * 2.5 * (TC / t),
        lambda t: ag * S * eta * 2.5 * (TC * TD / (t ** 2))
    ]
    return np.piecewise(T, condlist, funclist) * ST

def FN_SpectrumEC8_2(T, S_alfa, S_beta, F_alfa, F_beta, FT, csi):
    T = np.asarray(T)
    S_alfa, S_beta = S_alfa * F_alfa * FT, S_beta * F_beta * FT
    TA, Chi, FA = 0.02, 4, 2.5
    TC = (S_beta / S_alfa) * 1.0
    TD = max(1.0 + S_beta * 9.81, 2.0)
    
    tc_div_chi = TC / Chi
    TB = 0.05 if tc_div_chi < 0.05 else (tc_div_chi if tc_div_chi < 0.1 else 0.1)
    eta_const = max(0.55, (10 / (5 + csi * 100)) ** 0.5)
    
    condlist = [(T >= 0) & (T < TA), (T >= TA) & (T < TB), (T >= TB) & (T < TC), (T >= TC) & (T < TD), (T >= TD)]
    
    def branch_2(t):
        denom = TB - TA
        if denom <= 0: return np.full_like(t, S_alfa / FA)
        eta_t = np.maximum(np.sqrt((10 + (((TB - t) / denom) ** 3) * (csi * 100 - 5)) / (5 + csi * 100)), 0.55)
        return (S_alfa / denom) * (eta_t * (t - TA) + (TB - t) / FA)

    funclist = [
        lambda t: np.full_like(t, S_alfa / FA),
        branch_2,
        lambda t: np.full_like(t, S_alfa * eta_const),
        lambda t: eta_const * S_beta * 1.0 / t,
        lambda t: eta_const * TD * S_beta * 1.0 / (t ** 2)
    ]
    return np.piecewise(T, condlist, funclist) * FT

# =============================================================================
# INTERFACCIA STREAMLIT (SIDEBAR & CONFIGURAZIONI)
# =============================================================================

st.title("Nonlinear Static Analysis of Earth Retaining Structures")
st.markdown("#### Luigi Callisto, Sapienza University of Rome, Italy")
st.link_button("Based on L. Callisto (2027), Earth Retaining Structures, Design and Seismic Performance, CRC press", "https://www.routledge.com/Earth-Retaining-Structures-Design-and-Seismic-Performance/Callisto/p/book/9781041148449")
st.link_button("More software", "https://luigicallisto.site.uniroma1.it/software")
st.markdown("---") 

st.sidebar.markdown('<p style="font-size:28px; font-weight:bold; color:#00E676;">Initial choices</p>', unsafe_allow_html=True)
# Parametri Generali
Type_System = st.sidebar.radio("System Type", ['D', 'ND'], help="D = Displacing, ND = Non-Displacing")
Spectrum_Option = st.sidebar.selectbox("Spectrum Option", ['NTC', 'EC81', 'EC82 (evolution)', 'Custom'])

# Parametri Struttura e Modello
st.sidebar.markdown('<p style="font-size:28px; font-weight:bold; color:#00E676;">Input quantities</p>', unsafe_allow_html=True)
H = st.sidebar.number_input("Excavation height H (m)", value=4.5, step=0.1, format="%.2f")
kC = st.sidebar.number_input("Critical seismic coefficient", value=0.42, step=0.01, format="%.3f")
alpha = st.sidebar.number_input("Hyperbola cut-off parameter Alfa", value=0.85, step=0.01, format="%.3f")
sC = st.sidebar.number_input("Normalised displacement at system capacity sC", value=0.05, step=0.01, format="%.3f")

if Type_System == 'D':
    betaD = st.sidebar.number_input("Unloading-reloading factor Beta", value=1.0, step=0.1)
    csi_ur = st.sidebar.number_input("Damping ratio in unloading/reloading (%)", value=1.0, step=1.0, format="%.0f")
    csi_ur = csi_ur/100
    confidenza = st.sidebar.number_input("No. standard deviation on equivalent cycles", value=0.0, step=0.1, format="%.1f")
else:
    betaD = 1.5  
    csi_ur = 0.0
    confidenza = 0

# Ingressi specifici in base allo spettro scelto
st.sidebar.markdown('<p style="font-size:28px; font-weight:bold; color:#00E676;">Spectral parameters</p>', unsafe_allow_html=True)
gr = 9.81
T = np.arange(0, 20.0, 0.005)

if Spectrum_Option == 'NTC':
    ag = st.sidebar.number_input("Outcrop PGA ag (g)", value=0.25, format="%.3f")
    Tc_g = st.sidebar.number_input("Corner period TC* (s)", value=0.38, format="%.3f")
    F0 = st.sidebar.number_input("Amplification factor F0", value=2.30, format="%.3f")
    Cat = st.sidebar.selectbox("Subsoil Category", ['A', 'B', 'C', 'D', 'E'], index=3)
    ST = st.sidebar.number_input("Topographic amplification factor ST", value=1.0, format="%.3f")
    csi = st.sidebar.number_input("Spectrum damping ratio (%)", min_value=0.0, max_value=100.0, value=5.0, step=1.0, format="%.0f")
    csi = csi/100
    Sa = FN_SpectrumNorm(T, ag, F0, Tc_g, csi, Cat, ST)

elif Spectrum_Option == 'EC81':
    ag = st.sidebar.number_input("Outcrop PGA ag (g)", value=0.25, format="%.3f")
    Type = st.sidebar.selectbox("Spectrum type (Type)", [1, 2], index=0)
    GroundType = st.sidebar.selectbox("GroundType", ['A', 'B', 'C', 'D', 'E'], index=1)
    ST = st.sidebar.number_input("Topographic amplification factor ST", value=1.0, format="%.3f")
    csi = st.sidebar.number_input("Spectrum damping ratio (%)", min_value=0.0, max_value=100.0, value=5.0, step=1.0, format="%.0f")
    csi = csi/100
    Sa = FN_SpectrumEC8_1(T, ag, Type, GroundType, ST, csi)

elif Spectrum_Option == 'EC82 (evolution)':
    S_alfa = st.sidebar.number_input("Max spectral acceleration S_Alfa (g)", value=0.7, format="%.3f")
    S_beta = st.sidebar.number_input("Spectral acceleration at T=1s, S_Beta (g)", value=0.5)
    F_alfa = st.sidebar.number_input("Amplification factor F_Alfa", value=1.0)
    F_beta = st.sidebar.number_input("Amplification factor F_Beta", value=1.0)
    FT = st.sidebar.number_input("Topographic amplification factor FT", value=1.0, format="%.3f")
    csi = st.sidebar.number_input("Spectrum damping ratio (%)", min_value=0.0, max_value=100.0, value=5.0, step=1.0, format="%.0f")
    csi = csi/100
    Sa = FN_SpectrumEC8_2(T, S_alfa, S_beta, F_alfa, F_beta, FT, csi)

elif Spectrum_Option == 'Custom':
    csi = st.sidebar.number_input("Spectrum damping ratio (%)", min_value=0.0, max_value=100.0, value=5.0, step=1.0, format="%.0f")
    csi = csi/100
    uploaded_file = st.sidebar.file_uploader("(xlsx, csv, txt format)", type=["xlsx", "csv", "txt"])
    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        if filename.endswith(".xlsx"):
            FFF = pd.read_excel(uploaded_file, header=None).values
        elif filename.endswith(".csv"):
            FFF = pd.read_csv(uploaded_file, header=None).values
        elif filename.endswith(".txt"):
            FFF = pd.read_csv(uploaded_file, header=None, sep=r"\s+", engine='python').values
        T_Custom = FFF[:, 0]
        Sa_Custom = FFF[:, 1]
        f_interp = interp1d(T_Custom, Sa_Custom, bounds_error=False, fill_value="extrapolate")
        Sa = f_interp(T)
    else:
        st.sidebar.warning("""
        ⚠️ Upload .xlsx, .txt, or .csv file  
        • Periods (s) in first column  
        • Spectral acceleration (g) in second column  
        • 🚫 No headers.
        """)
        Sa = np.zeros_like(T)
        st.stop()

# =============================================================================
# CALCOLO
# =============================================================================
Sd = Sa * gr / (4 * np.pi**2) * T**2

# Capacity curve
s = Sd / H
kH = kC * s / (s * alpha + sC * (1 - alpha))
kH[kH > kC] = kC

conv = 0
smorz = csi
PGA = Sa[0]
Sa_orig = Sa.copy()
Sd_orig = Sd.copy()

while conv == 0:
    eta = max((10 / (5 + smorz * 100))**0.5, 0.55)
    Sa = Sa_orig * eta
    Sd = Sd_orig * eta
    D0 = kC / (sC * (1 - alpha))    # rigidezza iniziale normalizzata
    T0 = 2 * np.pi * np.sqrt(H / (gr * betaD * D0))
    
    s_int, kH_int = intersect(s, kH, Sd / H, Sa)
    
    if Type_System == 'D':
        I = kC * s_int / alpha - sC * (1 - alpha) * kC / alpha**2 * np.log(1 + s_int * alpha / (sC * (1 - alpha)))
        Iel = kH_int**2 / (2 * betaD * D0)
        WD = I - Iel
        WE = kH_int * s_int / 2
        csi_calc = WD / (4 * np.pi * WE)
    elif Type_System == 'ND':
        sR = kC / D0  
        csi_calc = 4 / np.pi * (1 + sR / s_int) * (1 - (np.log(1 + s_int / sR)) / (s_int / sR)) - 2 / np.pi
        csi_max = 4 / np.pi * (1 + sR / sC) * (1 - (np.log(1 + sC / sR)) / (sC / sR)) - 2 / np.pi
        csi_calc = min(csi_calc, csi_max)
        
    err = abs(smorz - csi_calc)
    smorz = csi_calc
    if err < 0.0005:
        conv = 1

T0_D = (4 * np.pi**2 * H / (betaD * D0) / gr)**0.5  
smorz = smorz + csi_ur  
eta = max((10 / (5 + smorz * 100))**0.5, 0.55)

Sa_ur = Sa_orig * eta
Sd_ur = Sa_ur * gr / (4 * np.pi**2) * T**2

s_A, kH_A = intersect(s, s * D0 * betaD, Sd_ur / H, Sa_ur)

DampingValues = np.array([.055, .10, .15, .20])
idx = np.argmin(abs(DampingValues - smorz))
NearValue = DampingValues[idx]
NameSpectra = f"SpettriN_{int(NearValue * 100)}.mat"

if Type_System == 'D':
    soglia = min(kC / kH_A, 1.0)
    
    # CHIAMATA ON-DEMAND: Carichiamo solo il singolo spettro necessario da Dropbox
    mat_data = get_single_spectrum(NameSpectra)
    N_cyDBMean = np.asarray(mat_data['N_cyDBMean'])
    N_cyDBMstd = np.asarray(mat_data['N_cyDBMstd'])

    T_Nmat = np.arange(0.01, 10, 0.01)
    dT = 0.01
    idxT_arr = np.where(abs(T_Nmat - T0) < dT / 2)[0]
    idxT = int(idxT_arr[0]) if len(idxT_arr) > 0 else int(np.argmin(abs(T_Nmat - T0)))
    idxT = min(idxT, N_cyDBMean.shape[0] - 1)

    dsoglie = 0.05
    val_soglie = np.arange(0, 1 + dsoglie, dsoglie)
    idxS_arr = np.where(abs(val_soglie - soglia) < dsoglie / 2)[0]
    idxS = int(idxS_arr[0]) if len(idxS_arr) > 0 else int(np.argmin(abs(val_soglie - soglia)))
    idxS = min(idxS, N_cyDBMean.shape[1] - 1)

    neq_val = N_cyDBMean[idxT, idxS] + N_cyDBMstd[idxT, idxS] * confidenza
    Neq = float(np.asarray(neq_val).item())
else:
    Neq = 0

s1_tot = min(s_int, sC)
s1 = s1_tot - kH_int / (betaD * D0)
s_tot = s1 + Neq * s_A  

T_int = 2 * np.pi * (H / gr * s_int / kH_int)**0.5

# =============================================================================
# VISUALIZZAZIONE RISULTATI (UI)
# =============================================================================

st.markdown("""
    <style>
    [data-testid="stMetricLabel"] p {
        font-size: 20px !important;       
        font-weight: bold !important;     
        color: var(--text-color) !important; 
        opacity: 0.85;                                    
    }
    </style>
    """, unsafe_allow_html=True)

st.subheader("Results")
col1, col2, col3, col4 = st.columns(4)
col5, col6, col7, col8 = st.columns(4)

col1.metric("Initial Natural Period ($T_0$)", f"{T0:.2f} s")
col5.metric("Secant Natural Period ($T_{int}$)", f"{T_int:.2f} s")
col2.metric("Damping Ratio (Performance Point)", f"{(smorz-csi_ur)*100:.1f} %")
col6.metric("Max Acceleration ($k_{H,int}$)", f"{kH_int:.3f} g")
if Type_System == 'D':
    col3.metric("No. Equivalent Cycles ($N_{eq}$)", f"{Neq:.2f}")
    col7.metric("First Displacement", f"{s1*H:.3f} m")
    col8.metric("Permanent Displacement", f"{s1 * H:.3f} m")
elif Type_System == 'D':
    col8.metric("Max Transient Displacement", f"{s1_tot * H:.3f} m")
    

st.markdown("---")

# =============================================================================
# GENERAZIONE DEI GRAFICI CONDIZIONALI
# =============================================================================

if Type_System == 'D':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
else:
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 3.5))
    ax2 = None  

# --- Primo Subplot ---
ax1.plot(s, kH, label='Capacity curve', color='blue', linewidth=2)
ax1.plot(Sd_orig / H, Sa_orig, '--', label='Original Spectrum', color='gray')
ax1.plot(Sd / H, Sa, label='Damped Spectrum', color='orange', linewidth=2)
ax1.plot(s_int, kH_int, 'ro', markersize=8, label='Performance Point')
ax1.set_xlim(0, np.max(Sd_orig / H * 1.05))
ax1.set_ylim(0, np.max(Sa_orig * 1.05))
ax1.set_xlabel('d/H, Sd/H')
ax1.set_ylabel('kH, Sa (g)')
ax1.legend()
ax1.grid(True)

# --- Secondo Subplot (SOLO se Type_System è 'D') ---
if ax2 is not None:
    ax2.plot(s, s * D0 * betaD, label='Unloading/Reloading line', color='green', linewidth=2)
    ax2.plot(Sd_ur / H, Sa_ur, label='Unloading/Reloading Spectrum', color='purple', linewidth=2)
    ax2.plot(s_A, kH_A, 'ro', markersize=8, label='Intersection A')
    ax2.set_xlim(0, np.max(Sd_orig / H * 1.05))
    ax2.set_ylim(0, np.max(Sa_orig * 1.05))
    ax2.set_xlabel('d/H, Sd/H')
    ax2.set_ylabel('kH, Sa (g)')
    ax2.set_title('Cyclic response')
    ax2.legend()
    ax2.grid(True)

plt.tight_layout()
st.pyplot(fig, width='content')
