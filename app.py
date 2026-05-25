import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from acc_mod import acc_mod  # Assicurati che il file acc_mod.py sia nella stessa cartella

st.set_page_config(page_title="Seismic response of nonlinear SDOF", layout="wide")

st.title("Seismic response of DISPLACING earth retaining structures")
st.markdown("#### Luigi Callisto, Sapienza University of Rome, Italy")
st.link_button("Based on L. Callisto (2027), Earth Retaining Strucutres, Design and Seismic Performance, CRC press", "https://www.routledge.com/Earth-Retaining-Structures-Design-and-Seismic-Performance/Callisto/p/book/9781041148449")
st.link_button("More software", "https://luigicallisto.site.uniroma1.it/software")
st.markdown("---") # Una linea orizzontale di separazione

# --- SIDEBAR PER GLI INPUT ---
st.sidebar.header("INPUT PARAMETERS") 
H = st.sidebar.number_input("Excavation height (m)", value=3.5)
kc = st.sidebar.number_input("Critical seismic coefficient", value=0.2)
alfa = st.sidebar.number_input("Hyperbola cut-off parameter Alfa", value=0.8)
sC = st.sidebar.number_input("Normalised displacement at system capacity sC", value=0.02, format="%.3f")
beta = st.sidebar.number_input("Unloading-reloading factor Beta", value=1.0)
csi_ur = st.sidebar.number_input("Damping ratio in unloading-reloading", value=0.01, format="%.3f")
ampl = st.sidebar.number_input("Accelerogram amplification factor", value=1.0)
segno = st.sidebar.selectbox("Accelerogram sign", [1, -1])
dt_fixed = st.sidebar.number_input("Accelerogram sampling time (s)", value=0.02, format="%.3f")
n_div = st.sidebar.slider("No. of subdivisions in dt", 1, 500, 200)

# --- CARICAMENTO FILE ---
uploaded_file = st.sidebar.file_uploader(
    "Upload accelerogram file (xlx, csv, txt format)",
    type=["xlsx", "csv", "txt"]
)
# -------------------------------------------------------------------------------
def acc_mod(k, n_div, col=1):
    """
    Suddivide il vettore k in n_div sotto-intervalli
    """
    n = len(k)
    k_mod = np.zeros((n * n_div, col))
    for i in range(col):
        for j in range(n):
            k_mod[j * n_div, i] = k[j]
    for i in range(col):
        for j in range(n):
            a1 = (j - 1) * n_div
            a2 = j * n_div
            for jj in range(1, n_div):
                if j == 0:
                    k_mod[jj, i] = (
                        0
                        + (k_mod[a2, i] - 0) / n_div * jj
                    )
                else:
                    k_mod[(j - 1) * n_div + jj, i] = (
                        k_mod[a1, i]
                        + (k_mod[a2, i] - k_mod[a1, i]) / n_div * jj)
    return k_mod.flatten()
# -------------------------------------------------------------------------------
if uploaded_file is not None:

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
    n = FFF.shape[0]
    time_orig = np.arange(1, n + 1) * dt_fixed
    dt_acc = time_orig[1] - time_orig[0]
    a_eq = FFF[:, 0] * ampl * segno
    
    g = 9.81
    T0 = 2 * np.pi * np.sqrt(sC * H * (1 - alfa) / kc / g)
    # Mostra informazioni preliminari
    st.success("File loaded")  
    st.metric("Initial vatural period $T_0$", f"{T0:.3f} s")

    # --- LOGICA DI CALCOLO ---
    if n_div > 1:
        a_base = acc_mod(a_eq * g, n_div, 1)
        n = n * n_div
        dt = dt_acc / n_div
    else:
        a_base = acc_mod(a_eq * g, 1, 1)
        dt = dt_fixed

    kc_vect = np.ones(n) * kc
    time = np.arange(1, n + 1) * dt
    
    D0 = kc / sC / (1 - alfa)
    dx = np.zeros(n)
    vel = np.zeros(n)
    accel = np.zeros(n)
    kH = np.zeros(n)
    kHy = 0

    # Ciclo di integrazione
    for j in range(n - 1):
        accel[j + 1] = -a_base[j] - kH[j] * g
        vel[j + 1] = vel[j] + accel[j] * dt
        dx[j + 1] = dx[j] + vel[j] * dt + accel[j] * dt**2
        
        if dx[j + 1] > dx[j] and kH[j] >= kHy:
            num = kc_vect[j] * (dx[j + 1] / H)
            den = (dx[j + 1] / H * alfa + sC * (1 - alfa))
            kH[j + 1] = num / den
            if kH[j + 1] > kc_vect[j]:
                kH[j + 1] = kc_vect[j]
            kHy = kH[j + 1]
        else:
            D = beta * D0
            kH[j + 1] = (kH[j] + 2 * csi_ur * np.sqrt(D / g / H) * (vel[j + 1] - vel[j]) + D * (dx[j + 1] - dx[j]) / H)

    # Downsampling per i grafici
    time_plot = time[::n_div]
    kH_plot = kH[::n_div]
    dx_plot = dx[::n_div]
    kc_plot = kc_vect[::n_div]
    a_eq_plot = a_eq # Già della dimensione corretta rispetto al tempo originario

    # --- RISULTATI ---
    col1, col2 = st.columns(2)
    col1.metric("Final displacement", f"{dx[-1]:.3f} m")
    col2.metric("Max Acceleration", f"{np.max(kH):.3f} g")

    # --- GRAFICI RIDimensionati ---
    # Riducendo figsize e gestendo il layout miglioriamo la compattezza
    fig, axs = plt.subplots(
    2, 2,
    figsize=(15, 7),
    gridspec_kw={'width_ratios': [2, 1]}  # prima colonna più larga
    )

    # --- [0,0]
    axs[0,0].plot(time_plot, kH_plot, label="kH", linewidth=1)
    axs[0,0].plot(time_plot, -a_eq_plot, label="-a_base", alpha=0.7, linewidth=1)
    axs[0,0].plot(time_plot, kc_plot, label="kC", alpha=0.7, linewidth=1)
    
    axs[0,0].set_ylabel("a (g)")
    axs[0,0].legend(fontsize='small')
    axs[0,0].grid(True)
    
    # --- [0,1]
    axs[0,1].plot(dx_plot, kH_plot, color='orange')
    
    axs[0,1].set_xlabel("dx (m)")
    axs[0,1].set_ylabel("kH")
    
    # --- [1,0]
    axs[1,0].plot(time_plot, dx_plot, color='green')
    
    axs[1,0].set_xlabel("time (s)")
    axs[1,0].set_ylabel("dx (m)")
    
    # --- [1,1] vuoto
    axs[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()
     # Visualizzazione con larghezza controllata
    st.pyplot(fig, use_container_width=False)

        # --- DATAFRAME RISULTATI ---
    results_df = pd.DataFrame({
        "time_s": time_plot,
        "kH_g": kH_plot,
        "a_base_g": -a_eq_plot,
        "kC_g": kc_plot,
        "dx_m": dx_plot
    })

    # Conversione CSV
    csv = results_df.to_csv(index=False).encode('utf-8')

    # Pulsante download
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name="seismic_response_results.csv",
        mime="text/csv"
    )
    
else:
    st.info("Upload accelerogram from sidebar: xlsx, txt, or csv format, a single column of data expressed in g")

