import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



st.markdown("""
### kelompok Visdat :
- Septian Si (0110223149)
- Dewi Fatimah Azzahra (0110223142)
- Muhamad Akbar Rabbani (0110223163)""", unsafe_allow_html=True)

# =============================================================================
# BAGIAN 1: PERSIAPAN DATA
# =============================================================================
# Karena saya tidak memiliki file data Anda, saya membuat fungsi untuk
# menghasilkan data contoh.
#
# --- GANTI BAGIAN INI DENGAN DATA ANDA ---
# Ganti bagian ini di file app.py Anda
@st.cache_data
def load_data():
    # df = pd.read_csv('nama_file_anda.csv')
    df = pd.read_csv('Datamentah.csv') # <-- UBAH DI SINI
    # Pastikan kolom tanggal diubah tipenya jika perlu
    # df['tglpo'] = pd.to_datetime(df['tglpo'])
    return df

# Memuat data
df = load_data()

# Memuat data
# Cukup ganti pemanggilan fungsi load_data() dengan pd.read_csv() atau metode lain
df = load_data()
# CONTOH:
# df = pd.read_csv('data_penjualan.csv')
# df['tglpo'] = pd.to_datetime(df['tglpo']) # Pastikan kolom tanggal berformat datetime
# --- AKHIR DARI BAGIAN YANG PERLU DIGANTI ---
# =============================================================================


# =============================================================================
# BAGIAN 2: KONFIGURASI APLIKASI STREAMLIT
# =============================================================================
st.set_page_config(layout="wide")
st.title("Dashboard Visualisasi Data Penjualan")

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("Navigasi Visualisasi")
pilihan_visualisasi = st.sidebar.selectbox(
    "Pilih Kategori Visualisasi:",
    [
        "Distribusi & Korelasi",
        "Analisis Time Series",
        "Analisis Performa",
        "Analisis Clustering",
    ]
)

# =============================================================================
# BAGIAN 3: TAMPILAN VISUALISASI BERDASARKAN PILIHAN
# =============================================================================

if pilihan_visualisasi == "Distribusi & Korelasi":
    st.header("Distribusi & Korelasi Data")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Total Bayar (Boxplot)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(y=df['total_bayar'], ax=ax)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'Rp {int(x):,}'))
        ax.set_title('Boxplot untuk Total Bayar')
        st.pyplot(fig)

    with col2:
        st.subheader("Matriks Korelasi")
        subset_df = df[['total_bayar', 'kode_perusahaan_id', 'marketing_id']]
        correlation_matrix = subset_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

elif pilihan_visualisasi == "Analisis Time Series":
    st.header("Analisis Runtun Waktu (Time Series)")
    st.subheader("Total Penjualan per Tahun")

    df_ts = df.copy()
    df_ts['tglpo'] = pd.to_datetime(df_ts['tglpo'])
    df_ts['tahun'] = df_ts['tglpo'].dt.to_period('Y')
    yearly_sales = df_ts.groupby('tahun')['total_bayar'].sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(yearly_sales.index.astype(str), yearly_sales, marker='o', color='b', linestyle='-', markersize=6, linewidth=2)
    ax.set_title('Total Bayar per Tahun', fontsize=16, fontweight='bold')
    ax.set_xlabel('Tahun', fontsize=12)
    ax.set_ylabel('Total Bayar', fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'Rp {int(x):,}'))
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

elif pilihan_visualisasi == "Analisis Performa":
    st.header("Analisis Performa dan Kontribusi")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Performa Berdasarkan Marketing")
        marketing_sales = df.groupby('marketing_name')['total_bayar'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        marketing_sales.plot(kind='bar', color='lightcoral', ax=ax)
        ax.set_title('Total Bayar Berdasarkan Marketing', fontsize=14, fontweight='bold')
        ax.set_xlabel('Nama Marketing', fontsize=10)
        ax.set_ylabel('Total Bayar', fontsize=10)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'Rp {int(x):,}'))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Kontribusi per Kategori Perusahaan")
        category_sales = df.groupby('kategori perushaan')['total_bayar'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        category_sales.plot(kind='bar', color='lightseagreen', ax=ax)
        ax.set_title('Total Bayar Berdasarkan Kategori Perusahaan', fontsize=14, fontweight='bold')
        ax.set_xlabel('Kategori Perusahaan', fontsize=10)
        ax.set_ylabel('Total Bayar', fontsize=10)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'Rp {int(x):,}'))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
    st.subheader("Proporsi Order per Kategori Perusahaan")
    counts = df['kategori perushaan'].value_counts()
    labels = counts.index.tolist()
    sizes  = counts.values.tolist()
    colors = ['gold', 'lightblue', 'lightcoral', 'limegreen', 'violet', 'skyblue']
    explode = [0.1] + [0] * (len(labels) - 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%',
           startangle=140, explode=explode, shadow=True)
    ax.set_title('Proporsi Order per Kategori Perusahaan', fontweight='bold')
    ax.legend(labels, loc='best')
    ax.axis('equal')
    st.pyplot(fig)

elif pilihan_visualisasi == "Analisis Clustering":
    st.header("Analisis Clustering")
    
    st.info("Pilih jenis clustering yang ingin dilihat:")
    cluster_type = st.radio(
        "Jenis Clustering:",
        ('Clustering Pelanggan (berdasarkan ID Perusahaan & Total Bayar)',
         'Clustering Marketing (berdasarkan ID Marketing & Total Bayar)'),
        horizontal=True
    )

    if 'Pelanggan' in cluster_type:
        st.subheader("1. Penentuan Jumlah Cluster Optimal (Elbow Method)")
        inertia = []
        k_values = range(1, 11)
        X = df[['kode_perusahaan_id', 'total_bayar']]
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_values, inertia, marker='o')
        ax.set_title('Elbow Method untuk Clustering Pelanggan')
        ax.set_xlabel('Jumlah Cluster (k)')
        ax.set_ylabel('Inertia')
        st.pyplot(fig)

        st.subheader("2. Visualisasi Hasil Clustering Pelanggan")
        k_optimal = st.slider("Pilih jumlah cluster (k) untuk visualisasi:", 2, 10, 3)
        
        kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='kode_perusahaan_id', y='total_bayar', hue='Cluster', palette='viridis', style='Cluster', s=100, ax=ax)
        ax.set_title('Clustering Perusahaan dan Total Bayar')
        ax.set_xlabel('Kode Perusahaan ID')
        ax.set_ylabel('Total Bayar')
        st.pyplot(fig)
        
    elif 'Marketing' in cluster_type:
        st.subheader("Visualisasi Hasil Clustering Marketing")
        
        df_cleaned = df.copy()
        features_clustering = ['marketing_id', 'total_bayar']
        X_clustering = df_cleaned[features_clustering]

        scaler = StandardScaler()
        X_clustering_scaled = scaler.fit_transform(X_clustering)

        kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
        df_cleaned['Cluster'] = kmeans.fit_predict(X_clustering_scaled)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df_cleaned, x='marketing_id', y='total_bayar', hue='Cluster',
            palette='viridis', style='Cluster', s=100, ax=ax
        )
        ax.set_title('Clustering Marketing dan Total Bayar')
        ax.set_xlabel('Marketing ID')
        ax.set_ylabel('Total Bayar')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)