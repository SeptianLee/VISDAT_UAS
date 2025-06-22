import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.ticker as ticker
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Data Penjualan",
    page_icon="📊",
    layout="wide"
)

# Judul aplikasi
st.title("📊 Analisis Data Penjualan")
st.write("Aplikasi ini melakukan analisis data penjualan dan visualisasi")

# Input path file lokal
st.sidebar.header("Konfigurasi Data")
file_path = st.sidebar.text_input(
    "Masukkan path lengkap file CSV Anda:",
    value="C:/Path/To/Your/Data Mentah_.csv"
)

# Fungsi untuk memuat data
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        
        # Preprocessing
        df['tglpo'] = pd.to_datetime(df['tglpo'], errors='coerce')
        df['total_bayar'] = df['total_bayar'].str.replace('Rp', '').str.replace(',', '').astype(float)
        df['marketing_id'] = df['marketing_id'].fillna(0).astype(int)
        df['marketing_name'].fillna(df['marketing_name'].mode()[0], inplace=True)
        df = df.drop_duplicates(subset=['tglpo', 'kode_perusahaan_id', 'marketing_id'])
        
        # Tambah kolom tahun
        df['tahun'] = df['tglpo'].dt.to_period('Y')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Memuat data
if os.path.exists(file_path):
    df = load_data(file_path)
    
    if df is not None:
        # Tampilkan data
        with st.expander("Tampilkan Data Mentah"):
            st.dataframe(df.head(100))
            st.write(f"Jumlah Baris: {df.shape[0]}, Jumlah Kolom: {df.shape[1]}")
            
        # Statistik Deskriptif
        with st.expander("Statistik Deskriptif"):
            st.dataframe(df.describe())
            
        # Visualisasi
        st.header("📈 Visualisasi Data")
        
        # Boxplot
        st.subheader("Distribusi Total Pembayaran")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(y=df['total_bayar'], ax=ax)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'Rp {x:,.0f}'))
        plt.title('Boxplot Total Pembayaran')
        st.pyplot(fig)
        
        # Analisis Waktu
        st.subheader("Analisis Tren Waktu")
        df['tahun'] = df['tahun'].astype(str)
        yearly_sales = df.groupby('tahun')['total_bayar'].sum()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(yearly_sales.index, yearly_sales, marker='o', color='b', linestyle='-', markersize=6, linewidth=2)
        ax.set_title('Total Bayar per Tahun', fontsize=16)
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Total Bayar')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'Rp {x:,.0f}'))
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Analisis Marketing
        st.subheader("Analisis Berdasarkan Marketing")
        marketing_sales = df.groupby('marketing_name')['total_bayar'].sum().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        marketing_sales.plot(kind='bar', color='lightcoral', ax=ax)
        ax.set_title('Total Bayar Berdasarkan Marketing', fontsize=16)
        ax.set_xlabel('Marketing Name')
        ax.set_ylabel('Total Bayar')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'Rp {x:,.0f}'))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Analisis Kategori Perusahaan
        st.subheader("Analisis Berdasarkan Kategori Perusahaan")
        category_sales = df.groupby('kategori perushaan')['total_bayar'].sum().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        category_sales.plot(kind='bar', color='lightseagreen', ax=ax)
        ax.set_title('Total Bayar Berdasarkan Kategori Perusahaan', fontsize=16)
        ax.set_xlabel('Kategori Perusahaan')
        ax.set_ylabel('Total Bayar')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'Rp {x:,.0f}'))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(category_sales):
            ax.text(i, v + 5000000, f'Rp {v:,.0f}', ha='center', va='bottom', fontsize=9)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Pie Chart Kategori
        st.subheader("Proporsi Order per Kategori")
        counts = df['kategori perushaan'].value_counts()

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(
            counts,
            labels=counts.index,
            autopct='%1.1f%%',
            startangle=140,
            shadow=True,
            colors=['gold', 'lightblue', 'lightcoral', 'limegreen', 'violet', 'skyblue']
        )
        ax.set_title('Proporsi Order per Kategori Perusahaan')
        st.pyplot(fig)
        
        # Clustering
        st.header("🧩 Analisis Clustering")
        
        # Pilihan fitur clustering
        clustering_option = st.radio(
            "Pilih fitur clustering:",
            ('Perusahaan', 'Marketing'),
            horizontal=True
        )

        if clustering_option == 'Perusahaan':
            features = ['kode_perusahaan_id', 'total_bayar']
            title = "Clustering Perusahaan dan Total Bayar"
            x_label = "Kode Perusahaan"
        else:
            features = ['marketing_id', 'total_bayar']
            title = "Clustering Marketing dan Total Bayar"
            x_label = "Marketing ID"

        # Slider jumlah cluster
        n_clusters = st.slider(
            "Jumlah Cluster",
            min_value=2,
            max_value=6,
            value=3,
            step=1
        )

        # Proses clustering
        X = df[features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        # Visualisasi clustering
        st.subheader("Visualisasi Cluster")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x=features[0],
            y=features[1],
            hue='Cluster',
            palette='viridis',
            style='Cluster',
            s=100,
            ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Total Bayar')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'Rp {x:,.0f}'))
        st.pyplot(fig)

        # Evaluasi clustering
        st.subheader("Evaluasi Cluster")
        silhouette = silhouette_score(X_scaled, kmeans.labels_)
        st.metric("Silhouette Score", f"{silhouette:.4f}")

        cluster_summary = df.groupby('Cluster')[features].mean()
        st.write("Rata-rata Fitur per Cluster:")
        st.dataframe(cluster_summary.style.format("{:.2f}"))
        
    else:
        st.warning("Gagal memuat data. Periksa format file Anda.")
else:
    st.warning("File tidak ditemukan. Silakan periksa path yang Anda masukkan.")

# Catatan penggunaan
st.sidebar.info("""
**Panduan Penggunaan:**
1. Masukkan path lengkap file CSV Anda (contoh: `C:/Data/Data_Mentah.csv`)
2. Pastikan file CSV memiliki kolom yang sesuai dengan kode asli
3. Aplikasi akan otomatis memproses dan menampilkan visualisasi
""")