import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as ticker
from sklearn.metrics import silhouette_score

# Konfigurasi Halaman
st.set_page_config(
    page_title="Analisis Data Penjualan",
    page_icon="📊",
    layout="wide"
)

# Judul Aplikasi
st.title("📊 Visualisasi Data Penjualan")

# Fungsi untuk Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('Data Mentah_.csv', encoding='latin-1')
    
    # Preprocessing
    df['tglpo'] = pd.to_datetime(df['tglpo'], errors='coerce')
    df['total_bayar'] = df['total_bayar'].str.replace('Rp', '').str.replace(',', '').astype(float)
    df['marketing_id'] = df['marketing_id'].fillna(0).astype(int)
    df['marketing_name'].fillna(df['marketing_name'].mode()[0], inplace=True)
    df = df.drop_duplicates(subset=['tglpo', 'kode_perusahaan_id', 'marketing_id'])
    
    return df

df = load_data()

# Sidebar
st.sidebar.header("Pengaturan Visualisasi")
selected_viz = st.sidebar.selectbox(
    "Pilih Visualisasi:",
    [
        "Statistik Deskriptif",
        "Trend Penjualan Tahunan",
        "Penjualan per Marketing",
        "Penjualan per Kategori Perusahaan",
        "Boxplot Total Bayar",
        "Heatmap Korelasi",
        "Clustering Perusahaan",
        "Clustering Marketing",
        "Area Chart Penjualan",
        "Proporsi Kategori Perusahaan"
    ]
)

# Tampilkan Data Mentah
if st.sidebar.checkbox("Tampilkan Data Mentah"):
    st.subheader("Data Mentah")
    st.dataframe(df)

# Visualisasi Berdasarkan Pilihan
if selected_viz == "Statistik Deskriptif":
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe())

elif selected_viz == "Trend Penjualan Tahunan":
    st.subheader("Trend Penjualan Tahunan")
    df['tahun'] = df['tglpo'].dt.to_period('Y')
    yearly_sales = df.groupby('tahun')['total_bayar'].sum()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly_sales.index.astype(str), yearly_sales, marker='o', color='b', linewidth=2)
    ax.set_title('Total Bayar per Tahun')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Total Bayar (Rp)')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'Rp{x:,.0f}'))
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

elif selected_viz == "Penjualan per Marketing":
    st.subheader("Penjualan per Marketing")
    top_n = st.slider("Jumlah Marketing Teratas:", 5, 20, 10)
    marketing_sales = df.groupby('marketing_name')['total_bayar'].sum().nlargest(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    marketing_sales.plot(kind='bar', color='lightcoral', ax=ax)
    ax.set_title(f'Top {top_n} Marketing Berdasarkan Total Bayar')
    ax.set_xlabel('Nama Marketing')
    ax.set_ylabel('Total Bayar (Rp)')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'Rp{x:,.0f}'))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

elif selected_viz == "Penjualan per Kategori Perusahaan":
    st.subheader("Penjualan per Kategori Perusahaan")
    category_sales = df.groupby('kategori perushaan')['total_bayar'].sum().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    category_sales.plot(kind='bar', color='lightseagreen', ax=ax)
    ax.set_title('Total Bayar Berdasarkan Kategori Perusahaan')
    ax.set_xlabel('Kategori Perusahaan')
    ax.set_ylabel('Total Bayar (Rp)')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'Rp{x:,.0f}'))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    # Tambahkan label nilai di atas bar
    for i, v in enumerate(category_sales):
        ax.text(i, v + 5000000, f'Rp{v:,.0f}', ha='center', va='bottom', fontsize=9)
    
    st.pyplot(fig)

elif selected_viz == "Boxplot Total Bayar":
    st.subheader("Distribusi Total Bayar")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(y=df['total_bayar'], ax=ax)
    ax.set_title('Distribusi Total Bayar')
    ax.set_ylabel('Total Bayar (Rp)')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'Rp{x:,.0f}'))
    st.pyplot(fig)

elif selected_viz == "Heatmap Korelasi":
    st.subheader("Heatmap Korelasi")
    subset_df = df[['total_bayar', 'kode_perusahaan_id', 'marketing_id']]
    correlation_matrix = subset_df.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Korelasi antar Variabel')
    st.pyplot(fig)

elif selected_viz == "Clustering Perusahaan":
    st.subheader("Clustering Perusahaan")
    
    # Persiapan data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['kode_perusahaan_id', 'total_bayar']])
    
    # Pilih jumlah cluster
    n_clusters = st.slider("Jumlah Cluster:", 2, 10, 3)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    
    # Visualisasi
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='kode_perusahaan_id',
        y='total_bayar',
        hue='Cluster',
        palette='viridis',
        style='Cluster',
        s=100,
        ax=ax
    )
    ax.set_title('Clustering Perusahaan dan Total Bayar')
    ax.set_xlabel('Kode Perusahaan')
    ax.set_ylabel('Total Bayar (Rp)')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'Rp{x:,.0f}'))
    st.pyplot(fig)
    
    # Tampilkan ringkasan cluster
    st.subheader("Ringkasan Cluster")
    cluster_summary = df.groupby('Cluster').agg({
        'kode_perusahaan_id': 'mean',
        'total_bayar': ['mean', 'count']
    })
    st.dataframe(cluster_summary)

elif selected_viz == "Clustering Marketing":
    st.subheader("Clustering Marketing")
    
    # Filter data
    df_cleaned = df.copy()
    scaler = StandardScaler()
    X_clustering = df_cleaned[['marketing_id', 'total_bayar']]
    X_clustering_scaled = scaler.fit_transform(X_clustering)
    
    # Pilih jumlah cluster
    n_clusters = st.slider("Jumlah Cluster:", 2, 10, 3)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_cleaned['Cluster'] = kmeans.fit_predict(X_clustering_scaled)
    
    # Visualisasi
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df_cleaned,
        x='marketing_id',
        y='total_bayar',
        hue='Cluster',
        palette='viridis',
        style='Cluster',
        s=100,
        ax=ax
    )
    ax.set_title('Clustering Marketing dan Total Bayar')
    ax.set_xlabel('Marketing ID')
    ax.set_ylabel('Total Bayar (Rp)')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'Rp{x:,.0f}'))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

elif selected_viz == "Area Chart Penjualan":
    st.subheader("Trend Penjualan per Bulan")
    df['tglpo'] = pd.to_datetime(df['tglpo'])
    monthly = df.groupby([df['tglpo'].dt.to_period('M'), 'kategori perushaan'])['total_bayar'].sum().unstack()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly.plot.area(ax=ax)
    ax.set_title('Total Bayar per Bulan per Kategori Perusahaan')
    ax.set_xlabel('Bulan-Tahun')
    ax.set_ylabel('Total Bayar (Rp)')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'Rp{x:,.0f}'))
    ax.legend(title='Kategori Perusahaan', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif selected_viz == "Proporsi Kategori Perusahaan":
    st.subheader("Proporsi Order per Kategori Perusahaan")
    counts = df['kategori perushaan'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['gold', 'lightblue', 'lightcoral', 'limegreen', 'violet', 'skyblue']
    explode = [0.1] + [0] * (len(counts) - 1)
    
    ax.pie(
        counts,
        labels=counts.index,
        colors=colors[:len(counts)],
        autopct='%1.1f%%',
        startangle=140,
        explode=explode,
        shadow=True
    )
    ax.set_title('Proporsi Order per Kategori Perusahaan')
    st.pyplot(fig)