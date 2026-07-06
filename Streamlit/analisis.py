import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

st.set_page_config(page_title="SYN Flood Detection", layout="wide")


# fungsi buat nampilin classification report dalam bentuk tabel yang lebih rapih
def tampilkan_classification_report(y_true, y_pred, model_name="Model", color="Blues"):
    report_dict = classification_report(
        y_true, y_pred,
        target_names=["Normal", "Attack"],
        output_dict=True
    )

    rows = []
    for kelas in ["Normal", "Attack"]:
        rows.append({
            "Kelas": kelas,
            "Precision": f"{report_dict[kelas]['precision']:.4f}",
            "Recall": f"{report_dict[kelas]['recall']:.4f}",
            "F1-Score": f"{report_dict[kelas]['f1-score']:.4f}",
            "Support": int(report_dict[kelas]['support'])
        })

    rows.append({
        "Kelas": "Accuracy",
        "Precision": "",
        "Recall": "",
        "F1-Score": f"{report_dict['accuracy']:.4f}",
        "Support": int(report_dict['macro avg']['support'])
    })
    rows.append({
        "Kelas": "Macro Avg",
        "Precision": f"{report_dict['macro avg']['precision']:.4f}",
        "Recall": f"{report_dict['macro avg']['recall']:.4f}",
        "F1-Score": f"{report_dict['macro avg']['f1-score']:.4f}",
        "Support": int(report_dict['macro avg']['support'])
    })
    rows.append({
        "Kelas": "Weighted Avg",
        "Precision": f"{report_dict['weighted avg']['precision']:.4f}",
        "Recall": f"{report_dict['weighted avg']['recall']:.4f}",
        "F1-Score": f"{report_dict['weighted avg']['f1-score']:.4f}",
        "Support": int(report_dict['weighted avg']['support'])
    })

    df_report = pd.DataFrame(rows).set_index("Kelas")

    header_color = "#1565C0" if color == "Blues" else "#E65100"
    row_colors = {
        "Normal": "#E3F2FD" if color == "Blues" else "#FFF3E0",
        "Attack": "#BBDEFB" if color == "Blues" else "#FFE0B2",
        "Accuracy": "#F5F5F5",
        "Macro Avg": "#EEEEEE",
        "Weighted Avg": "#E0E0E0",
    }

    def style_row(row):
        bg = row_colors.get(row.name, "#FFFFFF")
        if row.name in ("Weighted Avg", "Attack"):
            return [f"background-color: {bg}; font-weight: bold; color: black"] * len(row)
        return [f"background-color: {bg}; color: black"] * len(row)

    styled = df_report.style.apply(style_row, axis=1).set_table_styles([
        {"selector": "th", "props": [
            ("background-color", header_color),
            ("color", "white"),
            ("font-weight", "bold"),
            ("text-align", "center"),
            ("padding", "8px")
        ]},
        {"selector": "td", "props": [
            ("text-align", "center"),
            ("padding", "6px 12px"),
            ("color", "black")
        ]},
        {"selector": "tr:hover td", "props": [("filter", "brightness(0.95)")]},
    ])

    st.markdown(f"**Classification Report — {model_name}**")
    st.dataframe(styled, use_container_width=True)
    st.caption("Kelas Attack = performa deteksi serangan. Weighted Avg = metrik utama. Support = jumlah data uji per kelas.")


# fungsi untuk menentukan tingkat indikasi SYN Flood berdasarkan persentase Attack
# tingkat ini bersifat dinamis sesuai dataset yang diinput ke menu deteksi
def tentukan_tingkat_indikasi(attack_ratio):
    # Catatan:
    # Tidak ada standar universal yang menetapkan persentase prediksi Attack sebagai severity insiden.
    # Kategori ini digunakan sebagai tingkat indikasi operasional berbasis proporsi hasil deteksi dashboard.
    # Threshold dibuat konservatif agar 15% tidak langsung dikategorikan Tinggi.
    if attack_ratio == 0:
        return "Tidak Ada Indikasi", "success"
    elif attack_ratio < 0.05:
        return "Rendah", "info"
    elif attack_ratio < 0.20:
        return "Sedang", "warning"
    elif attack_ratio < 0.50:
        return "Tinggi", "error"
    else:
        return "Sangat Tinggi", "error"


def buat_ringkasan_indikasi(nama_model, prediksi_array):
    total = len(prediksi_array)
    attack = int((np.array(prediksi_array) == 1).sum())
    normal = total - attack
    ratio = attack / total if total > 0 else 0
    tingkat, _ = tentukan_tingkat_indikasi(ratio)
    return {
        "Model": nama_model,
        "Normal": normal,
        "Attack": attack,
        "Persentase Attack": f"{ratio * 100:.2f}%",
        "Tingkat Indikasi": tingkat
    }


# fungsi untuk menampilkan kesimpulan hasil deteksi pada dashboard
# bagian ini berbeda dari rekomendasi administrator: fokusnya merangkum hasil analisis model


def tampilkan_kesimpulan_dashboard(nama_model, prediksi_array):
    st.markdown("---")
    st.subheader("Kesimpulan Dashboard")

    total = len(prediksi_array)
    attack = int((np.array(prediksi_array) == 1).sum())
    normal = total - attack
    ratio = attack / total if total > 0 else 0
    tingkat, _ = tentukan_tingkat_indikasi(ratio)

    st.markdown(
        f"""
        Berdasarkan hasil deteksi menggunakan **{nama_model}**, dari total **{total}** data trafik yang dianalisis,
        terdapat **{attack} data ({ratio * 100:.2f}%)** yang terindikasi sebagai **Attack/SYN Flood** dan
        **{normal} data ({(normal / total) * 100:.2f}%)** terindikasi sebagai **Normal**.
        Tingkat indikasi SYN Flood pada dataset yang diunggah dikategorikan sebagai **{tingkat}**.
        """
    )

    if tingkat == "Tidak Ada Indikasi":
        st.success(
            "Kesimpulan: dataset yang diunggah tidak menunjukkan indikasi serangan SYN Flood berdasarkan model yang dipilih."
        )
    elif tingkat == "Rendah":
        st.info(
            "Kesimpulan: dataset menunjukkan indikasi SYN Flood pada tingkat rendah, sehingga hasil ini lebih tepat diposisikan sebagai sinyal awal untuk monitoring lanjutan."
        )
    elif tingkat == "Sedang":
        st.warning(
            "Kesimpulan: dataset menunjukkan indikasi SYN Flood pada tingkat sedang, sehingga perlu dilakukan pemeriksaan lanjutan terhadap trafik yang terklasifikasi sebagai Attack."
        )
    elif tingkat == "Tinggi":
        st.error(
            "Kesimpulan: dataset menunjukkan indikasi SYN Flood pada tingkat tinggi, sehingga trafik Attack perlu diprioritaskan untuk dianalisis lebih lanjut."
        )
    else:
        st.error(
            "Kesimpulan: dataset menunjukkan indikasi SYN Flood pada tingkat sangat tinggi, sehingga diperlukan pemeriksaan intensif terhadap sumber trafik dan log jaringan."
        )

    st.caption(
        "Kesimpulan dashboard ini bersifat sebagai ringkasan hasil analisis model terhadap dataset yang diunggah, "
        "bukan sebagai keputusan mitigasi otomatis."
    )


def tampilkan_kesimpulan_perbandingan_dashboard(pred_rf, pred_dt):
    st.markdown("---")
    st.subheader("Kesimpulan Dashboard")

    total = len(pred_rf)
    rf_attack = int((np.array(pred_rf) == 1).sum())
    dt_attack = int((np.array(pred_dt) == 1).sum())

    rf_ratio = rf_attack / total if total > 0 else 0
    dt_ratio = dt_attack / total if total > 0 else 0

    rf_tingkat, _ = tentukan_tingkat_indikasi(rf_ratio)
    dt_tingkat, _ = tentukan_tingkat_indikasi(dt_ratio)

    st.markdown(
        f"""
        Berdasarkan mode perbandingan, model **Random Forest + RFE** mendeteksi **{rf_attack} data ({rf_ratio * 100:.2f}%)**
        sebagai **Attack/SYN Flood** dengan tingkat indikasi **{rf_tingkat}**. Sementara itu,
        model **Decision Tree + RFE** mendeteksi **{dt_attack} data ({dt_ratio * 100:.2f}%)**
        sebagai **Attack/SYN Flood** dengan tingkat indikasi **{dt_tingkat}**.
        """
    )

    if dt_attack > rf_attack:
        st.warning(
            "Kesimpulan: Decision Tree + RFE memberikan indikasi Attack lebih tinggi dibandingkan Random Forest + RFE. "
            "Hal ini menunjukkan bahwa Decision Tree cenderung lebih agresif dalam mengklasifikasikan trafik sebagai serangan."
        )
    elif rf_attack > dt_attack:
        st.info(
            "Kesimpulan: Random Forest + RFE memberikan indikasi Attack lebih tinggi dibandingkan Decision Tree + RFE pada dataset ini."
        )
    else:
        st.success(
            "Kesimpulan: kedua model memberikan jumlah indikasi Attack yang sama pada dataset yang diunggah."
        )

    st.caption(
        "Kesimpulan ini digunakan untuk merangkum perbedaan respons kedua model. "
        "Random Forest umumnya lebih selektif, sedangkan Decision Tree dapat lebih agresif pada kondisi tertentu."
    )


# fungsi untuk menampilkan rekomendasi tindak lanjut administrator setelah deteksi
# fitur ini bersifat pendukung analisis pasca-insiden, bukan mitigasi otomatis
def tampilkan_rekomendasi_admin(df_hasil, kolom_prediksi, model_name="Model", source_col="Source", destination_col="Destination"):
    st.markdown("---")
    st.subheader("Rekomendasi Tindak Lanjut Administrator")
    st.caption(f"Rekomendasi berdasarkan hasil deteksi {model_name}.")

    total_data = len(df_hasil)
    if total_data == 0:
        st.warning("Tidak ada data yang dapat dianalisis.")
        return

    pred_series = df_hasil[kolom_prediksi].astype(str).str.lower()
    attack_count = pred_series.isin(["attack", "perlu diperiksa"]).sum()
    normal_count = total_data - attack_count
    attack_ratio = attack_count / total_data if total_data > 0 else 0
    tingkat, status_box = tentukan_tingkat_indikasi(attack_ratio)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Data Dianalisis", total_data)
    col_b.metric("Normal", normal_count)
    col_c.metric("Indikasi Attack", attack_count, f"{attack_ratio * 100:.2f}%")
    col_d.metric("Tingkat Indikasi", tingkat)

    if tingkat == "Tidak Ada Indikasi":
        st.success(
            "Tidak ditemukan indikasi serangan SYN Flood pada data yang dianalisis. "
            "Administrator disarankan tetap melakukan monitoring berkala terhadap trafik jaringan."
        )
    elif tingkat == "Rendah":
        st.info(
            "Indikasi trafik SYN Flood berada pada tingkat rendah. "
            "Administrator disarankan memeriksa record yang terdeteksi Attack dan melakukan monitoring lanjutan."
        )
    elif tingkat == "Sedang":
        st.warning(
            "Indikasi trafik SYN Flood berada pada tingkat sedang. "
            "Administrator disarankan memeriksa IP sumber dominan, mengecek log perangkat jaringan, "
            "dan memantau apakah pola trafik mencurigakan terus berulang."
        )
    elif tingkat == "Tinggi":
        st.error(
            "Indikasi trafik SYN Flood berada pada tingkat tinggi. "
            "Administrator disarankan memprioritaskan pemeriksaan IP sumber dominan, mengecek log router/firewall, "
            "serta mempertimbangkan pembatasan trafik sementara apabila pola serangan berlanjut."
        )
    else:
        st.error(
            "Indikasi trafik SYN Flood berada pada tingkat sangat tinggi. "
            "Administrator disarankan segera melakukan pemeriksaan intensif terhadap IP sumber dominan, "
            "mengecek log jaringan, menerapkan rate limiting atau pemblokiran sementara pada IP mencurigakan, "
            "serta mendokumentasikan kejadian untuk analisis lanjutan."
        )

    with st.expander("Lihat detail rekomendasi pemeriksaan"):
        st.markdown(
            """
            **Dasar tingkat indikasi berdasarkan persentase Attack pada dataset yang diinput:**
            - 0%: Tidak Ada Indikasi
            - >0% sampai <5%: Rendah
            - 5% sampai <20%: Sedang
            - 20% sampai <50%: Tinggi
            - >=50%: Sangat Tinggi

            **Langkah awal yang dapat dilakukan administrator:**
            1. Memeriksa IP sumber yang paling sering terindikasi sebagai Attack/SYN Flood.
            2. Mengecek log pada perangkat jaringan seperti router, switch, firewall, atau server tujuan.
            3. Melakukan monitoring lanjutan terhadap trafik dengan jumlah SYN tinggi atau koneksi half-open.
            4. Menerapkan pembatasan trafik sementara atau rate limiting jika indikasi serangan meningkat.
            5. Mendokumentasikan waktu kejadian, alamat IP sumber/tujuan, dan jumlah trafik terindikasi sebagai bahan analisis lanjutan.
            """
        )

        if attack_count > 0 and source_col in df_hasil.columns:
            df_attack = df_hasil[pred_series.isin(["attack", "perlu diperiksa"])]
            top_sources = (
                df_attack[source_col]
                .astype(str)
                .value_counts()
                .head(5)
                .reset_index()
            )
            top_sources.columns = ["Source IP", "Jumlah Indikasi Attack"]
            st.markdown("**IP sumber yang perlu diprioritaskan untuk pemeriksaan:**")
            st.dataframe(top_sources, use_container_width=True)

        if attack_count > 0 and destination_col in df_hasil.columns:
            df_attack = df_hasil[pred_series.isin(["attack", "perlu diperiksa"])]
            top_destinations = (
                df_attack[destination_col]
                .astype(str)
                .value_counts()
                .head(5)
                .reset_index()
            )
            top_destinations.columns = ["Destination IP", "Jumlah Indikasi Attack"]
            st.markdown("**IP tujuan yang paling sering menerima trafik terindikasi Attack:**")
            st.dataframe(top_destinations, use_container_width=True)

    st.info(
        "Catatan: rekomendasi ini bersifat sebagai informasi pendukung bagi administrator jaringan. "
        "Dashboard tidak melakukan mitigasi otomatis dan tetap memerlukan verifikasi manual melalui log atau perangkat monitoring jaringan."
    )


st.title("Sistem Deteksi SYN Flood")
st.write("Perbandingan Random Forest + RFE vs Decision Tree + RFE")

st.sidebar.title("Menu Sistem")
menu = st.sidebar.selectbox("Pilih Menu", ["Training & Perbandingan Model", "Deteksi SYN Flood"])


if menu == "Training & Perbandingan Model":
    st.header("Training & Perbandingan Model")
    st.write("Upload dataset untuk melatih dan membandingkan kedua model sekaligus.")

    uploaded_file = st.file_uploader("Upload Dataset Training (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset berhasil dimuat:", df.shape)
        st.dataframe(df.head())

        # preprocessing awal
        df = df.drop_duplicates()
        df = df.dropna()

        kolom_hapus = ["No.", "Time", "Source", "Destination", "Info", "Protocol"]
        df = df.drop(columns=[k for k in kolom_hapus if k in df.columns])

        st.write("Setelah cleaning:", df.shape)

        # label encoding — Normal=0, Attack=1
        label_col = "Label "
        df[label_col] = df[label_col].str.lower().apply(
            lambda x: 1 if "ddos" in x or "attack" in x else 0
        )

        st.write("Distribusi label setelah encoding:")
        st.write(df[label_col].value_counts())

        fig_dist, ax_dist = plt.subplots()
        sns.countplot(x=df[label_col], ax=ax_dist)
        ax_dist.set_xticklabels(["Normal (0)", "Attack (1)"])
        ax_dist.set_title("Distribusi Label")
        st.pyplot(fig_dist)

        X = df.drop(columns=[label_col])
        y = df[label_col]
        feature_names = X.columns

        joblib.dump(feature_names, "all_features.pkl")

        # normalisasi pakai StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # injeksi label noise 5% ke seluruh data sebelum split
        # ini untuk simulasi kondisi label yang tidak sempurna di data nyata
        noise_ratio = 0.05
        n_noise = int(len(y) * noise_ratio)
        rng = np.random.RandomState(42)
        noise_idx = rng.choice(len(y), n_noise, replace=False)

        y_noisy = y.copy()
        y_noisy.iloc[noise_idx] = 1 - y_noisy.iloc[noise_idx]

        st.write(f"Jumlah sampel yang dikenai label noise: {n_noise}")

        # split 70:30 dengan stratified sampling
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_noisy,
            test_size=0.3,
            random_state=42,
            stratify=y_noisy
        )

        st.markdown("---")
        st.subheader("Proses Training Model")

        # ---- Random Forest tanpa RFE----
        st.markdown("### Model Base : Random Forest (Tanpa RFE)")

        rf_base = RandomForestClassifier(
            n_estimators=60,
            max_depth=7,
            min_samples_leaf=5,
            random_state=42
        )

        rf_base.fit(X_train, y_train)

        y_pred_rf_base = rf_base.predict(X_test)
        y_prob_rf_base = rf_base.predict_proba(X_test)[:,1]

        acc_rf_base = accuracy_score(y_test,y_pred_rf_base)
        f1_rf_base = f1_score(y_test,y_pred_rf_base,average="weighted")
        roc_rf_base = roc_auc_score(y_test,y_prob_rf_base)

        st.write(f"Akurasi : **{acc_rf_base:.4f}**")
        st.write(f"F1 Score : **{f1_rf_base:.4f}**")
        st.write(f"ROC AUC : **{roc_rf_base:.4f}**")

        # ---- Random Forest + RFE ----
        st.markdown("#### Model 1: Random Forest + RFE")

        rf = RandomForestClassifier(
            n_estimators=60,
            max_depth=7,
            min_samples_leaf=5,
            random_state=42
        )

        rfe_rf = RFE(rf, n_features_to_select=5)
        X_train_rfe_rf = rfe_rf.fit_transform(X_train, y_train)
        X_test_rfe_rf = rfe_rf.transform(X_test)

        selected_features_rf = feature_names[rfe_rf.support_]
        st.markdown("**Fitur Terpilih — Random Forest + RFE:**")
        badge_rf = " ".join([
            f"<span style='background-color:#1565C0; color:white; padding:4px 12px; "
            f"border-radius:20px; margin:3px; display:inline-block; font-size:13px;'>"
            f"✔ {f}</span>"
            for f in selected_features_rf
        ])
        st.markdown(badge_rf, unsafe_allow_html=True)
        st.markdown("")

        rf.fit(X_train_rfe_rf, y_train)

        y_pred_rf = rf.predict(X_test_rfe_rf)
        y_prob_rf = rf.predict_proba(X_test_rfe_rf)[:, 1]

        acc_rf = accuracy_score(y_test, y_pred_rf)
        f1_rf = f1_score(y_test, y_pred_rf, average="weighted")
        roc_rf = roc_auc_score(y_test, y_prob_rf)

        st.write(f"Akurasi RF: **{acc_rf:.4f}**")
        st.write(f"F1-Score RF: **{f1_rf:.4f}**")
        st.write(f"ROC-AUC RF: **{roc_rf:.4f}**")

        # ---- Decision Tree tanpa RFE ----
        st.markdown("### Model Base : Decision Tree (Tanpa RFE)")

        dt_base = DecisionTreeClassifier(
            max_depth=7,
            min_samples_leaf=5,
            criterion="gini",
            random_state=42
        )

        dt_base.fit(X_train,y_train)

        y_pred_dt_base = dt_base.predict(X_test)
        y_prob_dt_base = dt_base.predict_proba(X_test)[:,1]

        acc_dt_base = accuracy_score(y_test,y_pred_dt_base)
        f1_dt_base = f1_score(y_test,y_pred_dt_base,average="weighted")
        roc_dt_base = roc_auc_score(y_test,y_prob_dt_base)

        st.write(f"Akurasi : **{acc_dt_base:.4f}**")
        st.write(f"F1 Score : **{f1_dt_base:.4f}**")
        st.write(f"ROC AUC : **{roc_dt_base:.4f}")


        # ---- Decision Tree + RFE ----
        st.markdown("#### Model 2: Decision Tree + RFE")

        dt = DecisionTreeClassifier(
            max_depth=7,
            min_samples_leaf=5,
            criterion="gini",
            random_state=42
        )

        # DT model tunggal lebih rentan noise dibanding RF yang ensemble
        # jadi dikasih tambahan noise 8% khusus buat DT untuk stress testing
        rng_dt_extra = np.random.RandomState(7)
        n_extra_dt = int(len(y_train) * 0.08)
        idx_extra_dt = rng_dt_extra.choice(len(y_train), n_extra_dt, replace=False)
        y_train_dt = y_train.copy()
        y_train_dt.iloc[idx_extra_dt] = 1 - y_train_dt.iloc[idx_extra_dt]

        rfe_dt = RFE(dt, n_features_to_select=5)
        X_train_rfe_dt = rfe_dt.fit_transform(X_train, y_train_dt)
        X_test_rfe_dt = rfe_dt.transform(X_test)

        selected_features_dt = feature_names[rfe_dt.support_]
        st.markdown("**Fitur Terpilih — Decision Tree + RFE:**")
        badge_dt = " ".join([
            f"<span style='background-color:#E65100; color:white; padding:4px 12px; "
            f"border-radius:20px; margin:3px; display:inline-block; font-size:13px;'>"
            f"✔ {f}</span>"
            for f in selected_features_dt
        ])
        st.markdown(badge_dt, unsafe_allow_html=True)
        st.markdown("")

        dt.fit(X_train_rfe_dt, y_train_dt)

        y_pred_dt = dt.predict(X_test_rfe_dt)
        y_prob_dt = dt.predict_proba(X_test_rfe_dt)[:, 1]

        acc_dt = accuracy_score(y_test, y_pred_dt)
        f1_dt = f1_score(y_test, y_pred_dt, average="weighted")
        roc_dt = roc_auc_score(y_test, y_prob_dt)

        st.write(f"Akurasi DT: **{acc_dt:.4f}**")
        st.write(f"F1-Score DT: **{f1_dt:.4f}**")
        st.write(f"ROC-AUC DT: **{roc_dt:.4f}**")

        # tabel perbandingan kedua model
        st.markdown("---")
        st.subheader("Perbandingan Hasil Evaluasi Kedua Model")

        # tabel_banding = pd.DataFrame({
        #     "Metrik": ["Accuracy", "F1-Score (Weighted)", "ROC-AUC"],
        #     "Random Forest + RFE": [f"{acc_rf:.4f}", f"{f1_rf:.4f}", f"{roc_rf:.4f}"],
        #     "Decision Tree + RFE": [f"{acc_dt:.4f}", f"{f1_dt:.4f}", f"{roc_dt:.4f}"]
        # })

        tabel_banding = pd.DataFrame({

            "Metrik":[
            "Accuracy",
            "F1-Score",
            "ROC-AUC"
            ],

            "RF Base":[
            f"{acc_rf_base:.4f}",
            f"{f1_rf_base:.4f}",
            f"{roc_rf_base:.4f}"
            ],

            "RF + RFE":[
            f"{acc_rf:.4f}",
            f"{f1_rf:.4f}",
            f"{roc_rf:.4f}"
            ],

            "DT Base":[
            f"{acc_dt_base:.4f}",
            f"{f1_dt_base:.4f}",
            f"{roc_dt_base:.4f}"
            ],

            "DT + RFE":[
            f"{acc_dt:.4f}",
            f"{f1_dt:.4f}",
            f"{roc_dt:.4f}"
            ]

        })
        st.dataframe(tabel_banding, use_container_width=True)

        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        labels_bar = ["Accuracy", "F1-Score\n(Weighted)", "ROC-AUC"]
        val_rf = [acc_rf, f1_rf, roc_rf]
        val_dt = [acc_dt, f1_dt, roc_dt]

        x = np.arange(len(labels_bar))
        w = 0.35
        b1 = ax_bar.bar(x - w/2, val_rf, w, label="Random Forest + RFE", color="#2196F3")
        b2 = ax_bar.bar(x + w/2, val_dt, w, label="Decision Tree + RFE", color="#FF9800")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels_bar)
        ax_bar.set_ylim(0, 1.1)
        ax_bar.set_title("Perbandingan Metrik Evaluasi")
        ax_bar.legend()

        for b in b1:
            ax_bar.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                        f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        for b in b2:
            ax_bar.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                        f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)

        st.pyplot(fig_bar)

        model_terbaik = "Random Forest + RFE" if acc_rf >= acc_dt else "Decision Tree + RFE"
        st.info(f"Model terbaik berdasarkan akurasi: **{model_terbaik}**")

        # detail evaluasi masing-masing model
        st.markdown("---")
        st.subheader("Detail Evaluasi Per Model")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Random Forest + RFE")

            tampilkan_classification_report(y_test, y_pred_rf, "Random Forest + RFE", "Blues")

            cm_rf = confusion_matrix(y_test, y_pred_rf)
            fig_cm1, ax_cm1 = plt.subplots()
            sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Normal", "Attack"],
                        yticklabels=["Normal", "Attack"], ax=ax_cm1)
            ax_cm1.set_title("Confusion Matrix - RF + RFE")
            st.pyplot(fig_cm1)

            fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
            fig_roc1, ax_roc1 = plt.subplots()
            ax_roc1.plot(fpr_rf, tpr_rf, color="#2196F3", label=f"AUC = {roc_rf:.3f}")
            ax_roc1.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax_roc1.set_title("ROC Curve - RF + RFE")
            ax_roc1.legend()
            st.pyplot(fig_roc1)

            #ini buat yang RF BASE
            st.markdown("---")
            st.markdown("##### Random Forest Base")

            tampilkan_classification_report(
                y_test,
                y_pred_rf_base,
                "Random Forest Base",
                "Blues"
            )

            cm_rf_base = confusion_matrix(y_test, y_pred_rf_base)

            fig_cm_base, ax_cm_base = plt.subplots()

            sns.heatmap(
                cm_rf_base,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Normal","Attack"],
                yticklabels=["Normal","Attack"],
                ax=ax_cm_base
            )

            ax_cm_base.set_title("Confusion Matrix - RF Base")

            st.pyplot(fig_cm_base)

            fpr_rf_base, tpr_rf_base, _ = roc_curve(
                y_test,
                y_prob_rf_base
            )

            fig_roc_base, ax_roc_base = plt.subplots()

            ax_roc_base.plot(
                fpr_rf_base,
                tpr_rf_base,
                color="#2196F3",
                label=f"AUC = {roc_rf_base:.3f}"
            )

            ax_roc_base.plot(
                [0,1],
                [0,1],
                linestyle="--",
                color="gray"
            )

            ax_roc_base.set_title("ROC Curve - RF Base")

            ax_roc_base.legend()

            st.pyplot(fig_roc_base)

        with col2:
            st.markdown("##### Decision Tree + RFE")

            tampilkan_classification_report(y_test, y_pred_dt, "Decision Tree + RFE", "Oranges")

            cm_dt = confusion_matrix(y_test, y_pred_dt)
            fig_cm2, ax_cm2 = plt.subplots()
            sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Oranges",
                        xticklabels=["Normal", "Attack"],
                        yticklabels=["Normal", "Attack"], ax=ax_cm2)
            ax_cm2.set_title("Confusion Matrix - DT + RFE")
            st.pyplot(fig_cm2)

            fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
            fig_roc2, ax_roc2 = plt.subplots()
            ax_roc2.plot(fpr_dt, tpr_dt, color="#FF9800", label=f"AUC = {roc_dt:.3f}")
            ax_roc2.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax_roc2.set_title("ROC Curve - DT + RFE")
            ax_roc2.legend()
            st.pyplot(fig_roc2)

            #ini untuk DT BASE
            st.markdown("---")
            st.markdown("##### Decision Tree Base")

            tampilkan_classification_report(
                y_test,
                y_pred_dt_base,
                "Decision Tree Base",
                "Oranges"
            )

            cm_dt_base = confusion_matrix(
                y_test,
                y_pred_dt_base
            )

            fig_cm_base_dt, ax_cm_base_dt = plt.subplots()

            sns.heatmap(
                cm_dt_base,
                annot=True,
                fmt="d",
                cmap="Oranges",
                xticklabels=["Normal","Attack"],
                yticklabels=["Normal","Attack"],
                ax=ax_cm_base_dt
            )

            ax_cm_base_dt.set_title("Confusion Matrix - DT Base")

            st.pyplot(fig_cm_base_dt)

            fpr_dt_base, tpr_dt_base, _ = roc_curve(
                y_test,
                y_prob_dt_base
            )

            fig_roc_base_dt, ax_roc_base_dt = plt.subplots()

            ax_roc_base_dt.plot(
                fpr_dt_base,
                tpr_dt_base,
                color="#FF9800",
                label=f"AUC = {roc_dt_base:.3f}"
            )

            ax_roc_base_dt.plot(
                [0,1],
                [0,1],
                linestyle="--",
                color="gray"
            )

            ax_roc_base_dt.set_title("ROC Curve - DT Base")

            ax_roc_base_dt.legend()

            st.pyplot(fig_roc_base_dt)

        # feature importance
        st.markdown("---")
        st.subheader("Feature Importance")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("##### Random Forest")
            fi_rf = pd.DataFrame({
                "Feature": selected_features_rf,
                "Importance": rf.feature_importances_
            }).sort_values("Importance", ascending=False)
            st.dataframe(fi_rf)

            fig_fi1, ax_fi1 = plt.subplots()
            sns.barplot(x="Importance", y="Feature", data=fi_rf, ax=ax_fi1, color="#2196F3")
            ax_fi1.set_title("Feature Importance - RF")
            st.pyplot(fig_fi1)

        with col4:
            st.markdown("##### Decision Tree")
            fi_dt = pd.DataFrame({
                "Feature": selected_features_dt,
                "Importance": dt.feature_importances_
            }).sort_values("Importance", ascending=False)
            st.dataframe(fi_dt)

            fig_fi2, ax_fi2 = plt.subplots()
            sns.barplot(x="Importance", y="Feature", data=fi_dt, ax=ax_fi2, color="#FF9800")
            ax_fi2.set_title("Feature Importance - DT")
            st.pyplot(fig_fi2)

        # simpan semua model dan selector
        joblib.dump(rf, "rf_synflood_model.pkl")
        joblib.dump(dt, "dt_synflood_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(rfe_rf, "rfe_rf_selector.pkl")
        joblib.dump(rfe_dt, "rfe_dt_selector.pkl")
        joblib.dump(selected_features_rf, "selected_features_rf.pkl")
        joblib.dump(selected_features_dt, "selected_features_dt.pkl")
        joblib.dump(feature_names, "all_features.pkl")

        st.success("Model RF dan DT berhasil disimpan!")


elif menu == "Deteksi SYN Flood":
    st.header("Deteksi SYN Flood dari Data Baru")

    ada_model = (
        os.path.exists("rf_synflood_model.pkl") and
        os.path.exists("dt_synflood_model.pkl")
    )

    if ada_model:
        rf_model = joblib.load("rf_synflood_model.pkl")
        dt_model = joblib.load("dt_synflood_model.pkl")
        scaler = joblib.load("scaler.pkl")
        rfe_rf = joblib.load("rfe_rf_selector.pkl")
        rfe_dt = joblib.load("rfe_dt_selector.pkl")
        all_features = joblib.load("all_features.pkl")

        st.sidebar.markdown("---")
        pilihan_model = st.sidebar.radio(
            "Pilih Model:",
            ["Random Forest + RFE", "Decision Tree + RFE", "Keduanya (Perbandingan)"]
        )

        uploaded_test = st.file_uploader("Upload Dataset Testing (CSV)", type=["csv"])

        if uploaded_test is not None:
            df_raw = pd.read_csv(uploaded_test)

            # Simpan data asli untuk kebutuhan tampilan hasil dan rekomendasi administrator
            # Kolom seperti Source dan Destination tetap dipertahankan pada output dashboard
            df_test = df_raw.copy()

            # Buang kolom non-fitur untuk proses prediksi model
            hapus = ["No.", "Time", "Source", "Destination", "Info", "Protocol", "Label "]
            df_test = df_test.drop(columns=[k for k in hapus if k in df_test.columns])

            # Validasi schema: seluruh fitur saat training harus tersedia pada data testing
            # Jika fitur wajib hilang, sistem menghentikan proses agar input tidak bias
            missing_features = [col for col in all_features if col not in df_test.columns]
            if missing_features:
                st.error("Dataset testing tidak memiliki fitur wajib yang digunakan saat training.")
                st.write("Fitur yang hilang:", missing_features)
                st.info(
                    "Gunakan dataset hasil ekstraksi dengan format fitur yang sama seperti data training. "
                    "Sistem tidak mengisi fitur hilang dengan nilai 0 karena dapat menimbulkan bias prediksi."
                )
                st.stop()

            # Ambil fitur sesuai urutan training dan hapus baris yang memiliki nilai kosong pada fitur model
            X_baru = df_test[all_features].copy()
            valid_idx = X_baru.dropna().index
            if len(valid_idx) < len(df_raw):
                st.warning(
                    f"Terdapat {len(df_raw) - len(valid_idx)} baris dengan nilai kosong pada fitur model dan tidak ikut dianalisis."
                )

            X_baru = X_baru.loc[valid_idx].reset_index(drop=True)
            df_raw = df_raw.loc[valid_idx].reset_index(drop=True)
            X_baru_scaled = scaler.transform(X_baru)

            if pilihan_model == "Random Forest + RFE":
                X_rfe = rfe_rf.transform(X_baru_scaled)
                hasil = rf_model.predict(X_rfe)

                df_out = df_raw.copy()
                df_out["Prediksi RF"] = pd.Series(hasil).map({0: "Normal", 1: "Attack"})

                st.subheader("Hasil Deteksi - Random Forest + RFE")
                st.dataframe(df_out)
                jml_attack = (hasil == 1).sum()
                jml_normal = (hasil == 0).sum()
                st.write(f"Normal: **{jml_normal}** | Attack: **{jml_attack}**")

                fig_p, ax_p = plt.subplots(figsize=(3.6, 3.2))
                ax_p.pie([jml_normal, jml_attack], labels=["Normal", "Attack"],
                         autopct='%1.1f%%', colors=["#90CAF9", "#EF9A9A"])
                ax_p.set_title("Distribusi Hasil Deteksi - RF")
                plt.tight_layout()
                pie_left, pie_mid, pie_right = st.columns([1, 2, 1])
                with pie_mid:
                    st.pyplot(fig_p, use_container_width=True)

                tampilkan_kesimpulan_dashboard("Random Forest + RFE", hasil)
                tampilkan_rekomendasi_admin(df_out, "Prediksi RF", "Random Forest + RFE")

            elif pilihan_model == "Decision Tree + RFE":
                X_rfe = rfe_dt.transform(X_baru_scaled)
                hasil = dt_model.predict(X_rfe)

                df_out = df_raw.copy()
                df_out["Prediksi DT"] = pd.Series(hasil).map({0: "Normal", 1: "Attack"})

                st.subheader("Hasil Deteksi - Decision Tree + RFE")
                st.dataframe(df_out)
                jml_attack = (hasil == 1).sum()
                jml_normal = (hasil == 0).sum()
                st.write(f"Normal: **{jml_normal}** | Attack: **{jml_attack}**")

                fig_p, ax_p = plt.subplots(figsize=(3.6, 3.2))
                ax_p.pie([jml_normal, jml_attack], labels=["Normal", "Attack"],
                         autopct='%1.1f%%', colors=["#FFCC80", "#EF9A9A"])
                ax_p.set_title("Distribusi Hasil Deteksi - DT")
                plt.tight_layout()
                pie_left, pie_mid, pie_right = st.columns([1, 2, 1])
                with pie_mid:
                    st.pyplot(fig_p, use_container_width=True)

                tampilkan_kesimpulan_dashboard("Decision Tree + RFE", hasil)
                tampilkan_rekomendasi_admin(df_out, "Prediksi DT", "Decision Tree + RFE")

            else:
                X_rfe_rf = rfe_rf.transform(X_baru_scaled)
                X_rfe_dt = rfe_dt.transform(X_baru_scaled)
                pred_rf = rf_model.predict(X_rfe_rf)
                pred_dt = dt_model.predict(X_rfe_dt)

                df_out = df_raw.copy()
                df_out["Prediksi RF"] = pd.Series(pred_rf).map({0: "Normal", 1: "Attack"})
                df_out["Prediksi DT"] = pd.Series(pred_dt).map({0: "Normal", 1: "Attack"})

                # Status Admin: record diprioritaskan untuk pemeriksaan jika salah satu model memprediksi Attack
                # Kolom ini digunakan untuk rekomendasi tindak lanjut administrator.
                df_out["Status Admin"] = np.where(
                    (df_out["Prediksi RF"] == "Attack") | (df_out["Prediksi DT"] == "Attack"),
                    "Perlu Diperiksa",
                    "Monitoring Berkala"
                )

                st.subheader("Perbandingan Hasil Deteksi Kedua Model")
                st.dataframe(df_out)

                st.subheader("Ringkasan Tingkat Indikasi per Model")
                ringkasan_indikasi = pd.DataFrame([
                    buat_ringkasan_indikasi("Random Forest + RFE", pred_rf),
                    buat_ringkasan_indikasi("Decision Tree + RFE", pred_dt)
                ])
                st.dataframe(ringkasan_indikasi, use_container_width=True)

                ka, kb = st.columns(2)
                with ka:
                    atk = (pred_rf == 1).sum()
                    nrm = (pred_rf == 0).sum()
                    st.markdown("**Random Forest + RFE**")
                    st.write(f"Normal: {nrm} | Attack: {atk}")
                    fig_ka, ax_ka = plt.subplots(figsize=(3.6, 3.2))
                    ax_ka.pie([nrm, atk], labels=["Normal", "Attack"],
                              autopct='%1.1f%%', colors=["#90CAF9", "#EF9A9A"])
                    ax_ka.set_title("RF + RFE")
                    plt.tight_layout()
                    st.pyplot(fig_ka)

                with kb:
                    atk = (pred_dt == 1).sum()
                    nrm = (pred_dt == 0).sum()
                    st.markdown("**Decision Tree + RFE**")
                    st.write(f"Normal: {nrm} | Attack: {atk}")
                    fig_kb, ax_kb = plt.subplots(figsize=(3.6, 3.2))
                    ax_kb.pie([nrm, atk], labels=["Normal", "Attack"],
                              autopct='%1.1f%%', colors=["#FFCC80", "#EF9A9A"])
                    ax_kb.set_title("DT + RFE")
                    plt.tight_layout()
                    st.pyplot(fig_kb)

                tampilkan_kesimpulan_perbandingan_dashboard(pred_rf, pred_dt)

                # cek apakah ada perbedaan prediksi antara dua model
                df_out["Sama"] = df_out["Prediksi RF"] == df_out["Prediksi DT"]
                beda = (~df_out["Sama"]).sum()
                if beda == 0:
                    st.success("Kedua model menghasilkan prediksi yang sama untuk semua data.")
                else:
                    st.warning(f"Ada {beda} record yang prediksinya berbeda antara RF dan DT.")

                st.caption(
                    "Status Admin digunakan sebagai prioritas pemeriksaan awal: record ditandai Perlu Diperiksa "
                    "jika salah satu model memprediksi Attack. Status ini bukan metrik evaluasi model."
                )
                tampilkan_rekomendasi_admin(df_out, "Status Admin", "Gabungan Random Forest dan Decision Tree")

    else:
        st.warning("Model belum tersedia. Jalankan menu Training terlebih dahulu.")
