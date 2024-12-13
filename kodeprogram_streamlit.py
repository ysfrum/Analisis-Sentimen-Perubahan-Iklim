import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import re
import emoji
import nltk
import string
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Setup Streamlit app layout
st.sidebar.title("Menu")
page = st.sidebar.selectbox("Pilih Halaman", ["Beranda", "Analisis Data", "Evaluasi Model", "Prediksi Model"])

if page == "Beranda":
    st.title("Analisis Sentimen Perubahan Iklim")
    st.write("Jelajahi sentimen yang diungkapkan dalam tweet terkait iklim. Gunakan bilah sisi untuk menavigasi berbagai fitur aplikasi ini.")
    st.write("Aplikasi ini mencakup:")
    st.write("- Eksplorasi Data: Memvisualisasikan dan melakukan praproses data sentimen.")
    st.write("- Evaluasi Model: Meninjau metrik kinerja model dan confusion matrix.")
    st.write("- Prediksi Model: Masukkan teks untuk menganalisis sentimen.")

elif page == "Analisis Data":
    st.title("Eksplorasi Data Sentimen")
    st.write("Visualisasi data sentimen tweet terkait perubahan iklim.")

    # File input
    file_dataset = "dataset_skripsi_truefix.xlsx"
    if file_dataset is not None:
        dataset = pd.read_excel(file_dataset, sheet_name='data kotor')
        
        # Copy dataset and preprocess
        df_data = dataset.copy()
        df_data = df_data.replace({'sentiment': {'negatif': '0', 'positif': '1', 'netral': '2'}})

        # Distribusi Sentimen
        st.subheader("Distribusi Sentimen")
        fig, ax = plt.subplots()
        sns.countplot(data=df_data, x='sentiment', palette="viridis", ax=ax)
        ax.set_title("Distribusi Sentimen dalam Dataset")
        st.pyplot(fig)

        # Fungsi untuk preprocessing text
        start_preprocess_time = time.time()

        def text_clean(text):
            text = emoji.replace_emoji(text, replace="")
            text = re.sub(r"(@[A-Za-z0-9_]+)|(#\w+)|(\w+:\/\/\S+)", "", text)
            text = re.sub(r'RT @\w+: ', '', text)
            text = ''.join([ch for ch in text if ch not in string.punctuation])
            text = text.lower()
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\W', ' ', text)
            text = re.sub(r'\n', ' ', text)
            text = re.sub(r'\b\d+\b', '', text)
            text = re.sub(' +', ' ', text)
            return text.strip()

        extra_stopwords = ["yg", "amp"]
        stopwords_list = nltk.corpus.stopwords.words('indonesian') + extra_stopwords

        def remove_stopwords(text):
            words = text.split()
            return ' '.join([word for word in words if word.lower() not in stopwords_list])

        lemmatizer = WordNetLemmatizer()
        def lemmatize(text):
            # Melakukan lemmatization untuk setiap kata dalam teks
            return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        
        # factory = StemmerFactory()
        # stemmer = factory.create_stemmer()

        # def stem_text(text):
        #     return stemmer.stem(text)
        
        # Apply preprocessing
        st.write("Memulai pra-pemrosesan text...")
        df_data['text'] = df_data['text'].apply(text_clean)
        df_data['text'] = df_data['text'].apply(remove_stopwords)
        df_data['text'] = df_data['text'].apply(lemmatize)
        # stem_text = joblib.load('stemmed_texts.pkl')
        # df_data['text'] = stem_text
        end_preprocess_time = time.time()
        preprocessing_duration = end_preprocess_time - start_preprocess_time
        st.write(f"Lama pra-pemrosesan teks: {round(preprocessing_duration, 2)} detik")
        st.write("Pra-pemrosesan teks selesai.")

        # Top Words Plot
        # def plot_top_words(data, sentiment_label, title):
        #     data['text'] = data['text'].astype(str)  
        #     all_words = ' '.join([text for text in data[data['sentiment'] == sentiment_label]['text']])
        #     word_freq = Counter(all_words.split())
        #     common_words = word_freq.most_common(10)
        #     words, counts = zip(*common_words)
        #     plt.figure(figsize=(10, 5))
        #     sns.barplot(x=list(counts), y=list(words), palette="viridis")
        #     plt.title(f"20 Kata Teratas untuk Sentimen {title}")
        #     plt.show()

        # # Panggil untuk setiap sentimen
        # plot_top_words(df_data, '0', "Negatif")
        # plot_top_words(df_data, '1', "Positif")
        # plot_top_words(df_data, '2', "Netral")
        # Top Words Plot
        st.subheader("Top 10 kata Teratas untuk Setiap Sentimen")
        def plot_top_words(data, sentiment_label, title):
            data['text'] = data['text'].astype(str)
            texts = data[data['sentiment'] == sentiment_label]['text']
            if not texts.empty:
                all_words = ' '.join(texts)
                word_freq = Counter(all_words.split())
                common_words = word_freq.most_common(10)
                words, counts = zip(*common_words)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x=list(counts), y=list(words), palette="viridis", ax=ax)
                ax.set_title(f"10 Kata Teratas untuk Sentimen {title}")
                st.pyplot(fig)
            else:
                st.write(f"Tidak ada data untuk sentimen {title}")

        # Plot untuk setiap sentimen
        plot_top_words(df_data, '0', "Negatif")
        plot_top_words(df_data, '1', "Positif")
        plot_top_words(df_data, '2', "Netral")


        # WordCloud
        def generate_wordcloud(data, sentiment_label, title):
            all_words = ' '.join([text for text in data[data['sentiment'] == sentiment_label]['text']])
            wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(all_words)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis('off')
            ax.set_title(f"WordCloud untuk Sentimen {title}")
            st.pyplot(fig)

        st.subheader("Awan Kata untuk Setiap Sentimen")
        generate_wordcloud(df_data, '0', "Negatif")
        generate_wordcloud(df_data, '1', "Positif")
        generate_wordcloud(df_data, '2', "Netral")

        # N-grams Visualization
        def plot_top_ngrams(data, ngram_range=(2, 2), title="Bigrams"):
            vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=20, stop_words=stopwords_list)
            X = vectorizer.fit_transform(data)
            ngram_counts = X.toarray().sum(axis=0)
            ngrams = vectorizer.get_feature_names_out()
            ngram_freq = dict(zip(ngrams, ngram_counts))
            sorted_ngrams = sorted(ngram_freq.items(), key=lambda x: x[1], reverse=True)
            ngrams, counts = zip(*sorted_ngrams)
            fig, ax = plt.subplots()
            sns.barplot(x=list(counts), y=list(ngrams), palette="viridis", ax=ax)
            ax.set_title(f"Top {title}")
            st.pyplot(fig)

        st.subheader("Bigram dan Trigram")
        plot_top_ngrams(df_data['text'], ngram_range=(2, 2), title="Bigram")
        plot_top_ngrams(df_data['text'], ngram_range=(3, 3), title="Trigram")

        # Trend dari waktu ke waktu
        st.subheader("Trend Sentimen dari Waktu ke Waktu")
        df_data['date'] = pd.to_datetime(df_data['created_at'])
        sentiment_trend = df_data.groupby([df_data['date'].dt.to_period("M"), 'sentiment']).size().unstack().fillna(0)

        fig, ax = plt.subplots(figsize=(12, 6))
        sentiment_trend.plot(kind='line', marker='o', ax=ax)
        ax.set_title("Trend Sentimen dari Waktu ke Waktu")
        ax.set_xlabel("Waktu")
        ax.set_ylabel("Jumlah Tweet")
        ax.legend(["Negatif", "Positif", "Netral"])
        st.pyplot(fig)
    pass

elif page == "Evaluasi Model":
    st.title("Evaluasi Model")
    st.write("Perbandingan kinerja berbagai model klasifikasi sentimen.")

    # Load hasil evaluasi model (asumsikan sudah disimpan sebelumnya)
    model_metrics = {
       "LSTM": {
            "Accuracy": 0.60,
            "Precision": 0.60,
            "Recall": 0.59,
            "F1-score": 0.59,
            "Training Duration": 88.37,
            "Confusion Matrix": np.array([
                [74, 34, 30],
                [19, 130, 12],
                [40, 40, 62]
            ])
        },
        "Naive Bayes": {
            "Accuracy": 0.34,
            "Precision": 0.34,
            "Recall": 0.34,
            "F1-score": 0.33,
            "Training Duration": 0.48
        },
        "Linear SVC": {
            "Accuracy": 0.35,
            "Precision": 0.35,
            "Recall": 0.35,
            "F1-score": 0.35,
            "Training Duration": 0.36
        },
        "Ensemble Model": {
            "Accuracy": 0.42,
            "Precision": 0.42,
            "Recall": 0.42,
            "F1-score": 0.42,
            "Training Duration": 212.6
        }
    }

    # Create DataFrame with all metrics including training duration
    metrics_df = pd.DataFrame({
        'Model': list(model_metrics.keys()),
        'Accuracy': [m['Accuracy'] for m in model_metrics.values()],
        'Precision': [m['Precision'] for m in model_metrics.values()],
        'Recall': [m['Recall'] for m in model_metrics.values()],
        'F1-Score': [m['F1-score'] for m in model_metrics.values()],
        'Training Duration': [m['Training Duration'] for m in model_metrics.values()]
    })

    # Display metrics table
    st.subheader("Metrik Performa Model")
    st.dataframe(metrics_df.style.format({
        'Accuracy': '{:.2%}',
        'Precision': '{:.2%}',
        'Recall': '{:.2%}',
        'F1-Score': '{:.2%}',
        'Training Duration': '{:.2f}'
    }))

    # Create the combined metrics plot
    st.subheader("Perbandingan Metrik Model dan Durasi Pelatihan")
    fig = plt.figure(figsize=(12, 6))
    
    # Plotting all metrics in one graph
    x = np.arange(len(metrics_df['Model']))
    width = 0.15  # Width of the bars
    
    plt.bar(x - width*2, metrics_df['Accuracy'], width, label='Accuracy')
    plt.bar(x - width, metrics_df['Precision'], width, label='Precision')
    plt.bar(x, metrics_df['Recall'], width, label='Recall')
    plt.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score')
    plt.bar(x + width*2, metrics_df['Training Duration']/max(metrics_df['Training Duration']), width, label='Training Duration')
    
    plt.xlabel('Model')
    plt.ylabel('Nilai')
    plt.title('Metrik Performa')
    plt.xticks(x, metrics_df['Model'], rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    st.pyplot(fig)

    # Display Training History Images
    st.subheader("Riwayat Pelatihan Model LSTM")
    
    # Create two columns for the images
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("train_acc1.jpg", use_column_width=True)
    
    with col2:
        st.image("train_loss1.jpg", use_column_width=True)


    # Display confusion matrix
    st.subheader("Confusion Matrix (LSTM)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(model_metrics['LSTM']['Confusion Matrix'], 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Negatif', 'Positif', 'Netral'],
                yticklabels=['Negatif', 'Positif', 'Netral'])
    plt.title('Confusion Matrix dari Model LSTM')
    st.pyplot(fig)

elif page == "Prediksi Model":
    st.title("Prediksi Model")
    st.write("Pilih model yang ingin Anda gunakan untuk prediksi sentimen.")

    # Load models and necessary files
    lstm_model = tf.keras.models.load_model('lstm_model.keras')
    nb_model = joblib.load('naive_bayes.pkl')
    lsvc_model = joblib.load('linear_svc.pkl')
    ensemble_model = joblib.load('ensemble_model.pkl')
    tfidf_vectorizer = joblib.load('TF_IDF_Vectorization.pkl')

    model_options = {
        "LSTM": lstm_model,
        "Naive Bayes": nb_model,
        "Linear SVC": lsvc_model,
        "Ensemble Model": ensemble_model
    }
    model_choice = st.selectbox("Pilih Model", list(model_options.keys()))

    # [Fungsi preprocessing tetap sama]
     # Membuat objek stemmer dari Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Fungsi untuk preprocessing text
    def text_clean(text):
        text = emoji.replace_emoji(text, replace="")
        text = re.sub(r"(@[A-Za-z0-9_]+)|(#\w+)|(\w+:\/\/\S+)", "", text)
        text = re.sub(r'RT @\w+: ', '', text)
        text = ''.join([ch for ch in text if ch not in string.punctuation])
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(' +', ' ', text)
        return text.strip()

    stopwords_list = nltk.corpus.stopwords.words('indonesian')
    def remove_stopwords(text):
        words = text.split()
        return ' '.join([word for word in words if word.lower() not in stopwords_list])

    # Fungsi untuk preprocessing teks input
    def preprocess_text(text):
        # Bersihkan teks
        text_cleaned = text_clean(text)
        # Hapus stopwords
        text_no_stopwords = remove_stopwords(text_cleaned)
        # Stemming teks
        text_stemmed = ' '.join([stemmer.stem(word) for word in text_no_stopwords.split()])
        return text_stemmed
        
    # Input teks pengguna untuk prediksi
    user_input = st.text_area("Masukkan teks untuk prediksi sentimen:")

    if st.button("Predict"):
        if user_input:
            try:
                # Preprocess the input text
                processed_text = preprocess_text(user_input)

                if model_choice == "Naive Bayes":
                    input_vectorized = tfidf_vectorizer
                    if input_vectorized.shape[1] > nb_model.n_features_in_:
                        input_vectorized = input_vectorized[:, :nb_model.n_features_in_]
                    sentiment = nb_model.predict(input_vectorized)
                
                elif model_choice == "LSTM":
                    tokenizer = Tokenizer()
                    tokenizer.fit_on_texts([processed_text])
                    sequences = tokenizer.texts_to_sequences([processed_text])
                    max_len = 1722
                    input_padded = pad_sequences(sequences, maxlen=max_len, padding='post')
                    input_pred = np.reshape(input_padded, (input_padded.shape[0], 1, input_padded.shape[1]))
                    predictions = lstm_model.predict(input_pred)
                    sentiment = np.argmax(predictions, axis=1)

                else:
                    input_vectorized = tfidf_vectorizer
                    if input_vectorized.shape[1] > nb_model.n_features_in_:
                        input_vectorized = input_vectorized[:, :nb_model.n_features_in_]
                    sentiment = model_options[model_choice].predict(input_vectorized)

                # Standarisasi output prediksi
                sentiment_label = {0: "Negatif", 1: "Positif", 2: "Netral"}
                sentiment_color = {"Negatif": "red", "Positif": "green", "Netral": "gray"}
                result_label = sentiment_label.get(sentiment[0].item(), "Tidak diketahui")
                
                # Tampilkan hasil prediksi dengan penjelasan standar
                st.markdown(f"**Prediksi Sentimen:** <span style='color:{sentiment_color[result_label]}'>{result_label}</span>", unsafe_allow_html=True)
                
                # Tampilkan penjelasan berdasarkan standarisasi
                st.subheader("Penjelasan Hasil:")
                if result_label == "Positif":
                    st.write("""
                    Teks ini dikategorikan sebagai POSITIF karena menunjukkan satu atau lebih karakteristik berikut:
                    - Mengandung kata-kata kunci yang menyiratkan keyakinan atau harapan
                    - Menunjukkan dukungan terhadap upaya penanganan perubahan iklim
                    - Menyoroti solusi atau inovasi untuk mengatasi masalah perubahan iklim
                    """)
                elif result_label == "Negatif":
                    st.write("""
                    Teks ini dikategorikan sebagai NEGATIF karena menunjukkan satu atau lebih karakteristik berikut:
                    - Mengandung ungkapan skeptisisme terhadap perubahan iklim
                    - Menunjukkan penolakan terhadap konsensus ilmiah
                    - Mengekspresikan pesimisme terhadap upaya mitigasi
                    - Berisi konotasi negatif terhadap upaya penanganan perubahan iklim
                    """)
                else:  # Netral
                    st.write("""
                    Teks ini dikategorikan sebagai NETRAL karena menunjukkan satu atau lebih karakteristik berikut:
                    - Bersifat informatif atau deskriptif tanpa menunjukkan dukungan/penolakan
                    - Berisi laporan faktual tanpa nuansa emosional
                    - Mengutip informasi tanpa menyisipkan pendapat pribadi
                    """)

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning("Silakan masukkan teks terlebih dahulu untuk prediksi.")