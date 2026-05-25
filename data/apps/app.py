import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

st.set_page_config(
    page_title="Классификатор обращений",
    page_icon="📋",
    layout="wide"
)

API_URL = "http://localhost:8000"

st.title("Классификатор обращений граждан")
st.markdown("Автоматическое определение категории жалоб")

menu = st.sidebar.radio(
    "Навигация",
    ["Классификация", "Статистика", "Справка"]
)

if menu == "Классификация":
    st.header("Определение категории обращения")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ввод текста")
        text_input = st.text_area(
            "Введите текст жалобы:",
            height=200,
            placeholder="Пример: 'Не было влажной уборки. Много битого стекла на лестнице.'"
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            predict_btn = st.button("Определить категорию", type="primary", use_container_width=True)

    with col2:
        st.subheader("Результат")
        if predict_btn and text_input:
            try:
                response = requests.post(f"{API_URL}/predict", json={"text": text_input})
                if response.status_code == 200:
                    result = response.json()
                    st.markdown(f"**Категория:** {result['category']}")
                    st.markdown(f"**Вероятность:** {result['probability']}%")
                    st.progress(result['probability'] / 100)
                else:
                    st.error(f"Ошибка API: {response.status_code}")
            except Exception as e:
                st.error(f"Не удалось подключиться к API. Запустите api.py: {e}")
        else:
            st.info("Введите текст и нажмите кнопку")

elif menu == "Статистика":
    st.header("Статистика датасета")

    @st.cache_data
    def load_data():
        df = pd.read_csv('augmented_dataset.csv')
        return df

    df = load_data()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Общее количество", f"{len(df):,}")
    with col2:
        avg_len = int(df['text'].str.len().mean())
        st.metric("Средняя длина", f"{avg_len} символов")
    with col3:
        st.metric("Количество категорий", df['category'].nunique())
    with col4:
        st.metric("Макс. длина", f"{df['text'].str.len().max():,}")

    st.subheader("Распределение по категориям")
    category_counts = df['category'].value_counts()

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    category_counts.plot(kind='bar', ax=ax1)
    ax1.set_xlabel("Категория")
    ax1.set_ylabel("Количество")
    ax1.set_title("Количество жалоб по категориям")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig1)

    st.subheader("Доля категорий")
    fig2 = px.pie(values=category_counts.values, names=category_counts.index, title="Распределение")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Облако слов")
    all_text = ' '.join(df['text'].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=100,
                          stopwords=['это', 'так', 'вот', 'быть', 'весь', 'который', 'и']).generate(all_text)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.axis('off')
    ax3.set_title('Облако слов всех жалоб')
    st.pyplot(fig3)

    st.subheader("Распределение длины текстов")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    df['text_length'] = df['text'].str.len()
    df['text_length'].hist(bins=50, ax=ax4, color='skyblue', edgecolor='black')
    ax4.set_xlabel("Длина текста")
    ax4.set_ylabel("Количество")
    ax4.set_title("Гистограмма длины текстов жалоб")
    st.pyplot(fig4)


elif menu == "Справка":
    st.header("Руководство пользователя")

    with st.expander("Как использовать классификацию"):
        st.markdown("""
        1. Перейдите во вкладку **«Классификация»**
        2. Введите текст жалобы в текстовое поле
        3. Нажмите кнопку **«Определить категорию»**
        4. Результат появится справа 
        """)

    with st.expander("Статистика"):
        st.markdown("""
        - Общее количество обращений
        - Распределение по категориям (график + диаграмма)
        - Облако слов
        - Гистограмма длины текстов
        """)

    with st.expander("О модели"):
        st.markdown("""
        - **Модель:** Логистическая регрессия
        - **Векторизация:** TF-IDF (1000 признаков)
        - **Категорий:** 15
        - **Размер корпуса:** 57 058 жалоб
        - **Точность:** 90.72%
        """)