import streamlit as st

st.header("Predições")
st.write("**Observe as predições geradas pelo modelo, juntamente com as probabilidades**:")
st.write("")

# Lista de imagens, predições e probabilidades
images = ["image1.jpg", "image2.jpg"]
preds = ["Highway", "Cafeteria"]
probabilidades_list = [
    {
        "airfield": 18.94,
        "bus interior": 20.41,
        "cafeteria": 19.26,
        "castle": 17.92,
        "highway": 23.46
    },
    {
        "airfield": 17.99,
        "bus interior": 19.63,
        "cafeteria": 22.51,
        "castle": 19.01,
        "highway": 20.86
    }
]

# Loop sobre as imagens e predições
for image, pred, probabilidades in zip(images, preds, probabilidades_list):
    col1, _, col2 = st.columns([2, 1, 3])  # Define o layout das colunas

    with col1:
        st.image(image, width=300)

    with col2:
        st.write(f"🔮 **Classe predita**: {pred.capitalize()}")
        st.write("📊 **Distribuição de Probabilidades:**")
        top_3 = sorted(probabilidades.items(), key=lambda x: x[1], reverse=True)[:3]
        for classe, prob in top_3:
            st.write(f"   → {classe.capitalize()}: {prob:.2f}%")


    st.write("-" * 35)