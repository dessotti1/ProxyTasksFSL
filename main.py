import os
import math
import time
import warnings
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_navigation_bar import st_navbar

from utils.files import delete_files_recursively, save_images, is_folder_empty
from utils.engine import generate_mean_support_embeddings, generate_query_embeddings, predict_classes


warnings.filterwarnings("ignore", category=UserWarning)

SUPPORT_DIR = 'dataset_suporte'
QUERY_DIR = 'dataset_consulta'


screen = ["Few-Shot Learning", "Imagens"]

# Render the navbar; it returns the currently selected page name
st.session_state.screen = st_navbar(screen).lower()

# Initialize session state on first load
if 'page' not in st.session_state:
    delete_files_recursively("dataset_consulta")
    delete_files_recursively("dataset_suporte")
    st.session_state.page = 'select_model'

if st.session_state.screen == 'few-shot learning':

    if st.session_state.page == 'select_model':

        st.header("Selecione o modelo a ser usado")

        st.markdown("""
        📚 **O que são Tarefas Substitutas?**

        **Tarefas substitutas**, ou *proxy tasks*, são subtarefas relacionadas à tarefas principal 
        que permitem o acesso a dados supervisionados, mesmo quando os dados da tarefa principal
        possuem restrições.

        Essas tarefas ajudam o modelo a aprender padrões visuais associados que podem ser aplicados 
        posteriormente na tarefa principal, mesmo sem acesso direto a dados sensíveis ou complexos.
        """)

        proxy_tasks = {
            "Classificação de Cenas Internas": 
                "Identifica o tipo de ambiente analisando os objetos essenciais presentes na imagem."
        }

        selected_task = st.selectbox("**Selecione uma tarefa substituta**:", list(proxy_tasks.keys()))

        st.markdown(f"**📝 Descrição da tarefa selecionada**:")
        st.info(proxy_tasks[selected_task])

        if st.button("Avançar"):
            st.session_state.page = 'suporte'
            st.rerun()

    elif st.session_state.page == 'suporte':

        # Description of Few-Shot Learning
        st.header("Conjunto de Suporte")
        st.markdown("""
        📚 **Few-Shot Learning (FSL)**

        Few-Shot Learning é uma técnica de aprendizado de máquina que permite ao modelo generalizar a partir de **poucas imagens por classe** — geralmente de 1 a 20.

        Nesta etapa, você irá montar o **conjunto de suporte**, fornecendo **imagens rotuladas por classe**, que servirão como referência para o modelo classificar imagens desconhecidas mais adiante no conjunto de consulta.

        Essa tarefa é conhecida como **N-way K-shot**, onde:
        - **N** é o número de classes.
        - **K** é o número de imagens por classe.
        """)

        st.write("➡️**Insira imagens categorizadas**:")

        # User input for the class name and image upload
        classe = st.text_input("Digite o nome da classe:")
        imagens = st.file_uploader("Selecione as imagens da classe", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

        # Save the class images into the support set
        if st.button("Salvar essa classe no suporte"):
            if not classe:
                st.warning("Digite o nome da classe.")
            elif not imagens:
                st.warning("Selecione ao menos uma imagem.")
            else:
                save_images(imagens, 'dataset_suporte', classe)
                st.success(f"Imagens da classe '{classe}' salvas no suporte!")

        # Move to the query stage
        if st.button("Avançar para Consulta"):
            st.session_state.page = 'consulta'
            st.rerun()

        if st.button("Reiniciar conjunto suporte"):
            delete_files_recursively(SUPPORT_DIR)
            st.rerun()

    elif st.session_state.page == 'consulta':

        # Explain what the query set is
        st.header("Conjunto de Consulta")
        st.markdown(
            """
            📚 O conjunto consulta contém as imagens que você deseja classificar usando o modelo Few-Shot Learning. 
            Essas imagens serão comparadas com o conjunto suporte, que contém exemplos rotulados, para que o modelo 
            possa prever as classes das imagens do conjunto consulta, mesmo com poucas amostras por classe.

            > Em outras palavras, o conjunto consulta é o conjunto de dados "desconhecidos" que queremos identificar, 
            > utilizando o conhecimento aprendido a partir do conjunto suporte.
            """
        )
        st.write("")
        
        st.write("**Insira as imagens que serão categorizadas pelo modelo**:")

        # Upload query images
        imagens_consulta = st.file_uploader("Selecione as imagens do conjunto consulta", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

        # Save query images and run prediction
        if st.button("Salvar imagens no conjunto consulta"):
            if not imagens_consulta and len(os.listdir(QUERY_DIR)) == 0:
                st.warning("Selecione ao menos uma imagem.")
            else:
                save_images(imagens_consulta, 'dataset_consulta')
                st.success(f"{len(os.listdir(QUERY_DIR))} imagens presentes no conjunto consulta!")

                # Show a loading spinner while processing
                with st.spinner("Analisando imagens..."):
                    start_time = time.time()

                    # Generate embeddings for both sets and perform prediction
                    emb_sup = generate_mean_support_embeddings()
                    emb_con = generate_query_embeddings()

                    st.session_state.predicts = predict_classes(emb_sup, emb_con)
                    st.session_state.elapsed_time = time.time() - start_time 

                    # Move to the results page
                    st.session_state.page = 'resultados'
                    st.rerun()


    elif st.session_state.page == 'resultados':

        st.header("Predições")
        st.write("**Observe as predições geradas pelo modelo, juntamente com as probabilidades**:")
        st.write("")

        # Display each prediction result
        for img_path, resultado in st.session_state.predicts.items():
            col1, _, col2 = st.columns([2, 1, 3])

            # Show the image
            with col1:
                image = Image.open(img_path)
                st.image(image, width=300)

            # Show the predicted class
            with col2:
                st.write(f"🔮 **Classe predita**: {resultado['predicted_class'].capitalize()}")
                st.write("📊 **Distribuição de Probabilidades:**")
                for classe, prob in resultado["top3"]:
                    st.write(f"   → {classe.capitalize()}: {prob * 100:.2f}%")

            st.write("-" * 35)

        st.warning(f"Tempo de execução: {round(st.session_state.elapsed_time, 2)} segundos", icon="⏱️")

        st.write("-" * 35)

        # Create buttons in two columns
        col_suporte, col_consulta = st.columns(2)

        with col_suporte:
            st.markdown("### 🗂️ Suporte")

            if st.button("➕ Adicionar mais imagens ao suporte"):
                st.session_state.page = 'suporte'
                st.rerun()

            if st.button("🧹 Reiniciar suporte (limpar imagens)"):
                delete_files_recursively("dataset_suporte")
                st.success("Conjunto de suporte reiniciado!")
                st.session_state.page = 'suporte'
                st.rerun()

        with col_consulta:
            st.markdown("### 🔍 Consulta")

            if st.button("➕ Adicionar mais imagens à consulta"):
                st.session_state.page = 'consulta'
                st.rerun()

            if st.button("🧹 Reiniciar consulta (limpar imagens)"):
                delete_files_recursively("dataset_consulta")
                st.success("Conjunto de consulta reiniciado!")
                st.session_state.page = 'consulta'
                st.rerun()

        st.write("-" * 35)

        if st.button("🗑️ Reiniciar conjuntos de suporte e de consulta (limpar imagens)"):
                delete_files_recursively("dataset_suporte")
                delete_files_recursively("dataset_consulta")
                st.success("Conjunto de suporte reiniciado!")
                st.session_state.page = 'suporte'
                st.rerun()


elif st.session_state.screen == 'imagens':

    st.header("Conjunto de Suporte")

    data = []

    if not os.path.exists(SUPPORT_DIR) or is_folder_empty(SUPPORT_DIR):
        st.info("Sem imagens no conjunto de suporte.")
        df = pd.DataFrame(columns=["Nome da Imagem", "Classe", "Caminho"])
    else:
        for class_name in sorted(os.listdir(SUPPORT_DIR)):
            class_path = os.path.join(SUPPORT_DIR, class_name)
            if not os.path.isdir(class_path):
                continue
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(class_path, filename)
                    data.append({
                        "Nome da Imagem": filename,
                        "Classe": class_name,
                        "Caminho": full_path
                    })

        df = pd.DataFrame(data)

        total_images = len(df)
        total_classes = df['Classe'].nunique()
        st.markdown(f"**Total de classes:** {total_classes} | **Total de imagens:** {total_images}")

        classes = ['Todas classes'] + sorted(df['Classe'].unique().tolist())
        selected_class = st.selectbox("Filtre por classe:", classes)

        if selected_class != 'Todas classes':
            if st.button(f"❌ Deletar classe {selected_class}"):
                class_to_delete = os.path.join(SUPPORT_DIR, selected_class)
                if os.path.exists(class_to_delete):
                    import shutil
                    shutil.rmtree(class_to_delete)
                    st.success(f"Classe '{selected_class}' deletada com sucesso!")
                    st.rerun()  

        if selected_class != 'Todas classes':
            df_filtered = df[df['Classe'] == selected_class]
        else:
            df_filtered = df

        st.subheader("Relação de Imagens do Conjunto de Suporte")
        st.dataframe(
            df_filtered[['Nome da Imagem', 'Classe']].reset_index(drop=True), 
            use_container_width=True
        )

        st.subheader("Visualização das Imagens")

        # If filtering a single class, show images of that class in a grid of max 3 per row
        if selected_class != 'Todas classes':
            images_to_show = df_filtered
            st.markdown(f"**Classe: {selected_class}**")
            num_images = len(images_to_show)
            cols_per_row = 3
            rows = math.ceil(num_images / cols_per_row)

            for r in range(rows):
                cols = st.columns(cols_per_row)
                for c in range(cols_per_row):
                    idx = r * cols_per_row + c
                    if idx >= num_images:
                        break
                    img_path = images_to_show.iloc[idx]['Caminho']
                    image = Image.open(img_path)
                    with cols[c]:
                        st.image(image, width=150)
                        st.caption(images_to_show.iloc[idx]['Nome da Imagem'])

        else:
            # Show all classes separately, each with their images in grid 3 per row
            grouped = df_filtered.groupby('Classe')
            for class_name, group in grouped:
                st.markdown(f"### Classe: {class_name}")
                num_images = len(group)
                cols_per_row = 3
                rows = math.ceil(num_images / cols_per_row)

                for r in range(rows):
                    cols = st.columns(cols_per_row)
                    for c in range(cols_per_row):
                        idx = r * cols_per_row + c
                        if idx >= num_images:
                            break
                        img_path = group.iloc[idx]['Caminho']
                        image = Image.open(img_path)
                        with cols[c]:
                            st.image(image, width=120)
                            st.caption(group.iloc[idx]['Nome da Imagem'])

                st.write("-" * 35)