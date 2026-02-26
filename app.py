import streamlit as st
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser

st.title("🔍 Docling Node 분할 시각화")

uploaded_file = st.file_uploader("PDF 업로드", type="pdf")

if uploaded_file:
    # 임시 파일 저장 (Docling 로드를 위해)
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("문서 구조 분석 및 노드 분할 중..."):
        # 1. Docling으로 읽기
        reader = DoclingReader()
        documents = reader.load_data("temp.pdf")
        
        # 2. 노드로 분할
        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents(documents)

    st.success(f"분할 완료! 총 {len(nodes)}개의 노드가 생성되었습니다.")

    # 시각화
    for i, node in enumerate(nodes):
        with st.expander(f"📦 Node #{i+1} | 유형: {type(node).__name__}"):
            # 메타데이터 정보 표시
            st.caption(f"Page: {node.metadata.get('page_label', 'N/A')} | Length: {len(node.get_content())} chars")
            
            # 노드 내용 표시 (마크다운 렌더링)
            st.markdown("---")
            st.markdown(node.get_content())
            
            # 실제 원문(Raw Text) 확인
            with st.expander("원본 텍스트 보기"):
                st.code(node.get_content())