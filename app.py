import os
import uuid
import tempfile
import hashlib

import chromadb
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.core.schema import TextNode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser

# --- [1. 페이지 설정] ---
st.set_page_config(page_title="PaperBrain", layout="wide")
st.title("📚 논문 분석 비서 PaperBrain")

# Gemini API Key 설정 (Streamlit secrets 또는 환경 변수)
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')


# --- [2. 모델 로드] ---
@st.cache_resource
def load_models():
    # 작성하신 embedding.ipynb의 모델 설정 로직 반영
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    llm = GoogleGenAI(model="gemini-2.5-flash",
                      api_key=GOOGLE_API_KEY)
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    return embed_model, llm


embed_model, llm = load_models()


# --- [3. Session State 초기화] ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "is_analyzing" not in st.session_state:
    st.session_state.is_analyzing = False
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "current_paper_id" not in st.session_state:
    st.session_state.current_paper_id = None
    
# --- [4. Paper ID 해시 생성] ---
def make_paper_id(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


# --- [5. RAG 엔진 구축] ---
def build_rag_engine(uploaded_file):
    file_bytes = uploaded_file.read()
    paper_id = make_paper_id(file_bytes)
    
    # 임시 파일로 저장하여 DoclingReader로 읽기
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        # ChromaDB 설정
        db = chromadb.PersistentClient(path="./chromadb")
        chroma_collection = db.get_or_create_collection("paper_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 이미 인덱싱된 논문인지 확인 (작성하신 setup_db 로직 반영)
        existing = chroma_collection.get(where={"paper_id": paper_id}, include=[])
        
        if len(existing["ids"]) > 0:
            st.info(f"이미 분석된 논문입니다. 기존 VectorDB를 사용합니다.")
            index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        else:
            with st.spinner("지식 베이스 구축 중 (이 작업은 3~5분 정도 소요됩니다)..."):
                # Docling 파싱
                reader = DoclingReader()
                documents = reader.load_data(tmp_path)
                parser = MarkdownNodeParser()
                nodes = parser.get_nodes_from_documents(documents)
                
                for node in nodes:
                    node.metadata["paper_id"] = paper_id
                
                index = VectorStoreIndex(nodes, storage_context=storage_context)
                st.success(f"논문 분석 완료! ({len(nodes)}개 노드 생성)")
        

        qa_prompt = PromptTemplate(
            """
            당신은 논문을 설명하는 AI 연구 도우미입니다.
            
            다음 규칙을 반드시 지켜서 답변하세요.
            
            규칙:
            1. 반드시 한국어로 답변하세요.
            2. 제공된 Context 정보만 사용하세요.
            3. 답변에 사용한 정보는 반드시 [번호] 형식으로 출처를 표시하세요.
            4. Context에 없는 내용은 추측하지 말고 "모르겠습니다"라고 답하세요.
            5. 출처 표기시 문단 혹은 섹션 제목만 사용하세요.
            
            답변 형식:
            
            <한국어 답변>
            
            출처:
            [번호] 섹션 또는 문단 제목
            
            Context:
            ---------------------
            {context_str}
            ---------------------
            
            질문:
            {query_str}
            
            답변:
            """
        )

        filters = MetadataFilters(filters=[ExactMatchFilter(key="paper_id", value=paper_id)])
        retriever = index.as_retriever(similarity_top_k=20,
                                       filters=filters)

        query_engine = RetrieverQueryEngine.from_args(retriever=retriever,
                                                      text_qa_template=qa_prompt,
                                                      llm=Settings.llm)
        
        return query_engine, paper_id

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# --- [6. 채팅 세션 관리] ---
def create_or_update_chat(paper_id, prompt=None, force_new=False):
    """
    paper_id: 현재 논문 ID
    prompt: 첫 질문이 들어오면 세션 이름을 자동 생성
    force_new: True이면 기존 세션이 존재해도 새 세션 생성
    """    
    if paper_id not in st.session_state.chat_sessions:
        st.session_state.chat_sessions[paper_id] = {}

    sessions = st.session_state.chat_sessions[paper_id]
    
    # --- 새 세션 생성 조건 ---
    if force_new or len(sessions) == 0:
        session_id = str(uuid.uuid4())[:8]
        default_name = f"대화 {len(sessions)+1}"  # 기본 대화명
        sessions[session_id] = {"name": default_name,
                                "messages": [],
                                "id_default_name": True}
        st.session_state.current_session_id = session_id

        # 자동 생성 이고 prompt 있으면 이름 변경
        if prompt and not force_new:
            sessions[session_id]["name"] = prompt[:30]
            sessions[session_id]["id_default_name"] = False

        return session_id

    # --- 기존 마지막 세션 선택 ---
    session_id = list(sessions.keys())[-1]
    st.session_state.current_session_id = session_id
    session = sessions[session_id]

    # --- 이름 변경 규칙 ---
    if prompt and session.get("id_default_name", True):
            # 새 대화 버튼 클릭 후 첫 질문
            # 기본 세션명이면 prompt 기반으로 변경
            session["name"] = prompt[:30]
            session["id_default_name"] = False

    return session_id

    
# --- [7. Sidebar] ---
with st.sidebar:
    # 논문 업로드 및 분석
    st.header("📄 논문 업로드")
    uploaded_file = st.file_uploader("PDF 파일을 선택하세요", type="pdf", disabled=st.session_state.is_analyzing)
    
    if uploaded_file and not st.session_state.is_analyzing:
        if st.button("논문 분석 시작", disabled=st.session_state.is_analyzing):
            st.session_state.is_analyzing = True

    # 채팅 관리
    st.sidebar.divider()
    st.sidebar.subheader("💬 채팅 세션")
    
    paper_id = st.session_state.current_paper_id
    
    if paper_id:
        sessions = st.session_state.chat_sessions.get(paper_id, {})
        
        if st.sidebar.button("➕ 새 대화", disabled=st.session_state.is_analyzing):
            create_or_update_chat(paper_id, prompt=None, force_new=True)
            st.rerun()
            
        for session_id, session_data in sessions.items():
            label = session_data["name"]
            
            if st.sidebar.button(label, disabled=st.session_state.is_analyzing, key=session_id):
                st.session_state.current_session_id = session_id
                st.rerun()
                st.stop()

    st.sidebar.divider()
    st.sidebar.subheader("✏️ 대화 이름 변경")
    
    paper_id = st.session_state.current_paper_id
    session_id = st.session_state.current_session_id
    
    if paper_id and session_id:
        current_name = st.session_state.chat_sessions[paper_id][session_id]["name"]
        new_name = st.sidebar.text_input("세션 이름",
                                         value=current_name,
                                         disabled=st.session_state.is_analyzing)
    
        if st.sidebar.button("이름 변경", disabled=st.session_state.is_analyzing):
            st.session_state.chat_sessions[paper_id][session_id]["name"] = new_name
            st.rerun()
            st.stop()
    
            
# --- [8. 논문 분석 실행] ---
if st.session_state.is_analyzing and uploaded_file:
    with st.status("🔍 논문 분석중...", expanded=True) as status:
        query_engine, paper_id = build_rag_engine(uploaded_file)
        st.session_state.query_engine = query_engine

        if st.session_state.current_paper_id != paper_id:
            st.session_state.current_paper_id = paper_id
            st.session_state.current_session_id = None  # 현재 논문 세션 초기화
            if paper_id not in st.session_state.chat_sessions:
                st.session_state.chat_sessions[paper_id] = {}
        
        st.session_state.is_analyzing = False # 완료 시 플래그 해제
        status.update(label="✅ 분석 완료!", state="complete", expanded=False)
    st.rerun()
    
# --- [9. 채팅 기록 표시] ---
paper_id = st.session_state.current_paper_id
session_id = st.session_state.current_session_id

if paper_id and session_id:
    messages = st.session_state.chat_sessions[paper_id][session_id]["messages"]
    
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- [10. Chat Input] ---
if st.session_state.query_engine is None:
    st.info("📄 왼쪽 사이드바에서 논문을 업로드하고 분석을 시작해주세요.")
    st.stop()

prompt = st.chat_input("논문에 대해 질문해보세요", disabled=st.session_state.is_analyzing)
if prompt:
    if st.session_state.is_analyzing:
        st.warning("논문 분석이 끝난 후 질문해주세요.")
        st.stop()
    
    paper_id = st.session_state.current_paper_id
    session_id = create_or_update_chat(paper_id, prompt=prompt, force_new=False)

    messages = st.session_state.chat_sessions[paper_id][session_id]["messages"]
    messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        
    if st.session_state.query_engine:
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하고 있습니다..."):
                response = st.session_state.query_engine.query(prompt)
                st.markdown(response.response)
                messages.append({"role": "assistant", "content": response.response})
    st.rerun()