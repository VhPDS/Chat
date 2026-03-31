"""
================================================================================
 TEMPLATE: CHATBOT COM MEMÓRIA DE SESSÃO USANDO LCEL + LANGCHAIN
================================================================================
 O que esse código faz:
   Cria um chatbot com memória de conversa por sessão usando LangChain.
   O fluxo funciona assim:
     1. Um prompt define a personalidade/papel do assistente (system message)
     2. O histórico da conversa é armazenado em memória por sessão (in-memory)
     3. Uma lista de perguntas é enviada em sequência, simulando um diálogo
     4. O modelo responde levando em conta tudo que foi dito antes na sessão

   Útil para: chatbots, assistentes virtuais, atendimento automatizado,
   qualquer fluxo que precise de contexto acumulado entre mensagens.

 Tecnologia principal: LangChain + LCEL + RunnableWithMessageHistory
 Modelo padrão: OpenAI GPT (configurável)
================================================================================
"""

# ==============================================================================
# SEÇÃO 1 — INSTALAÇÕES NECESSÁRIAS
# ==============================================================================
# Execute no terminal antes de rodar o script:
#
#   pip install langchain langchain-openai langchain-core python-dotenv
#
# ==============================================================================


# ==============================================================================
# SEÇÃO 2 — IMPORTAÇÕES
# ==============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# ==============================================================================
# SEÇÃO 3 — CONFIGURAÇÕES GERAIS
# ==============================================================================

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Chave da API — certifique-se de ter um arquivo .env com: OPENAI_API_KEY=sk-...
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ID da sessão — identifica o "fio" da conversa na memória
# Troque para um valor dinâmico (ex: ID do usuário) em aplicações reais
SESSION_ID = "minha_sessao"


# ==============================================================================
# SEÇÃO 4 — MODELO LLM
# ==============================================================================
# Configure aqui o modelo e os parâmetros de geração.
# temperature: 0.0 = mais determinístico | 1.0 = mais criativo

modelo = ChatOpenAI(
    model="gpt-3.5-turbo",  # ← Troque para "gpt-4o" ou outro modelo se quiser
    temperature=0.5,         # ← Ajuste a criatividade das respostas (0.0 a 1.0)
    api_key=OPENAI_API_KEY
)


# ==============================================================================
# SEÇÃO 5 — PROMPT
# ==============================================================================
# Define a personalidade/papel do assistente e a estrutura da conversa.
#
# Slots disponíveis:
#   {historico}  → preenchido automaticamente com o histórico da sessão
#   {query}      → preenchido com a mensagem atual do usuário

SYSTEM_MESSAGE = """
Você é um assistente especializado em [TEMA].
Apresente-se como [NOME DO ASSISTENTE].
"""   # ← Defina aqui a personalidade do bot

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        ("placeholder", "{historico}"),   # ← Histórico injetado automaticamente
        ("human", "{query}")              # ← Mensagem atual do usuário
    ]
)


# ==============================================================================
# SEÇÃO 6 — CADEIA LCEL
# ==============================================================================
# O operador "|" conecta: prompt → modelo → parser de saída (texto puro)

cadeia = prompt | modelo | StrOutputParser()


# ==============================================================================
# SEÇÃO 7 — GERENCIAMENTO DE MEMÓRIA POR SESSÃO
# ==============================================================================
# A memória é um dicionário que mapeia session_id → histórico da conversa.
# InMemoryChatMessageHistory armazena tudo em RAM (dados perdidos ao reiniciar).
# Para persistência real, substitua por RedisChatMessageHistory ou similar.

memoria = {}  # Dicionário global que guarda os históricos por sessão

def historico_por_sessao(session_id: str) -> InMemoryChatMessageHistory:
    """Retorna o histórico existente ou cria um novo para a sessão informada."""
    if session_id not in memoria:
        memoria[session_id] = InMemoryChatMessageHistory()
    return memoria[session_id]


# ==============================================================================
# SEÇÃO 8 — CADEIA COM MEMÓRIA
# ==============================================================================
# RunnableWithMessageHistory "embrulha" a cadeia para injetar o histórico
# automaticamente a cada invocação, baseado no session_id fornecido.

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_por_sessao,
    input_messages_key="query",       # ← Chave da mensagem do usuário no input
    history_messages_key="historico"  # ← Chave do histórico no prompt
)


# ==============================================================================
# SEÇÃO 9 — PERGUNTAS / MENSAGENS DO USUÁRIO
# ==============================================================================
# Adicione ou remova perguntas conforme necessário.
# Cada pergunta é enviada em sequência dentro da mesma sessão,
# então o modelo terá acesso a todas as respostas anteriores.

LISTA_PERGUNTAS = [
    "Primeira pergunta aqui",   # ← Substitua pelas suas perguntas
    "Segunda pergunta aqui",
    "Terceira pergunta aqui",
]


# ==============================================================================
# SEÇÃO 10 — EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(f"Iniciando conversa | Sessão: {SESSION_ID}")
    print("=" * 60, "\n")

    for pergunta in LISTA_PERGUNTAS:
        resposta = cadeia_com_memoria.invoke(
            {"query": pergunta},
            config={"session_id": SESSION_ID}
        )
        print(f"Usuário: {pergunta}")
        print(f"IA:      {resposta}\n")