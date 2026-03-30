"""
================================================================================
 TEMPLATE: CADEIA SEQUENCIAL COM LCEL (LangChain Expression Language)
================================================================================
 O que esse código faz:
   Este template monta uma cadeia sequencial de chamadas a um LLM usando LCEL.
   O fluxo funciona assim:
     1. Recebe um INTERESSE do usuário (ex: "praias", "montanhas", "história")
     2. Usa um LLM para recomendar uma CIDADE com base nesse interesse
     3. Passa a cidade recomendada para um segundo prompt que sugere RESTAURANTES
     4. Por fim, sugere ATIVIDADES CULTURAIS na mesma cidade

   Cada etapa alimenta a próxima automaticamente, formando um pipeline com "|".
   As respostas estruturadas (cidade e restaurantes) são parseadas como JSON
   via Pydantic, e a resposta final é texto livre.

 Tecnologia principal: LangChain + LCEL (LangChain Expression Language)
 Modelo padrão: OpenAI GPT (configurável)
================================================================================
"""

# ==============================================================================
# SEÇÃO 1 — INSTALAÇÕES NECESSÁRIAS
# ==============================================================================
# Execute no terminal antes de rodar o script:
#
#   pip install langchain langchain-openai langchain-core pydantic python-dotenv
#
# ==============================================================================


# ==============================================================================
# SEÇÃO 2 — IMPORTAÇÕES
# ==============================================================================

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.globals import set_debug
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

# ==============================================================================
# SEÇÃO 3 — CONFIGURAÇÕES GERAIS
# ==============================================================================

# Ativa o modo debug da LangChain (mostra o fluxo interno das chamadas)
# Troque para False para desativar os logs detalhados
set_debug(False)

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Chave da API — certifique-se de ter um arquivo .env com: OPENAI_API_KEY=sk-...
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==============================================================================
# SEÇÃO 4 — SCHEMAS PYDANTIC (estrutura esperada das respostas JSON)
# ==============================================================================
# Cada classe define o "contrato" de saída de uma etapa da cadeia.
# O LLM será instruído a responder exatamente nesse formato.

class Destino(BaseModel):
    """Schema da resposta da etapa 1: recomendação de cidade."""
    cidade: str = Field(description="A cidade recomendada para visitar")
    motivo: str = Field(description="Motivo pelo qual é interessante visitar essa cidade")


class Restaurantes(BaseModel):
    """Schema da resposta da etapa 2: recomendação de restaurantes."""
    cidade: str = Field(description="A cidade recomendada para visitar")
    restaurantes: str = Field(description="Restaurantes recomendados na cidade")


# ==============================================================================
# SEÇÃO 5 — PARSERS DE SAÍDA
# ==============================================================================
# Os parsers convertem a resposta do LLM para o formato definido nos schemas.
# JsonOutputParser → retorna um dicionário Python
# StrOutputParser  → retorna texto puro (string)

parseador_destino      = JsonOutputParser(pydantic_object=Destino)
parseador_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)
parseador_texto        = StrOutputParser()

# ==============================================================================
# SEÇÃO 6 — PROMPTS
# ==============================================================================
# Cada PromptTemplate define as instruções enviadas ao LLM em cada etapa.
# {interesse}, {cidade} e {formato_de_saida} são variáveis preenchidas dinamicamente.
# partial_variables injeta o formato JSON esperado automaticamente no prompt.

# --- Etapa 1: recebe o interesse e recomenda uma cidade ---
PROMPT_CIDADE_TEMPLATE = """
Sugira uma cidade dado o meu interesse por {interesse}.
{formato_de_saida}
"""

prompt_cidade = PromptTemplate(
    template=PROMPT_CIDADE_TEMPLATE,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador_destino.get_format_instructions()}
)

# --- Etapa 2: recebe a cidade e recomenda restaurantes ---
PROMPT_RESTAURANTES_TEMPLATE = """
Sugira restaurantes populares entre locais em {cidade}.
{formato_de_saida}
"""

prompt_restaurantes = PromptTemplate(
    template=PROMPT_RESTAURANTES_TEMPLATE,
    input_variables=["cidade"],
    partial_variables={"formato_de_saida": parseador_restaurantes.get_format_instructions()}
)

# --- Etapa 3: recebe a cidade e sugere atividades culturais ---
PROMPT_CULTURAL_TEMPLATE = """
Sugira atividades e locais culturais interessantes em {cidade}.
"""

prompt_cultural = PromptTemplate(
    template=PROMPT_CULTURAL_TEMPLATE,
    input_variables=["cidade"]
)

# ==============================================================================
# SEÇÃO 7 — MODELO LLM
# ==============================================================================
# Configure aqui o modelo e os parâmetros de geração.
# temperature: 0.0 = mais determinístico | 1.0 = mais criativo

modelo = ChatOpenAI(
    model="gpt-3.5-turbo",   # ← Troque para "gpt-4o" ou outro modelo se quiser
    temperature=0.5,          # ← Ajuste a criatividade das respostas (0.0 a 1.0)
    api_key=OPENAI_API_KEY
)

# ==============================================================================
# SEÇÃO 8 — MONTAGEM DA CADEIA COM LCEL
# ==============================================================================
# O operador "|" conecta as etapas em sequência (pipe, como no terminal Linux).
# A saída de cada etapa vira a entrada da próxima automaticamente.
#
# Fluxo:
#   {interesse} → cadeia_1 → {cidade, motivo}
#               → cadeia_2 → {cidade, restaurantes}
#               → cadeia_3 → texto com atividades culturais

cadeia_1 = prompt_cidade      | modelo | parseador_destino       # Retorna dict: {cidade, motivo}
cadeia_2 = prompt_restaurantes | modelo | parseador_restaurantes  # Retorna dict: {cidade, restaurantes}
cadeia_3 = prompt_cultural    | modelo | parseador_texto          # Retorna string

# Cadeia completa encadeada
cadeia_completa = cadeia_1 | cadeia_2 | cadeia_3

# ==============================================================================
# SEÇÃO 9 — EXECUÇÃO
# ==============================================================================
# Altere o valor de "interesse" para testar diferentes cenários.
# Exemplos: "praias", "montanhas", "gastronomia", "história", "aventura"

INPUT_USUARIO = {
    "interesse": "praias"   # ← Mude aqui conforme sua necessidade
}

if __name__ == "__main__":
    print("=" * 60)
    print("Iniciando cadeia LangChain LCEL...")
    print("=" * 60)

    resposta = cadeia_completa.invoke(INPUT_USUARIO)

    print("\n📍 Resposta Final (atividades culturais):")
    print(resposta)