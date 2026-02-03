import tiktoken
import litellm
from litellm import completion, embedding, RateLimitError, APIError, APIConnectionError
import logging
import random
import os
from datetime import datetime
import time
import json
import PyPDF2
import copy
import asyncio
import pymupdf
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
import yaml
from pathlib import Path
from types import SimpleNamespace as config
import re

# Configurações da API - carregadas do .env
NVIDIA_NIM_API_KEY = os.getenv("NVIDIA_NIM_API_KEY")
NVIDIA_NIM_API_BASE = os.getenv("NVIDIA_NIM_API_BASE", "https://integrate.api.nvidia.com/v1/")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "nvidia_nim/meta/llama3-70b-instruct")

# Configurar LiteLLM
if NVIDIA_NIM_API_KEY:
    os.environ['NVIDIA_NIM_API_KEY'] = NVIDIA_NIM_API_KEY
if NVIDIA_NIM_API_BASE:
    os.environ['NVIDIA_NIM_API_BASE'] = NVIDIA_NIM_API_BASE

# Configuração de logging (evita sobrescrever o root logger)
logger = logging.getLogger(__name__)


def count_tokens(text, model=None):
    """Conta tokens de um texto para um modelo específico."""
    if not text:
        return 0
    try:
        # Remove o prefixo nvidia_nim/ se presente
        model_name = model.replace("nvidia_nim/", "") if model and "nvidia_nim/" in model else model
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("o200k_base")
    tokens = enc.encode(text)
    return len(tokens)


# Semáforo global para limitar concorrência de chamadas async
_GLOBAL_ASYNC_SEMAPHORE = asyncio.Semaphore(int(os.getenv("PI_MAX_CONCURRENCY", "3")))


def _jitter_sleep(base_seconds):
    # jitter entre 0.8x e 1.2x para evitar bursts sincronizados
    jitter = random.uniform(0.8, 1.2)
    time.sleep(base_seconds * jitter)


def _jitter_async_sleep(base_seconds):
    jitter = random.uniform(0.8, 1.2)
    return asyncio.sleep(base_seconds * jitter)


def LiteLLM_API_with_finish_reason(model, prompt, api_key=None, chat_history=None):
    """
    Chamada síncrona ao LiteLLM com retorno do motivo de finalização.
    Inclui backoff exponencial para rate limits.
    """
    max_retries = 10
    
    # Se não especificar modelo, usa o padrão do .env
    if not model:
        model = DEFAULT_MODEL
    
    # Garantir que o modelo tenha o prefixo nvidia_nim/
    if not model.startswith("nvidia_nim/"):
        model = f"nvidia_nim/{model}"
    
    for i in range(max_retries):
        try:
            if chat_history:
                messages = chat_history.copy()
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [{"role": "user", "content": prompt}]
            
            response = completion(
                model=model,
                messages=messages,
                temperature=0,
                api_key=api_key or NVIDIA_NIM_API_KEY,
            )
            
            # LiteLLM retorna objeto similar ao OpenAI
            finish_reason = response.choices[0].finish_reason
            content = response.choices[0].message.content
            
            if finish_reason == "length":
                return content, "max_output_reached"
            else:
                return content, "finished"

        except RateLimitError as e:
            wait_time = min(2 ** i * 2, 60)
            logger.warning(f"Rate limit atingido (tentativa {i+1}/{max_retries}). Aguardando {wait_time}s...")
            _jitter_sleep(wait_time)
            
        except (APIError, APIConnectionError) as e:
            wait_time = min(2 ** i, 30)
            logger.warning(f"Erro de API (tentativa {i+1}/{max_retries}): {e}. Aguardando {wait_time}s...")
            _jitter_sleep(wait_time)
            
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            if i < max_retries - 1:
                _jitter_sleep(2 ** i)
            else:
                logger.error(f'Max retries atingido. Prompt: {prompt[:100]}...')
                return "Error", "error"
    
    return "Error", "error"


async def LiteLLM_API_with_finish_reason_async(model, prompt, api_key=None, chat_history=None):
    """
    Chamada assíncrona ao LiteLLM com retorno do motivo de finalização.
    Inclui backoff exponencial para rate limits.
    """
    max_retries = 10

    if not model:
        model = DEFAULT_MODEL

    if not model.startswith("nvidia_nim/"):
        model = f"nvidia_nim/{model}"

    for i in range(max_retries):
        try:
            if chat_history:
                messages = chat_history.copy()
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [{"role": "user", "content": prompt}]

            async with _GLOBAL_ASYNC_SEMAPHORE:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=0,
                    api_key=api_key or NVIDIA_NIM_API_KEY,
                )

            finish_reason = response.choices[0].finish_reason
            content = response.choices[0].message.content

            if finish_reason == "length":
                return content, "max_output_reached"
            return content, "finished"

        except RateLimitError as e:
            wait_time = min(2 ** i * 2, 60)
            logger.warning(f"Rate limit atingido (tentativa {i+1}/{max_retries}). Aguardando {wait_time}s...")
            await _jitter_async_sleep(wait_time)

        except (APIError, APIConnectionError) as e:
            wait_time = min(2 ** i, 30)
            logger.warning(f"Erro de API (tentativa {i+1}/{max_retries}): {e}. Aguardando {wait_time}s...")
            await _jitter_async_sleep(wait_time)

        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            if i < max_retries - 1:
                await _jitter_async_sleep(2 ** i)
            else:
                logger.error(f'Max retries atingido. Prompt: {prompt[:100]}...')
                return "Error", "error"

    return "Error", "error"

def LiteLLM_API(model, prompt, api_key=None, chat_history=None):
    """
    Chamada síncrona ao LiteLLM.
    Inclui backoff exponencial para rate limits.
    """
    max_retries = 10
    
    # Se não especificar modelo, usa o padrão do .env
    if not model:
        model = DEFAULT_MODEL
    
    # Garantir que o modelo tenha o prefixo nvidia_nim/
    if not model.startswith("nvidia_nim/"):
        model = f"nvidia_nim/{model}"
    
    for i in range(max_retries):
        try:
            if chat_history:
                messages = chat_history.copy()
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [{"role": "user", "content": prompt}]
            
            response = completion(
                model=model,
                messages=messages,
                temperature=0,
                api_key=api_key or NVIDIA_NIM_API_KEY,
            )
            
            return response.choices[0].message.content
            
        except RateLimitError as e:
            wait_time = min(2 ** i * 2, 60)
            logger.warning(f"Rate limit atingido (tentativa {i+1}/{max_retries}). Aguardando {wait_time}s...")
            _jitter_sleep(wait_time)
            
        except (APIError, APIConnectionError) as e:
            wait_time = min(2 ** i, 30)
            logger.warning(f"Erro de API (tentativa {i+1}/{max_retries}): {e}. Aguardando {wait_time}s...")
            _jitter_sleep(wait_time)
            
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            if i < max_retries - 1:
                _jitter_sleep(2 ** i)
            else:
                logger.error(f'Max retries atingido. Prompt: {prompt[:100]}...')
                return "Error"
    
    return "Error"


async def LiteLLM_API_async(model, prompt, api_key=None):
    """
    Chamada assíncrona ao LiteLLM.
    Inclui backoff exponencial para rate limits.
    """
    max_retries = 10
    
    # Se não especificar modelo, usa o padrão do .env
    if not model:
        model = DEFAULT_MODEL
    
    # Garantir que o modelo tenha o prefixo nvidia_nim/
    if not model.startswith("nvidia_nim/"):
        model = f"nvidia_nim/{model}"
    
    messages = [{"role": "user", "content": prompt}]
    
    for i in range(max_retries):
        try:
            async with _GLOBAL_ASYNC_SEMAPHORE:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=0,
                    api_key=api_key or NVIDIA_NIM_API_KEY,
                )
            return response.choices[0].message.content
                
        except RateLimitError as e:
            wait_time = min(2 ** i * 2, 60)
            logger.warning(f"Rate limit atingido (tentativa {i+1}/{max_retries}). Aguardando {wait_time}s...")
            await _jitter_async_sleep(wait_time)
            
        except (APIError, APIConnectionError) as e:
            wait_time = min(2 ** i, 30)
            logger.warning(f"Erro de API (tentativa {i+1}/{max_retries}): {e}. Aguardando {wait_time}s...")
            await _jitter_async_sleep(wait_time)
            
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            if i < max_retries - 1:
                await _jitter_async_sleep(2 ** i)
            else:
                logger.error(f'Max retries atingido. Prompt: {prompt[:100]}...')
                return "Error"
    
    return "Error"


__all__ = [
    "LiteLLM_API_with_finish_reason",
    "LiteLLM_API_with_finish_reason_async",
    "LiteLLM_API",
    "LiteLLM_API_async",
]


def get_json_content(response):
    """Extrai conteúdo JSON de uma resposta com markdown."""
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]
        
    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]
    
    json_content = response.strip()
    return json_content


def extract_json(content):
    """Extrai e parseia JSON de uma string, com tratamento de erros robusto."""
    try:
        # Tenta extrair JSON entre ```json e ```
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            json_content = content.strip()

        # Limpeza mínima de problemas comuns (sem destruir quebras dentro de strings)
        json_content = json_content.replace('None', 'null')

        return json.loads(json_content)

    except json.JSONDecodeError as e:
        logger.error(f"Falha ao extrair JSON: {e}")
        try:
            # Tentativa leve de correção de vírgulas soltas
            json_content = json_content.replace(',]', ']').replace(',}', '}')
            return json.loads(json_content)
        except Exception:
            logger.error("Falha ao parsear JSON após limpeza")
            return {}

    except Exception as e:
        logger.error(f"Erro inesperado ao extrair JSON: {e}")
        return {}


def write_node_id(data, node_id=0):
    """Adiciona IDs únicos aos nós de forma recursiva."""
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if 'nodes' in key:
                node_id = write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for index in range(len(data)):
            node_id = write_node_id(data[index], node_id)
    return node_id


def get_nodes(structure):
    """Extrai todos os nós de uma estrutura hierárquica."""
    if isinstance(structure, dict):
        structure_node = copy.deepcopy(structure)
        structure_node.pop('nodes', None)
        nodes = [structure_node]
        for key in list(structure.keys()):
            if 'nodes' in key:
                nodes.extend(get_nodes(structure[key]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(get_nodes(item))
        return nodes


def structure_to_list(structure):
    """Converte estrutura hierárquica em lista plana."""
    if isinstance(structure, dict):
        nodes = [structure]
        if 'nodes' in structure:
            nodes.extend(structure_to_list(structure['nodes']))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes


def get_leaf_nodes(structure):
    """Retorna apenas os nós folha (sem filhos)."""
    if isinstance(structure, dict):
        if not structure.get('nodes'):
            structure_node = copy.deepcopy(structure)
            structure_node.pop('nodes', None)
            return [structure_node]
        else:
            leaf_nodes = []
            for key in list(structure.keys()):
                if 'nodes' in key:
                    leaf_nodes.extend(get_leaf_nodes(structure[key]))
            return leaf_nodes
    elif isinstance(structure, list):
        leaf_nodes = []
        for item in structure:
            leaf_nodes.extend(get_leaf_nodes(item))
        return leaf_nodes


def is_leaf_node(data, node_id):
    """Verifica se um nó específico é folha."""
    def find_node(data, node_id):
        if isinstance(data, dict):
            if data.get('node_id') == node_id:
                return data
            for key in data.keys():
                if 'nodes' in key:
                    result = find_node(data[key], node_id)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = find_node(item, node_id)
                if result:
                    return result
        return None

    node = find_node(data, node_id)
    if node and not node.get('nodes'):
        return True
    return False


def get_last_node(structure):
    """Retorna o último nó de uma lista."""
    return structure[-1]


def extract_text_from_pdf(pdf_path):
    """Extrai todo o texto de um PDF."""
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


def get_pdf_title(pdf_path):
    """Obtém o título do PDF dos metadados."""
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    meta = pdf_reader.metadata
    title = meta.title if meta and meta.title else 'Untitled'
    return title


def get_text_of_pages(pdf_path, start_page, end_page, tag=True):
    """Extrai texto de páginas específicas do PDF."""
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(start_page-1, end_page):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        if tag:
            text += f"<start_index_{page_num+1}>\n{page_text}\n<end_index_{page_num+1}>\n"
        else:
            text += page_text
    return text


def get_first_start_page_from_text(text):
    """Extrai o primeiro índice de página de um texto com tags."""
    start_page = -1
    start_page_match = re.search(r'<start_index_(\d+)>', text)
    if start_page_match:
        start_page = int(start_page_match.group(1))
    return start_page


def get_last_start_page_from_text(text):
    """Extrai o último índice de página de um texto com tags."""
    start_page = -1
    start_page_matches = re.finditer(r'<start_index_(\d+)>', text)
    matches_list = list(start_page_matches)
    if matches_list:
        start_page = int(matches_list[-1].group(1))
    return start_page


def sanitize_filename(filename, replacement='-'):
    """Remove caracteres inválidos de nomes de arquivo."""
    return filename.replace('/', replacement).replace('\0', replacement)


def get_pdf_name(pdf_path):
    """Obtém o nome do PDF a partir do caminho ou metadados."""
    if isinstance(pdf_path, str):
        pdf_name = os.path.basename(pdf_path)
    elif isinstance(pdf_path, BytesIO):
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        meta = pdf_reader.metadata
        pdf_name = meta.title if meta and meta.title else 'Untitled'
        pdf_name = sanitize_filename(pdf_name)
    return pdf_name


class JsonLogger:
    """Logger que salva mensagens em formato JSON."""
    
    def __init__(self, file_path):
        pdf_name = get_pdf_name(file_path)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{pdf_name}_{current_time}.json"
        os.makedirs("./logs", exist_ok=True)
        self.log_data = []

    def log(self, level, message, **kwargs):
        if isinstance(message, dict):
            self.log_data.append(message)
        else:
            self.log_data.append({'message': message})
        
        with open(self._filepath(), "w", encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False)

    def info(self, message, **kwargs):
        self.log("INFO", message, **kwargs)

    def error(self, message, **kwargs):
        self.log("ERROR", message, **kwargs)

    def debug(self, message, **kwargs):
        self.log("DEBUG", message, **kwargs)

    def exception(self, message, **kwargs):
        kwargs["exception"] = True
        self.log("ERROR", message, **kwargs)

    def _filepath(self):
        return os.path.join("logs", self.filename)


def list_to_tree(data):
    """Converte lista plana em estrutura hierárquica."""
    def get_parent_structure(structure):
        if not structure:
            return None
        parts = str(structure).split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else None
    
    nodes = {}
    root_nodes = []
    
    for item in data:
        structure = item.get('structure')
        node = {
            'title': item.get('title'),
            'start_index': item.get('start_index'),
            'end_index': item.get('end_index'),
            'nodes': []
        }
        
        nodes[structure] = node
        parent_structure = get_parent_structure(structure)
        
        if parent_structure:
            if parent_structure in nodes:
                nodes[parent_structure]['nodes'].append(node)
            else:
                root_nodes.append(node)
        else:
            root_nodes.append(node)
    
    def clean_node(node):
        if not node['nodes']:
            del node['nodes']
        else:
            for child in node['nodes']:
                clean_node(child)
        return node
    
    return [clean_node(node) for node in root_nodes]


def add_preface_if_needed(data):
    """Adiciona prefácio se o primeiro item não começar na página 1."""
    if not isinstance(data, list) or not data:
        return data

    if data[0].get('physical_index') is not None and data[0]['physical_index'] > 1:
        preface_node = {
            "structure": "0",
            "title": "Preface",
            "physical_index": 1,
        }
        data.insert(0, preface_node)
    return data


def get_page_tokens(pdf_path, model=None, pdf_parser="PyPDF2"):
    """Calcula tokens para cada página do PDF."""
    # Remove o prefixo nvidia_nim/ se presente para tiktoken
    model_for_tiktoken = model.replace("nvidia_nim/", "") if model and "nvidia_nim/" in model else (model or "gpt-4o")
    
    try:
        enc = tiktoken.encoding_for_model(model_for_tiktoken)
    except:
        enc = tiktoken.get_encoding("o200k_base")
    
    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            token_length = len(enc.encode(page_text))
            page_list.append((page_text, token_length))
        return page_list
        
    elif pdf_parser == "PyMuPDF":
        if isinstance(pdf_path, BytesIO):
            doc = pymupdf.open(stream=pdf_path, filetype="pdf")
        elif isinstance(pdf_path, str) and os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
            doc = pymupdf.open(pdf_path)
        else:
            raise ValueError(f"Invalid PDF path: {pdf_path}")
            
        page_list = []
        for page in doc:
            page_text = page.get_text()
            token_length = len(enc.encode(page_text))
            page_list.append((page_text, token_length))
        return page_list
    else:
        raise ValueError(f"Unsupported PDF parser: {pdf_parser}")


def get_text_of_pdf_pages(pdf_pages, start_page, end_page):
    """Obtém texto de páginas específicas a partir da lista de páginas."""
    text = ""
    for page_num in range(start_page-1, end_page):
        if page_num < len(pdf_pages):
            text += pdf_pages[page_num][0]
    return text


def get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page):
    """Obtém texto de páginas com labels de índice."""
    text = ""
    for page_num in range(start_page-1, end_page):
        if page_num < len(pdf_pages):
            text += f"<physical_index_{page_num+1}>\n{pdf_pages[page_num][0]}\n<physical_index_{page_num+1}>\n"
    return text


def get_number_of_pages(pdf_path):
    """Retorna o número total de páginas do PDF."""
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    return len(pdf_reader.pages)


def post_processing(structure, end_physical_index):
    """Pós-processamento da estrutura com índices de páginas."""
    for i, item in enumerate(structure):
        item['start_index'] = item.get('physical_index')
        if i < len(structure) - 1:
            if structure[i + 1].get('appear_start') == 'yes':
                item['end_index'] = structure[i + 1]['physical_index'] - 1
            else:
                item['end_index'] = structure[i + 1]['physical_index']
        else:
            item['end_index'] = end_physical_index
            
    tree = list_to_tree(structure)
    if len(tree) != 0:
        return tree
    else:
        for node in structure:
            node.pop('appear_start', None)
            node.pop('physical_index', None)
        return structure


def clean_structure_post(data):
    """Remove campos desnecessários da estrutura."""
    if isinstance(data, dict):
        data.pop('page_number', None)
        data.pop('start_index', None)
        data.pop('end_index', None)
        if 'nodes' in data:
            clean_structure_post(data['nodes'])
    elif isinstance(data, list):
        for section in data:
            clean_structure_post(section)
    return data


def remove_fields(data, fields=['text']):
    """Remove campos específicos da estrutura recursivamente."""
    if isinstance(data, dict):
        return {k: remove_fields(v, fields)
                for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data


def print_toc(tree, indent=0):
    """Imprime índice em formato hierárquico."""
    for node in tree:
        print('  ' * indent + node['title'])
        if node.get('nodes'):
            print_toc(node['nodes'], indent + 1)


def print_json(data, max_len=40, indent=2):
    """Imprime JSON com strings longas truncadas."""
    def simplify_data(obj):
        if isinstance(obj, dict):
            return {k: simplify_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [simplify_data(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + '...'
        else:
            return obj
    
    simplified = simplify_data(data)
    print(json.dumps(simplified, indent=indent, ensure_ascii=False))


def remove_structure_text(data):
    """Remove campo 'text' de todos os nós."""
    if isinstance(data, dict):
        data.pop('text', None)
        if 'nodes' in data:
            remove_structure_text(data['nodes'])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


def check_token_limit(structure, limit=110000, model=None):
    """Verifica se algum nó excede o limite de tokens."""
    nodes_list = structure_to_list(structure)
    model_for_count = model or DEFAULT_MODEL
    
    for node in nodes_list:
        if 'text' in node:
            num_tokens = count_tokens(node['text'], model=model_for_count)
            if num_tokens > limit:
                logger.warning(f"Node ID: {node.get('node_id')} tem {num_tokens} tokens")
                logger.warning(f"Start Index: {node.get('start_index')}")
                logger.warning(f"End Index: {node.get('end_index')}")
                logger.warning(f"Title: {node.get('title')}\n")


def convert_physical_index_to_int(data):
    """Converte índices físicos de string para int."""
    if isinstance(data, list):
        for i in range(len(data)):
            if isinstance(data[i], dict) and 'physical_index' in data[i]:
                if isinstance(data[i]['physical_index'], str):
                    if data[i]['physical_index'].startswith('<physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].rstrip('>').strip())
                    elif data[i]['physical_index'].startswith('physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].strip())
                        
    elif isinstance(data, str):
        if data.startswith('<physical_index_'):
            data = int(data.split('_')[-1].rstrip('>').strip())
        elif data.startswith('physical_index_'):
            data = int(data.split('_')[-1].strip())
        if isinstance(data, int):
            return data
        else:
            return None
    return data


def convert_page_to_int(data):
    """Converte campo 'page' de string para int."""
    for item in data:
        if 'page' in item and isinstance(item['page'], str):
            try:
                item['page'] = int(item['page'])
            except ValueError:
                pass
    return data


def add_node_text(node, pdf_pages):
    """Adiciona texto aos nós baseado em índices de página."""
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        if start_page and end_page:
            node['text'] = get_text_of_pdf_pages(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for item in node:
            add_node_text(item, pdf_pages)


def add_node_text_with_labels(node, pdf_pages):
    """Adiciona texto aos nós com labels de índice."""
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        if start_page and end_page:
            node['text'] = get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text_with_labels(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for item in node:
            add_node_text_with_labels(item, pdf_pages)


async def generate_node_summary(node, model=None):
    """Gera sumário para um nó específico."""
    if 'text' not in node or not node['text']:
        return "No content available"
    
    model = model or DEFAULT_MODEL
        
    prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

Partial Document Text: {node['text']}

Directly return the description, do not include any other text.
"""
    response = await LiteLLM_API_async(model, prompt)
    return response


async def generate_summaries_for_structure(structure, model=None, batch_size=5, delay_between_batches=2):
    """
    Gera sumários para toda a estrutura com controle de rate limit.
    Processa em lotes pequenos com delays entre eles.
    """
    model = model or DEFAULT_MODEL
    nodes = structure_to_list(structure)
    logger.info(f"Gerando sumários para {len(nodes)} nós usando modelo {model}...")
    
    # Semáforo para limitar concorrência
    semaphore = asyncio.Semaphore(3)
    
    async def safe_generate(node, node_idx):
        """Wrapper com retry e semáforo."""
        async with semaphore:
            for attempt in range(5):
                try:
                    summary = await generate_node_summary(node, model=model)
                    logger.info(f"Sumário gerado para nó {node_idx + 1}/{len(nodes)}")
                    return summary
                except RateLimitError:
                    wait_time = min(2 ** attempt * 3, 60)
                    logger.warning(f"Rate limit no nó {node_idx + 1}. Aguardando {wait_time}s...")
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    logger.error(f"Erro ao gerar sumário para nó {node_idx + 1}: {e}")
                    if attempt < 4:
                        await asyncio.sleep(2 ** attempt)
            return "Error generating summary"
    
    summaries = []
    
    # Processar em lotes
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i + batch_size]
        batch_start_idx = i
        
        logger.info(f"Processando lote {i//batch_size + 1}/{(len(nodes)-1)//batch_size + 1}")
        
        tasks = [safe_generate(node, batch_start_idx + idx) for idx, node in enumerate(batch)]
        batch_summaries = await asyncio.gather(*tasks)
        summaries.extend(batch_summaries)
        
        # Delay entre lotes (exceto no último)
        if i + batch_size < len(nodes):
            logger.info(f"Aguardando {delay_between_batches}s antes do próximo lote...")
            await asyncio.sleep(delay_between_batches)
    
    # Atribuir sumários aos nós
    for node, summary in zip(nodes, summaries):
        node['summary'] = summary
    
    logger.info("Geração de sumários concluída!")
    return structure


def create_clean_structure_for_description(structure):
    """Cria estrutura limpa excluindo campos desnecessários."""
    if isinstance(structure, dict):
        clean_node = {}
        for key in ['title', 'node_id', 'summary', 'prefix_summary']:
            if key in structure:
                clean_node[key] = structure[key]
        
        if 'nodes' in structure and structure['nodes']:
            clean_node['nodes'] = create_clean_structure_for_description(structure['nodes'])
        
        return clean_node
    elif isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure]
    else:
        return structure


def generate_doc_description(structure, model=None):
    """Gera descrição de uma linha para o documento."""
    model = model or DEFAULT_MODEL
    
    prompt = f"""Your are an expert in generating descriptions for a document.
You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.
    
Document Structure: {structure}

Directly return the description, do not include any other text.
"""
    response = LiteLLM_API(model, prompt)
    return response


def reorder_dict(data, key_order):
    """Reordena dicionário baseado em lista de chaves."""
    if not key_order:
        return data
    return {key: data[key] for key in key_order if key in data}


def format_structure(structure, order=None):
    """Formata estrutura reordenando chaves e removendo nós vazios."""
    if not order:
        return structure
    if isinstance(structure, dict):
        if 'nodes' in structure:
            structure['nodes'] = format_structure(structure['nodes'], order)
        if not structure.get('nodes'):
            structure.pop('nodes', None)
        structure = reorder_dict(structure, order)
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure]
    return structure


class ConfigLoader:
    """Carregador de configurações YAML com validação."""
    
    def __init__(self, default_path: str = None):
        if default_path is None:
            default_path = Path(__file__).parent / "config.yaml"
        self._default_dict = self._load_yaml(default_path)

    @staticmethod
    def _load_yaml(path):
        """Carrega arquivo YAML."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _validate_keys(self, user_dict):
        """Valida se todas as chaves são conhecidas."""
        unknown_keys = set(user_dict) - set(self._default_dict)
        if unknown_keys:
            raise ValueError(f"Chaves de configuração desconhecidas: {unknown_keys}")

    def load(self, user_opt=None) -> config:
        """Carrega configuração mesclando opções do usuário com padrões."""
        if user_opt is None:
            user_dict = {}
        elif isinstance(user_opt, config):
            user_dict = vars(user_opt)
        elif isinstance(user_opt, dict):
            user_dict = user_opt
        else:
            raise TypeError("user_opt deve ser dict, config(SimpleNamespace) ou None")

        self._validate_keys(user_dict)
        merged = {**self._default_dict, **user_dict}
        return config(**merged)
