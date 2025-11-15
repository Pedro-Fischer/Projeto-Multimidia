from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import base64
import os
import threading
import queue
import requests
import concurrent.futures
from io import BytesIO
from PIL import Image
import speech_recognition as sr

from openai import OpenAI
from faster_whisper import WhisperModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import requests
import concurrent.futures
import base64
import os

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'gior-secret-key-2025'
socketio = SocketIO(app, cors_allowed_origins="*")

class GiorWeb:
    def __init__(self):
        self.ai_name = 'Consultor de Moda GIOR'
        self.historico_conversa = ''
        self.contexto = """
Você é o 'Consultor de Moda GIOR', um crítico de moda experiente, direto, honesto e com um olhar apurado para estilo. Sua personalidade é **direta e impiedosa**, mas o feedback é sempre construtivo.
Sua função principal é analisar a imagem capturada (que mostra a roupa do utilizador) e fornecer uma crítica de moda verbal, relevante e **brutalmente honesta**, baseada na imagem e na pergunta feita.

Instruções e Regras Essenciais:

1.  **Análise Visual:**
    * Identifique as principais peças, a paleta de cores, as texturas e o caimento da roupa.
    * Avalie a **adequação e a funcionalidade** do look para o contexto do ambiente (se for possível inferir) ou para o contexto mencionado pelo usuário (ex: 'festa formal').
2.  **A Regra da Crítica Honesta:**
    * **Se o look estiver inadequado, com cores dissonantes, caimento ruim ou completamente estranho, você DEVE dizer isso de forma clara, sem floreios.** A honestidade é sua principal ferramenta.
    * Mesmo a crítica negativa deve ser seguida por uma sugestão de como consertar o erro.
3.  **Formato da Crítica (Foco em 3 Partes):**
    * A sua resposta deve ser estruturada de forma concisa em três pontos, usando linguagem de moda (ex: 'caimento', 'coordenação', 'ponto focal', 'silhueta'):
        * **1. O Veredito:** Uma declaração direta e honesta sobre a peça/combinação. Use frases de impacto (ex: "Não funcionou", "Erro grave de proporção").
        * **2. O Conserto:** A sugestão de como consertar o erro primário (ex: "Substitua a peça X pela peça Y", "Ajuste o comprimento").
        * **3. Dica de Styling:** Uma sugestão de peça, acessório ou combinação para elevar o look restante.
4.  **Linguagem e Tom:**
    * Fale sempre em **Português Brasileiro**, com um tom direto, confiante e profissional.
    * O seu nome como consultor é **GIOR**. Comece a resposta com 'O Consultor GIOR tem um veredito: '
5.  **Resposta a Comandos Específicos:**
    * **Se a pergunta for 'Combina com [evento/ambiente]?'**: Diga diretamente se a roupa é adequada ou um erro total para o local.

Exemplo de uma Crítica DURA (se a roupa for horrível e o usuário perguntar 'Posso ir a uma reunião de negócios assim?'):
'O Consultor GIOR tem um veredito: Não. Este look é totalmente inadequado para uma reunião de negócios. **Veredito:** O jeans rasgado e a camiseta desbotada demonstram falta de seriedade. **O Conserto:** Use uma calça chino escura e um blazer simples imediatamente. **Dica de Styling:** Um lenço de bolso elegante traria um toque de autoridade.'
        """
        
        self.sistema_ativo = False
        self.gravando_audio = False
        self.pergunta = ''
        self.imagem_capturada = None
        self.camera = None
        self.current_frame = None
        self.audio_queue = queue.Queue()
        
        # OpenAI e modelos
        self.client = OpenAI()
        self.llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        
        # Whisper
        print("Carregando modelo Whisper...")
        self.whisper_model = WhisperModel("medium")
        print("Modelo Whisper carregado!")
        
        # Speech Recognition
        self.recognizer = sr.Recognizer()
        
        # Diretórios
        if not os.path.exists('frames'):
            os.makedirs('frames')
        if not os.path.exists('static'):
            os.makedirs('static')

    def ativar_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            self.sistema_ativo = True
            return True
        return False

    def desativar_camera(self):
        if self.camera:
            self.camera.release()
            self.camera = None
            self.sistema_ativo = False
            return True
        return False

    def get_frame(self):
        if self.camera and self.sistema_ativo:
            success, frame = self.camera.read()
            if success:
                self.current_frame = frame.copy()
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                return frame_bytes
        return None

    def capturar_imagem(self):
        if self.current_frame is not None:
            cv2.imwrite('frames/captured_frame.jpg', self.current_frame)
            self.imagem_capturada = 'frames/captured_frame.jpg'
            
            # Salvar também na pasta static para exibição
            cv2.imwrite('static/captured.jpg', self.current_frame)
            return True
        return False

    def encode_image(self):
        try:
            image_path = self.imagem_capturada if self.imagem_capturada else 'frames/last_frame.jpg'
            
            if not os.path.exists(image_path):
                return None
            
            pil_image = Image.open(image_path)
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # RETORNA A STRING BASE64 PURA, sem prefixo URL
            return img_str 
            
        except Exception as e:
            print(f"Erro ao codificar imagem: {e}")
            return None

    # --- NOVOS MÉTODOS: Google Gemini e Hugging Face ---
    def obter_analise_gemini(self, encoded_image, pergunta):
        """API 2: Google Gemini - Análise de Visão contextualizada."""
        key = os.getenv('GEMINI_API_KEY')
        model = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        if not key:
            return "Erro: GEMINI_API_KEY não configurada no .env."

        # Normaliza nome do modelo caso usuário informe sem o prefixo "models/"
        if not model.startswith('models/'):
            model = f'models/{model}'

        url = f'https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={key}'

        # Prompt que sempre gera resposta, mesmo sem contexto visual
        prompt_text = (
            f"Você é um consultor de moda especializado. "
            f"Pergunta do usuário: '{pergunta}'. "
            f"Dê uma resposta prática e objetiva sobre moda e estilo em no máximo 40 palavras. "
            f"Se a pergunta for genérica,ou se não houver pergunta, faça comentários gerais sobre a roupa."
        )

        payload = {
            "contents": [{
                "parts": [{"text": prompt_text}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 5000,
                "candidateCount": 1
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }

        headers = {'Content-Type': 'application/json'}

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=25)
            try:
                j = r.json()
            except Exception:
                j = None

            if r.status_code != 200:
                body = (r.text or '')[:800]
                return f"Erro no Gemini: status={r.status_code}, body={body}"

            # DEBUG: Vamos ver o formato real da resposta
            print(f"DEBUG Gemini Response: {j}")

            # Parse da resposta do Gemini (várias tentativas)
            if isinstance(j, dict):
                # Formato padrão: candidates[].content.parts[].text
                if 'candidates' in j and j['candidates']:
                    candidates = j['candidates']
                    if len(candidates) > 0:
                        candidate = candidates[0]
                        
                        # Verifica se há parts com texto
                        content = candidate.get('content', {})
                        if isinstance(content, dict) and 'parts' in content:
                            parts = content['parts']
                            if parts and len(parts) > 0:
                                for part in parts:
                                    if isinstance(part, dict) and 'text' in part:
                                        text = part['text']
                                        if text and text.strip():
                                            return text.strip()
                        
                        # Se não tem parts mas tem finishReason, significa que foi bloqueado ou vazio
                        finish_reason = candidate.get('finishReason', '')
                        if finish_reason == 'MAX_TOKENS':
                            return "O Gemini iniciou uma resposta mas atingiu o limite. Tente uma pergunta mais curta."
                        elif finish_reason == 'SAFETY':
                            return "Resposta bloqueada por filtros de segurança do Gemini."
                        elif finish_reason:
                            return f"Gemini finalizou sem texto (motivo: {finish_reason})."
                
                # Formato alternativo: text direto no root
                if 'text' in j:
                    return j['text'].strip()

            return f'Erro na resposta do Gemini: sem texto gerado. JSON: {str(j)[:400]}'

        except Exception as e:
            return f"Erro inesperado no Gemini: {str(e)}"

    def obter_analise_hf(self, pergunta):
        """API 3: Hugging Face - Feedback de Texto usando o novo router."""
        key = os.getenv('HUGGINGFACE_API_KEY')
        # Modelos disponíveis no router (testados e funcionais)
        model = os.getenv('HUGGINGFACE_MODEL', 'meta-llama/Llama-3.2-3B-Instruct')

        if not key:
            return "Erro: HUGGINGFACE_API_KEY não configurada no .env."

        # NOVO ENDPOINT ROUTER (obrigatório desde 2024)
        url = f'https://router.huggingface.co/hf-inference/{model}'
        headers = {
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json'
        }

        # Prompt otimizado para modelos instruct
        hf_prompt = f"Give a brief fashion tip about: {pergunta}"

        payload = {
            "inputs": hf_prompt,
            "parameters": {
                "max_new_tokens": 60,
                "temperature": 0.7,
                "top_p": 0.9,
                "return_full_text": False
            }
        }

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            try:
                j = r.json()
            except Exception:
                j = None

            # DEBUG
            print(f"DEBUG HF Status: {r.status_code}")
            print(f"DEBUG HF Response: {j}")

            if r.status_code != 200:
                body = (r.text or '')[:500]
                # Se modelo não disponível, tenta fallback
                if r.status_code in [404, 503, 410]:
                    return self._tentar_hf_alternativo(pergunta, key)
                return f"Erro no Hugging Face: status={r.status_code}"

            # Parse da resposta
            if isinstance(j, list) and len(j) > 0:
                first = j[0]
                if isinstance(first, dict) and 'generated_text' in first:
                    text = first['generated_text'].strip()
                    return text[:200] if text else 'Dica de moda processando...'
            
            if isinstance(j, dict):
                if 'generated_text' in j:
                    return j['generated_text'].strip()[:200]
                if 'error' in j:
                    return self._tentar_hf_alternativo(pergunta, key)

            return 'Modelo carregando, tente novamente em alguns segundos.'

        except Exception as e:
            return f"Erro inesperado no HF: {str(e)}"

    def _tentar_hf_alternativo(self, pergunta, key):
        """Fallback para modelo compacto e rápido."""
        # Tenta modelos menores que costumam estar sempre disponíveis
        modelos_backup = [
            'openai-community/gpt2',
            'facebook/opt-125m',
            'EleutherAI/gpt-neo-125m'
        ]
        
        for model in modelos_backup:
            try:
                url = f'https://router.huggingface.co/hf-inference/{model}'
                headers = {
                    'Authorization': f'Bearer {key}',
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    "inputs": f"Fashion tip: {pergunta[:50]}",
                    "parameters": {"max_new_tokens": 40}
                }
                
                r = requests.post(url, headers=headers, json=payload, timeout=30)
                
                if r.status_code == 200:
                    j = r.json()
                    if isinstance(j, list) and j and 'generated_text' in j[0]:
                        return j[0]['generated_text'][:150]
                    if isinstance(j, dict) and 'generated_text' in j:
                        return j['generated_text'][:150]
            except:
                continue
        
        # Se todos falharem, retorna dica genérica
        return "💡 Combine cores complementares e escolha peças que valorizem sua silhueta!"
        
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=40)
            j = r.json()
            
            if isinstance(j, list) and len(j) > 0:
                return j[0].get('generated_text', 'Dica: combine cores neutras!')[:100]
            if isinstance(j, dict) and 'generated_text' in j:
                return j['generated_text'][:100]
                
        except Exception:
            pass
        
        return "💡 Dica genérica: Combine cores complementares e invista em peças versáteis!"

    def formatar_pergunta(self, pergunta):
        try:
            encoded_image = self.encode_image()
            
            if encoded_image is None:
                inputs = [
                    [HumanMessage(
                        content=f'{self.contexto}\n\nConversa atual:\n{self.historico_conversa}\n\nAluno: {pergunta}\n'
                    )],
                ]
                return inputs

            inputs = [
                [HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": f'{self.contexto}\n\nConversa atual:\n{self.historico_conversa}\n\nAluno: {pergunta}\nimagem: \n'
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                )],
            ]
            return inputs
        except Exception as e:
            print(f"Erro ao formatar pergunta: {e}")
            return None

    def obter_resposta(self, pergunta):
        try:
            input_data = self.formatar_pergunta(pergunta)
            
            if input_data is None:
                return "Erro ao processar a pergunta."
            
            answer = self.llm.stream(input_data[0])
            self.historico_conversa += f"Aluno: {pergunta}\n {self.ai_name}:\n"
            
            resposta_completa = ""
            for resp in answer:
                resposta_completa += resp.content
            
            self.historico_conversa += f"{resposta_completa}\n"
            return resposta_completa
            
        except Exception as e:
            print(f"Erro durante a obtenção de resposta: {e}")
            return f"Erro: {str(e)}"

    def gerar_audio(self, texto):
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="onyx",
                input=texto,
                speed=1.3
            )
            
            audio_path = 'static/resposta.mp3'
            response.stream_to_file(audio_path)
            return audio_path
        except Exception as e:
            print(f"Erro ao gerar áudio: {e}")
            return None

    def transcribe_audio(self, audio_file_path):
        try:
            segments, _ = self.whisper_model.transcribe(audio=audio_file_path, language='pt', beam_size=5)
            transcricao = ""
            for segment in segments:
                transcricao += segment.text + " "
            return transcricao.strip()
        except Exception as e:
            print(f"Erro na transcrição: {e}")
            return ""


# Instância global
gior = GiorWeb()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = gior.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('ativar_sistema')
def handle_ativar():
    success = gior.ativar_camera()
    emit('sistema_status', {'ativo': gior.sistema_ativo, 'mensagem': 'Sistema ativado!' if success else 'Câmera já está ativa'})

@socketio.on('desativar_sistema')
def handle_desativar():
    success = gior.desativar_camera()
    emit('sistema_status', {'ativo': gior.sistema_ativo, 'mensagem': 'Sistema desativado!' if success else 'Sistema já está inativo'})

@socketio.on('capturar_imagem')
def handle_capturar():
    if gior.capturar_imagem():
        emit('imagem_capturada', {'success': True, 'mensagem': 'Imagem capturada!', 'image_url': '/static/captured.jpg'})
    else:
        emit('imagem_capturada', {'success': False, 'mensagem': 'Erro ao capturar imagem'})

@socketio.on('processar_audio')
def handle_audio(data):
    try:
        # Receber áudio em base64
        audio_data = data['audio']

        # Decodificar e salvar
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        audio_path = 'static/temp_audio.wav'
        with open(audio_path, 'wb') as f:
            f.write(audio_bytes)

        # Transcrever
        transcricao = gior.transcribe_audio(audio_path)
        gior.pergunta = transcricao

        emit('transcricao_completa', {'transcricao': transcricao})

    except Exception as e:
        emit('erro', {'mensagem': f'Erro ao processar áudio: {str(e)}'})

@socketio.on('obter_descricao')
def handle_descricao():
    try:
        pergunta = gior.pergunta if gior.pergunta else "Descreva a cena e me dê o seu veredito de moda."
        
        emit('processando', {'mensagem': 'Processando os 3 feedbacks em paralelo...'})

        encoded_image = gior.encode_image()
        if not encoded_image:
            emit('erro', {'mensagem': 'Erro: Nenhuma imagem capturada para análise. Por favor, capture ou envie uma foto.'})
            return

        # --- Chamadas de API em Paralelo (OpenAI, Gemini, Hugging Face) ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # OpenAI (Já é contextual)
            future_openai = executor.submit(gior.obter_resposta, pergunta)
            
            # Gemini (Agora recebe a pergunta)
            future_gemini = executor.submit(gior.obter_analise_gemini, encoded_image, pergunta) 
            
            # Hugging Face (Agora recebe a pergunta)
            future_hf = executor.submit(gior.obter_analise_hf, pergunta)

            # Coleta dos resultados
            try: resposta_gior = future_openai.result(timeout=60)
            except Exception as e: resposta_gior = f"Erro OpenAI: {str(e)}"

            try: resposta_gemini = future_gemini.result(timeout=60)
            except Exception as e: resposta_gemini = f"Erro Gemini: {str(e)}"

            try: resposta_hf = future_hf.result(timeout=60)
            except Exception as e: resposta_hf = f"Erro Hugging Face: {str(e)}"
        
        # --- GERAÇÃO DE HTML COMPLETO NO PYTHON ---
        
        def format_to_html(text):
            return text.replace('**', '<strong>').replace('\n', '<br>')

        # 1. Crítica do GIOR (OpenAI)
        html_gior = f"""
            <div class="feedback-container">
                <h3>1. 👔 Crítica do Consultor GIOR (OpenAI):</h3>
                <p>{format_to_html(resposta_gior)}</p>
            </div>
        """
        # 2. Análise Técnica (Gemini Vision)
        html_gemini = f"""
            <div class="feedback-container">
                <h3>2. Análise Do Gemini:</h3>
                <p>{format_to_html(resposta_gemini)}</p>
            </div>
        """
        # 3. Feedback Aleatório/Contextual (Hugging Face)
        html_hf = f"""
            <div class="feedback-container">
                <h3>3. 💬 Feedback Do Hugging Face:</h3>
                <p>{format_to_html(resposta_hf)}</p>
            </div>
        """

        feedback_final_html = html_gior + html_gemini + html_hf

        texto_audio = f"A Crítica de Moda GIOR: {resposta_gior}"
        audio_path = gior.gerar_audio(texto_audio)
        
        gior.pergunta = ''

        
        emit('descricao_completa', {
            'resposta': feedback_final_html, 
            'audio_url': '/' + audio_path if audio_path else None
        })
        
    except Exception as e:
        print(f"Erro no handle_descricao principal: {e}")
        emit('erro', {'mensagem': f'Erro: {str(e)}'})

@socketio.on('limpar_pergunta')
def handle_limpar():
    gior.pergunta = ''
    emit('pergunta_limpa', {'mensagem': 'Pergunta limpa'})

@socketio.on('upload_imagem')
def handle_upload(data):
    try:
        # Receber imagem em base64
        image_data = data['image']
        
        # Decodificar e salvar
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image_path = 'frames/uploaded_frame.jpg'
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        # Salvar também para exibição
        with open('static/captured.jpg', 'wb') as f:
            f.write(image_bytes)
        
        # Definir como imagem capturada
        gior.imagem_capturada = image_path
        
        emit('imagem_uploaded', {
            'success': True, 
            'mensagem': 'Imagem enviada com sucesso!',
            'image_url': '/static/captured.jpg'
        })
        
    except Exception as e:
        emit('erro', {'mensagem': f'Erro ao enviar imagem: {str(e)}'})

@app.route('/api/feedbacks', methods=['GET'])
def api_feedbacks():
    pergunta = request.args.get('q', 'Descreva a cena')
    # montar prompt: contexto + histórico + pergunta + imagem (se houver)
    encoded_image = gior.encode_image()
    prompt_text = gior.contexto + "\n\nConversa atual:\n" + (gior.historico_conversa or "") + "\n\nAluno: " + pergunta
    if encoded_image:
        prompt_text += "\n\n[IMAGEM_BASE64]:" + encoded_image

    # chamar em paralelo: OpenAI (obter_resposta), Gemini e Hugging Face
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        f_openai = ex.submit(gior.obter_resposta, pergunta)
        f_gemini = ex.submit(gior.obter_analise_gemini, encoded_image, pergunta)
        f_hf = ex.submit(gior.obter_analise_hf, pergunta)

        try:
            openai_res = f_openai.result(timeout=35)
            openai_obj = {'source': 'openai', 'ok': True, 'data': openai_res}
        except Exception as e:
            openai_obj = {'source': 'openai', 'ok': False, 'error': str(e)}

        try:
            gemini_text = f_gemini.result(timeout=35)
            gemini_res = {'source': 'gemini', 'ok': True, 'data': gemini_text}
        except Exception as e:
            gemini_res = {'source': 'gemini', 'ok': False, 'error': str(e)}

        try:
            hf_text = f_hf.result(timeout=35)
            hf_res = {'source': 'huggingface', 'ok': True, 'data': hf_text}
        except Exception as e:
            hf_res = {'source': 'huggingface', 'ok': False, 'error': str(e)}

    results = [openai_obj, gemini_res, hf_res]
    return jsonify({'count': len(results), 'results': results})


@app.route('/api/gemini_models', methods=['GET'])
def api_gemini_models():
    """Lista modelos disponíveis no Generative Language API para a GEMINI_API_KEY configurada.
    Retorna o JSON direto da API do Google. Não expõe a chave no corpo da resposta do servidor.
    """
    key = os.getenv('GEMINI_API_KEY')
    if not key:
        return jsonify({'ok': False, 'error': 'GEMINI_API_KEY não configurada'}), 400

    url = f'https://generativelanguage.googleapis.com/v1/models?key={key}'
    try:
        r = requests.get(url, timeout=15)
        try:
            j = r.json()
        except Exception:
            j = {'raw_text': (r.text or '')[:1000]}

        if r.status_code != 200:
            body = (r.text or '')[:500]
            return jsonify({'ok': False, 'status': r.status_code, 'body': body}), r.status_code

        # retornar apenas a lista de modelos (se existir)
        models = j.get('models') if isinstance(j, dict) else None
        return jsonify({'ok': True, 'models': models or j})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
