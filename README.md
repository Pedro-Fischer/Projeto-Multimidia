# 👔 Consultor de Moda GIOR

**Críticas de moda brutalmente honestas com Inteligência Artificial**

O Consultor de Moda GIOR é uma aplicação web que usa IA para analisar seu look e fornecer críticas diretas, honestas e construtivas sobre suas escolhas de moda. Com um olhar apurado e feedback impiedoso (mas sempre útil), o GIOR ajuda você a entender o que funciona e o que não funciona no seu visual.

## ✨ Características

- 👗 **Análise Visual Completa** - Avalia peças, cores, texturas e caimento
- 💬 **Críticas Honestas** - Feedback direto sem rodeios
- 🎯 **Estrutura em 3 Pontos** - Veredito, Conserto e Dica de Styling
- 📸 **Captura de Look** - Tire foto do seu outfit
- 🎤 **Perguntas por Voz** - Grave suas dúvidas sobre moda
- 🔊 **Resposta em Áudio** - Ouça a crítica com voz natural
- 🌐 **Interface Web Moderna** - Design elegante e responsivo

## 🎯 Exemplos de Uso

### Perguntas que você pode fazer:

- "Este look combina para um jantar formal?"
- "Posso ir assim para uma reunião de negócios?"
- "Essa combinação funciona?"
- "O que você acha deste outfit?"
- "Este look está adequado para um casamento?"

### Como o GIOR responde:

A crítica é estruturada em **3 partes**:

1. **O Veredito** - Declaração direta sobre o look
2. **O Conserto** - Como corrigir o erro principal
3. **Dica de Styling** - Sugestão para elevar o visual

**Exemplo de resposta:**
> "O Consultor GIOR tem um veredito: Este look precisa de ajustes urgentes. **Veredito:** A combinação de cores está confusa e sem harmonia. **O Conserto:** Troque a peça superior por algo em tom neutro para balancear. **Dica de Styling:** Um cinto estruturado criaria um ponto focal necessário."

## 🚀 Instalação

### Requisitos
- Python 3.11+
- Webcam
- Microfone
- Chave API da OpenAI

### Instalação Rápida

1. **Instalar dependências:**
```bash
pip install flask flask-socketio python-socketio speechrecognition pyaudio opencv-python pillow openai langchain langchain-openai langchain-community faster-whisper python-dotenv
```

2. **Configurar API Key:**

Crie arquivo `.env`:
```
OPENAI_API_KEY="sua-chave-aqui"
```

3. **Executar:**
```bash
python app.py
```

4. **Acessar:**
```
http://localhost:5000
```

## 🎮 Como Usar

### Passo a Passo:

1. **🚀 Ativar Sistema** - Liga a câmera
2. **📸 Capturar Look** - Tire foto do seu outfit completo
3. **🎤 Fazer Pergunta** - Grave sua dúvida sobre o look
4. **⏹️ Parar** - Finalize a gravação
5. **💬 Obter Crítica** - Receba e ouça o feedback do GIOR

### Dicas para melhor análise:

✅ Posicione-se de corpo inteiro na câmera  
✅ Boa iluminação é essencial  
✅ Mostre todos os detalhes do look  
✅ Seja específico na pergunta (mencione o evento/contexto)  

## 🎨 Personalidade do GIOR

O Consultor de Moda GIOR é:

- ⚡ **Direto e Honesto** - Sem rodeios ou falsas cortesias
- 🎯 **Construtivo** - Toda crítica vem com solução
- 👔 **Profissional** - Usa vocabulário técnico de moda
- 💪 **Impiedoso mas Útil** - A verdade dói, mas ajuda

## 📁 Estrutura do Projeto

```
narrador-gior/
├── app.py                 # Backend Flask
├── templates/
│   └── index.html        # Interface web (tema dark/gold)
├── static/               # Assets gerados
│   ├── captured.jpg      # Look capturado
│   └── resposta.mp3      # Áudio da crítica
├── frames/               # Frames da câmera
├── .env                  # Configuração (não commitar!)
└── pyproject.toml        # Dependências
```

## 🛠️ Tecnologias

- **Flask + SocketIO** - Backend web real-time
- **OpenCV** - Captura de vídeo
- **GPT-4o-mini** - Análise de moda com IA
- **OpenAI TTS** - Síntese de voz
- **Faster Whisper** - Transcrição de áudio
- **Langchain** - Framework LLM

## 🎨 Design

- **Tema Dark & Gold** - Elegante e sofisticado
- **Gradientes Dourados** - Botões com acabamento premium
- **Layout Responsivo** - Funciona em qualquer tela
- **Animações Suaves** - Feedback visual para todas as ações

## 🌐 Acesso Remoto

Para acessar de outros dispositivos:

1. A aplicação roda em: `http://0.0.0.0:5000`
2. Encontre seu IP: `ipconfig` (Windows) ou `ifconfig` (Mac/Linux)
3. Acesse de outros dispositivos: `http://SEU_IP:5000`

## 💡 Casos de Uso

- 📱 **Antes de Sair** - Valide seu look rapidamente
- 👔 **Entrevistas** - Garanta que está apropriado
- 💼 **Reuniões** - Check de profissionalismo
- 🎉 **Eventos** - Confirme a adequação ao dress code
- 🎓 **Aprendizado** - Entenda conceitos de moda

## ⚠️ Avisos

- 🎯 O GIOR é **direto** - prepare-se para críticas honestas
- 💬 Feedback construtivo, mas sem floreios
- 📸 Qualidade da imagem afeta a análise
- 🌐 Requer conexão com internet (APIs OpenAI)

## 📝 Licença

Projeto educacional e experimental - Uso livre para fins não comerciais.

---

**Desenvolvido para ajudar você a arrasar no visual! 👔✨**
