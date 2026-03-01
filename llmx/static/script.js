const chat = document.getElementById('chat');
const input = document.getElementById('msg');
const sendBtn = document.getElementById('send');
const trashBtn = document.getElementById('new');
const inputArea = document.getElementById('input-area');
const imageModal = document.getElementById('image-modal');
let abortController = null;

imageModal.addEventListener('click', () => imageModal.classList.remove('active'));
document.addEventListener('keydown', e => { 
    if (e.key === 'Escape') {
        if (imageModal.classList.contains('active')) {
            imageModal.classList.remove('active');
        } else if (abortController) {
            abortController.abort();
        }
    }
});

function addMessage(content, type = 'assistant') {
    const div = document.createElement('div');
    div.className = `message ${type}`;
    div.innerHTML = content;
    chat.appendChild(div);
    for (const img of div.querySelectorAll('img')) {
        img.addEventListener('click', e => {
            imageModal.innerHTML = '';
            const fullImg = document.createElement('img');
            fullImg.src = img.src;
            imageModal.appendChild(fullImg);
            imageModal.classList.add('active');
        });
    }
    const last = chat.lastElementChild;
    if (last) last.scrollIntoView({block: 'nearest', behavior: 'auto'});
}

function setTyping() {
    addMessage('<span class="typing">Thinking</span>', 'system');
}

function clearTyping() {
    for (const el of chat.querySelectorAll('.message.system')) {
        el.remove();
    }
}

async function send() {
    const msg = input.value.trim();
    if (!msg) return;
    
    addMessage(msg, 'user');
    input.value = '';
    sendBtn.disabled = false;
    sendBtn.classList.add('loading');
    abortController = new AbortController();
    
    setTyping();
    
    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: msg}),
            signal: abortController.signal
        });
        
        if (!res.ok) {
            clearTyping();
            addMessage(`Error: ${res.status}`, 'system');
            sendBtn.disabled = false;
            sendBtn.classList.remove('loading');
            return;
        }
        
        clearTyping();
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let done = false;
        
        while (!done) {
            const result = await reader.read();
            done = result.done;
            
            if (result.value) {
                buffer += decoder.decode(result.value, {stream: !done});
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') {
                            done = true;
                            break;
                        }
                        try {
                            const event = JSON.parse(data);
                            handleEvent(event);
                        } catch (e) {
                        }
                    }
                }
            }
        }
    } catch (e) {
        clearTyping();
        if (e.name === 'AbortError') {
            addMessage('<span style="color: var(--peach)">Stopped</span>', 'system');
        } else {
            addMessage(`Error: ${e.message}`, 'system');
        }
    }
    
    sendBtn.disabled = false;
    sendBtn.classList.remove('loading');
    abortController = null;
}

function handleEvent(event) {
    const type = event.type;
    const content = event.content || '';
    
    if (type === 'thinking') {
        addMessage(content, 'thinking');
    } else if (type === 'tool_call') {
        addMessage(content, 'tool-call');
    } else if (type === 'message') {
        addMessage(content, 'assistant');
    }
}

sendBtn.addEventListener('click', () => {
    if (sendBtn.classList.contains('loading') && abortController) {
        abortController.abort();
        return;
    }
    send();
});
trashBtn.addEventListener('click', () => {
    if (confirm('Will create a new session and delete this, continue?')) {
        window.location.href = '/';
    }
});
input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
    }
});

window.addEventListener('load', () => input.focus());
inputArea.addEventListener('click', () => input.focus());
input.addEventListener('touchstart', () => input.focus());
