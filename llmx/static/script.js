const chat = document.getElementById('chat');
const input = document.getElementById('msg');
const sendBtn = document.getElementById('send');
const trashBtn = document.getElementById('new');
const inputArea = document.getElementById('input-area');
const imageModal = document.getElementById('image-modal');
let abortController = null;
let sessionId = null;

function getSessionIdFromUrl() {
    const path = window.location.pathname;
    const match = path.match(/^\/([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$/);
    return match ? match[1] : null;
}

function formatMessageContent(content) {
    return content;
}

async function loadHistory() {
    sessionId = getSessionIdFromUrl();
    if (!sessionId) return;
    
    try {
        const res = await fetch(`/session/${sessionId}/messages`);
        if (!res.ok) return;
        
        const data = await res.json();
        const messages = data.messages || [];
        
        for (const msg of messages) {
            if (msg.role === 'system') continue;
            
            if (msg.role === 'user') {
                addMessage(msg.content, 'user');
            } else if (msg.role === 'assistant') {
                if (msg.reasoning) {
                    addMessage(`<span class='thinking-header'>thinking</span>${escapeHtml(msg.reasoning)}`, 'thinking');
                }
                if (msg.content) {
                    addMessage(msg.content, 'assistant');
                }
                if (msg.tool_calls) {
                    for (const tc of msg.tool_calls) {
                        const func = tc.function || {};
                        const args = func.arguments ? JSON.parse(func.arguments) : {};
                        const toolId = tc.id || '';
                        const result = msg.tool_call_id ? '' : null;
                        addMessage(formatToolCallHtml(func.name, args, result), 'tool-call');
                    }
                }
            } else if (msg.role === 'tool') {
                const prevToolCall = chat.querySelector('.message.tool-call:last-child');
                if (prevToolCall) {
                    const resultHtml = `<span class="result-toggle" onclick="this.classList.toggle('expanded'); const c = this.nextElementSibling; c.classList.toggle('collapsed'); this.textContent = this.classList.contains('expanded') ? '[▲ result]' : '[▼ result]'">[▼ result]</span><pre class="result-content collapsed"><code>${escapeHtml(msg.content)}</code></pre>`;
                    prevToolCall.insertAdjacentHTML('beforeend', resultHtml);
                }
            }
        }
    } catch (e) {
        console.error('Failed to load history:', e);
    }
}

function formatToolCallHtml(toolName, args, result) {
    const argsStr = JSON.stringify(args, null, 2);
    const escapedArgs = escapeHtml(argsStr);
    let resultHtml = '';
    if (result) {
        const truncated = result.length > 500 ? result.slice(0, 500) + '...' : result;
        const escapedResult = escapeHtml(truncated);
        resultHtml = `<span class="result-toggle" onclick="this.classList.toggle('expanded'); const c = this.nextElementSibling; c.classList.toggle('collapsed'); this.textContent = this.classList.contains('expanded') ? '[▲ result]' : '[▼ result]'">[▼ result]</span><pre class="result-content collapsed"><code>${escapedResult}</code></pre>`;
    }
    return `<span class="tool-name">${escapeHtml(toolName)}</span>\n<div class="tool-args"><pre><code>${escapedArgs}</code></pre></div>${resultHtml}`;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

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
    inputArea.classList.add('loading');
    abortController = new AbortController();
    
    setTyping();
    
    try {
        const url = sessionId ? `/chat?session_id=${sessionId}` : '/chat';
        const res = await fetch(url, {
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
            inputArea.classList.remove('loading');
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
        
        if (buffer.startsWith('data: ')) {
            const data = buffer.slice(6);
            if (data !== '[DONE]') {
                try {
                    const event = JSON.parse(data);
                    handleEvent(event);
                } catch (e) {
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
    inputArea.classList.remove('loading');
    abortController = null;
}

function handleEvent(event) {
    const type = event.type;
    const content = event.content || '';
    
    if (type === 'thinking') {
        addMessage(content, 'thinking');
    } else if (type === 'tool_call') {
        addMessage(content, 'tool-call');
    } else if (type === 'tool_result') {
        const lastToolCall = chat.querySelector('.message.tool-call:last-child');
        if (lastToolCall) {
            const loading = lastToolCall.querySelector('.tool-loading');
            if (loading) loading.remove();
            const resultHtml = `<span class="result-toggle" onclick="this.classList.toggle('expanded'); const c = this.nextElementSibling; c.classList.toggle('collapsed'); this.textContent = this.classList.contains('expanded') ? '[▲ result]' : '[▼ result]'">[▼ result]</span><pre class="result-content collapsed"><code>${content}</code></pre>`;
            lastToolCall.insertAdjacentHTML('beforeend', resultHtml);
        }
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

window.addEventListener('load', async () => {
    await loadHistory();
    input.focus();
});
inputArea.addEventListener('click', () => input.focus());
input.addEventListener('touchstart', () => input.focus());
