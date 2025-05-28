// content.js

console.log('🚀 Content script loaded');

(() => {
  let sidebarEl, msgList, chatForm, chatInput, newChatBtn;
  let isLoggedIn = localStorage.getItem('gazer_logged_in') === 'true';
  let userId = localStorage.getItem('gazer_user_id');
  let chatHistory = JSON.parse(localStorage.getItem('chat_history') || '[]');
  let aiPlaceholderEl = null;

  // 지난 7일치 텍스트 불러오기
  async function fetchRecentTexts(userId, sinceId = 0) {
    console.log('🚧 fetchRecentTexts', { userId, sinceId });
    const res = await fetch(
      `http://yodin2327.dothome.co.kr/gazer/pc/fetch_texts.php?user_id=${userId}&since_id=${sinceId}`
    );
    const json = await res.json();
    console.log('✅ fetched texts', json);
    return json.status === 'ok' ? json.texts.map(t => t.content) : [];
  }

  // 불용어 제거
  function preprocess(contents) {
    console.log('🚧 preprocess', contents);
    const stop = new Set([
      'of','the','and','to','a','in','for','is','on',
      'that','this','with','as','by','an','be','are','or','from','at'
    ]);
    const tokens = [];
    contents.forEach(txt =>
      txt
        .toLowerCase()
        .replace(/[^a-z\s]/g, ' ')
        .split(/\s+/)
        .forEach(w => { if (w && !stop.has(w)) tokens.push(w); })
    );
    console.log('✅ tokens', tokens);
    return tokens;
  }

  // 공동 등장 계산
  function computeCoocc(tokens, windowSize = 2) {
    console.log('🚧 computeCoocc', { tokens, windowSize });
    const cooc = {};
    for (let i = 0; i < tokens.length; i++) {
      for (let j = Math.max(0, i - windowSize); j <= Math.min(tokens.length - 1, i + windowSize); j++) {
        if (i === j) continue;
        const [a, b] = [tokens[i], tokens[j]].sort();
        const key = `${a},${b}`;
        cooc[key] = (cooc[key] || 0) + 1;
      }
    }
    console.log('✅ cooccurrence', cooc);
    return cooc;
  }

  // Perplexity Chat Completions 호출
  async function callAI(messages) {
    console.log('🚧 callAI via background', messages);
    const apiKey = localStorage.getItem('sonal_api_key');
    if (!apiKey) {
      console.warn('⚠️ No Sonal API key set');
      return '[API Key 없음]';
    }
    const payload = {
      model: 'sonar-pro',
      messages
    };
    return new Promise(resolve => {
      chrome.runtime.sendMessage({ type: 'callSonal', apiKey, payload }, resp => {
        console.log('✅ Content: Sonal answered', resp);
        resolve(resp.answer || '[응답 없음]');
      });
    });
  }

  // 메시지 추가
  function appendMsg(role, text) {
    console.log('🚧 appendMsg', { role, text });
    const d = document.createElement('div');
    d.className = `message ${role}`;
    d.textContent = text;
    msgList.appendChild(d);
    msgList.scrollTop = msgList.scrollHeight;
    if (role === 'user' || role === 'ai') {
      chatHistory.push({ who: role, text });
      localStorage.setItem('chat_history', JSON.stringify(chatHistory));
    }
  }

  // 전송 처리
  async function handleSubmit(e) {
    e.preventDefault();
    console.log('🚧 handleSubmit');
    const text = chatInput.value.trim();
    if (!text) return;
    chatInput.value = '';

    if (text.toLowerCase() === 'my api') {
      appendMsg('ai', `현재 Sonal API Key: ${localStorage.getItem('sonal_api_key') || '[없음]'}`);
      return;
    }
    if (text.toLowerCase() === 'change api') {
      appendMsg('ai', '[API 변경 모드로 전환합니다]');
      renderSidebarWithApiInput();
      return;
    }

    appendMsg('user', text);

    aiPlaceholderEl = document.createElement('div');
    aiPlaceholderEl.className = 'message ai placeholder';
    aiPlaceholderEl.textContent = 'AI가 응답 중...';
    msgList.appendChild(aiPlaceholderEl);
    msgList.scrollTop = msgList.scrollHeight;

    const sinceId = parseInt(localStorage.getItem('last_text_id') || '0', 10);
    const texts = await fetchRecentTexts(userId, sinceId);
    const cooc = texts.length ? computeCoocc(preprocess(texts)) : {};

    const sys = { role: 'system', content: '연관관계표 요약:\n' + JSON.stringify(cooc) + 'give short message less than 10 sentence. Keyword that I gave you was my interests. give Citations link individually with the number of the link URL after the end of the chat'};
    const usr = { role: 'user', content: text };
    const aiResp = await callAI([sys, usr]);

    console.log('🚧 replacing placeholder with AI response');
    aiPlaceholderEl.textContent = aiResp;
    aiPlaceholderEl.classList.remove('placeholder');
    chatHistory.push({ who: 'ai', text: aiResp });
    localStorage.setItem('chat_history', JSON.stringify(chatHistory));
    aiPlaceholderEl = null;
  }

  // API 키 입력 모드 렌더링
  function renderSidebarWithApiInput() {
    sidebarEl.innerHTML = '';
    const c = document.createElement('div');
    c.className = 'content';
    c.innerHTML = `
      <div class="status">API Key 변경</div>
      <input type="text" id="api-key-input" placeholder="새 API Key 입력..." class="api-input" />
      <button id="api-key-save" class="api-button">저장</button>
      <button id="back-to-chat" class="api-button">채팅으로 돌아가기</button>
    `;
    sidebarEl.appendChild(c);
    document.getElementById('api-key-save').onclick = () => {
      const k = document.getElementById('api-key-input').value.trim();
      if (!k) return alert('API Key를 입력하세요');
      localStorage.setItem('sonal_api_key', k);
      appendMsg('ai', '새 Sonal API Key 저장되었습니다. 새로운 채팅을 시작합니다.');
      resetChat();
      renderSidebar();
    };
    document.getElementById('back-to-chat').onclick = renderSidebar;
  }

  // 새 채팅 초기화
  function resetChat() {
    chatHistory = [];
    localStorage.setItem('chat_history', JSON.stringify(chatHistory));
    msgList.innerHTML = '';
  }

  // 사이드바 초기화
  function initSidebar() {
    sidebarEl = document.getElementById('gazer-sidebar');
    if (!sidebarEl) {
      sidebarEl = document.createElement('div');
      sidebarEl.id = 'gazer-sidebar';
      document.body.appendChild(sidebarEl);
    }
  }

  // UI 렌더링
  function renderSidebar() {
    sidebarEl.innerHTML = '';
    const c = document.createElement('div');
    c.className = 'content';

    if (!isLoggedIn) {
      c.innerHTML = `
        <input type="text" id="login-username" placeholder="Username" class="login-input"/>
        <input type="password" id="login-password" placeholder="Password" class="login-input"/>
        <button id="login-submit" class="login-button">로그인</button>
      `;
      sidebarEl.appendChild(c);
      document.getElementById('login-submit').onclick = () => {
        const u = document.getElementById('login-username').value.trim();
        const p = document.getElementById('login-password').value.trim();
        console.log('🚧 Content: login button clicked', { u, p });
        if (!u || !p) return alert('아이디와 비밀번호를 입력하세요');
        chrome.runtime.sendMessage(
          { type: 'login', username: u, password: p },
          resp => {
            console.log('🚧 Content: login response callback', resp);
            if (resp.success && resp.data.status === 'ok') {
              isLoggedIn = true;
              localStorage.setItem('gazer_logged_in', 'true');
              userId = resp.data.user_id;
              localStorage.setItem('gazer_user_id', userId);
              localStorage.setItem('last_text_id', '0');
              renderSidebar();
            } else {
              alert('로그인 실패: ' + (resp.data?.message || resp.error));
            }
          }
        );
      };
    } else {
      userId = userId || localStorage.getItem('gazer_user_id');
      c.innerHTML = `
        <div class="status">로그인됨 (User ID: ${userId})</div>
        <button id="new-chat-btn" class="api-button">새 채팅</button>
        <div id="chat-container">
          <div id="message-list"></div>
          <form id="chat-form">
            <input type="text" id="chat-input" placeholder="메시지 입력..." autocomplete="off"/>
            <button type="submit">전송</button>
          </form>
        </div>
      `;
      sidebarEl.appendChild(c);

      newChatBtn = document.getElementById('new-chat-btn');
      msgList    = document.getElementById('message-list');
      chatForm   = document.getElementById('chat-form');
      chatInput  = document.getElementById('chat-input');

      msgList.innerHTML = '';
      chatHistory.forEach(m => appendMsg(m.who, m.text));

      newChatBtn.onclick = resetChat;
      chatForm.addEventListener('submit', handleSubmit);
    }
  }

  // 메시지 토글 리스너
  chrome.runtime.onMessage.addListener(msg => {
    if (msg.toggleSidebar) {
      initSidebar();
      renderSidebar();
      setTimeout(() => sidebarEl.classList.add('visible'), 0);
    } else {
      sidebarEl?.classList.remove('visible');
    }
  });
})();
