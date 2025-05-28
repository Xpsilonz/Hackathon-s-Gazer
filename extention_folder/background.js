// background.js

console.log('⚡️ Background script loaded');

let sidebarVisible = false;

// Toolbar icon 클릭 시 사이드바 토글
chrome.action.onClicked.addListener((tab) => {
  console.log('⚡️ Background: icon clicked');
  sidebarVisible = !sidebarVisible;
  chrome.tabs.sendMessage(tab.id, { toggleSidebar: sidebarVisible }, () => {
    if (chrome.runtime.lastError) {
      console.warn('⚠️ Background: no content listener:', chrome.runtime.lastError.message);
    } else {
      console.log('✅ Background: toggleSidebar sent, visible=', sidebarVisible);
    }
  });
});

// content.js 로부터 메시지 수신
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  console.log('⚡️ Background: received message', msg);

  // --- LOGIN 핸들러 (변경 없음) ---
  if (msg.type === 'login') {
    console.log('⚡️ Background: login attempt for', msg.username);
    fetch('http://yodin2327.dothome.co.kr/gazer/ex/login_ext.php', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: msg.username, password: msg.password })
    })
    .then(async res => {
      console.log('⚡️ Background: login HTTP status', res.status);
      const text = await res.text();
      console.log('⚡️ Background: login raw response', text);
      try {
        const data = JSON.parse(text);
        console.log('✅ Background: login JSON parsed', data);
        sendResponse({ success: true, data });
      } catch (e) {
        console.error('❌ Background: login JSON parse error', e);
        sendResponse({ success: false, error: 'Invalid JSON', details: text });
      }
    })
    .catch(err => {
      console.error('❌ Background: login network error', err);
      sendResponse({ success: false, error: err.message });
    });
    return true;  // async sendResponse 유지
  }

  // --- Perplexity Chat Completions 호출 ---
  if (msg.type === 'callSonal') {
    console.log('⚡️ Background: Sonal API request', msg.payload);
    fetch('https://api.perplexity.ai/chat/completions', {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${msg.apiKey}`
      },
      body: JSON.stringify({
        model: msg.payload.model,       // e.g. 'sonar-pro'
        messages: msg.payload.messages  // [{role,content}, …]
      })
    })
    .then(async res => {
      const text = await res.text();
      if (!res.ok) {
        console.error('❌ Background: HTTP error', res.status, text);
        sendResponse({ answer: `[API 오류: HTTP ${res.status}]` });
        return;
      }
      try {
        const json = JSON.parse(text);
        console.log('✅ Background: Sonal JSON', json);
        const answer = json.choices?.[0]?.message?.content;
        sendResponse({ answer: answer ?? '[응답 없음]' });
      } catch (e) {
        console.error('❌ Background: JSON parse error', e, text);
        sendResponse({ answer: '[응답 없음]' });
      }
    })
    .catch(err => {
      console.error('❌ Background: network error', err);
      sendResponse({ answer: `[API 오류: ${err.message}]` });
    });
    return true;  // async sendResponse 유지
  }

  // 그 외 메시지는 처리하지 않음
});
