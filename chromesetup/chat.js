document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chatMessages');
    const chatInput = document.getElementById('chatInput');
    const sendChatMessageButton = document.getElementById('sendChatMessage');
    const chatLoader = document.querySelector('.chat-loader');
    let currentPostContent = '';

    const BOT_ICON = `<svg version="1.0" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 90 90" class="message-icon"><g transform="translate(0.000000,90.000000) scale(0.017578,-0.017578)"><path d="M2345 4889 c-820 -85 -1509 -558 -1873 -1285 -49 -99 -132 -298 -132 -319 0 -3 149 -5 331 -5 l331 0 -6 -32 c-4 -18 -9 -117 -13 -220 l-5 -186 -37 -7 c-20 -3 -69 -22 -109 -41 -268 -128 -351 -483 -167 -717 71 -90 179 -158 275 -173 l40 -7 0 -136 c0 -78 6 -164 14 -202 37 -171 151 -303 321 -370 59 -23 69 -24 423 -27 351 -3 362 -4 373 -24 8 -14 7 -28 -1 -48 -9 -21 -9 -33 0 -50 9 -17 9 -29 0 -51 -10 -23 -10 -35 0 -58 9 -22 9 -34 0 -51 -9 -17 -9 -29 0 -50 7 -16 9 -33 6 -38 -3 -5 -52 -12 -108 -15 -207 -13 -408 -88 -533 -200 l-59 -54 44 -26 c72 -41 278 -132 365 -160 392 -127 823 -151 1225 -67 193 40 459 140 610 227 l44 26 -59 54 c-68 62 -227 145 -320 168 -82 20 -189 35 -257 35 -64 0 -75 9 -58 50 9 21 9 33 0 50 -9 17 -9 29 0 51 10 23 10 35 0 58 -9 22 -9 34 0 51 9 17 9 29 0 50 -8 20 -9 34 -1 48 11 20 22 21 373 24 342 4 366 5 420 25 167 63 287 201 324 372 8 38 14 124 14 202 l0 136 42 7 c23 4 70 20 103 36 186 89 293 290 264 498 -12 92 -66 199 -134 266 -57 58 -168 120 -233 131 l-40 6 -5 187 c-4 103 -9 202 -13 220 l-6 32 331 0 c182 0 331 2 331 5 0 3 -16 50 -36 103 -295 790 -1012 1365 -1855 1487 -130 19 -425 27 -544 14z m-114 -2430 c19 -19 21 -29 16 -96 -7 -92 -33 -147 -96 -206 -135 -126 -347 -90 -436 75 -23 43 -30 68 -33 134 -6 118 -17 114 285 114 235 0 244 -1 264 -21z m1188 5 c22 -16 23 -22 19 -98 -3 -66 -10 -91 -33 -134 -89 -164 -296 -200 -433 -77 -65 59 -92 115 -99 208 -5 67 -3 77 16 96 20 20 29 21 264 21 209 0 246 -2 266 -16z"/></g></svg>`;
    
    const USER_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 90 90" class="message-icon"><path d="M 45 0 C 20.147 0 0 20.147 0 45 c 0 24.853 20.147 45 45 45 s 45 -20.147 45 -45 C 90 20.147 69.853 0 45 0 z M 45 22.007 c 8.899 0 16.14 7.241 16.14 16.14 c 0 8.9 -7.241 16.14 -16.14 16.14 c -8.9 0 -16.14 -7.24 -16.14 -16.14 C 28.86 29.248 36.1 22.007 45 22.007 z M 45 83.843 c -11.135 0 -21.123 -4.885 -27.957 -12.623 c 3.177 -5.75 8.144 -10.476 14.05 -13.341 c 2.009 -0.974 4.354 -0.958 6.435 0.041 c 2.343 1.126 4.857 1.696 7.473 1.696 c 2.615 0 5.13 -0.571 7.473 -1.696 c 2.083 -1 4.428 -1.015 6.435 -0.041 c 5.906 2.864 10.872 7.591 14.049 13.341 C 66.123 78.957 56.135 83.843 45 83.843 z"/></svg>`;
  
    // Function to show/hide loader
    const setLoading = (loading, isInitial = false) => {
      chatLoader.style.display = loading ? 'flex' : 'none';
      
      if (loading) {
        chatLoader.innerHTML = `
          ${BOT_ICON}
          <div class="loader-content">
            <span>${isInitial ? 'Loading post content' : 'Thinking'}</span>
            <div class="dots">
              <div class="dot"></div>
              <div class="dot"></div>
              <div class="dot"></div>
            </div>
          </div>
        `;
      }
    };

    // Function to get current post content
    const getCurrentPostContent = async () => {
      try {
        setLoading(true, true);
        // Get the current tab and send message to content.js
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        const response = await chrome.tabs.sendMessage(tab.id, { action: "getUrl" });
        if (response && response.url) {
          // Fetch the summary from our server
          const encodedUrl = encodeURIComponent(response.url);
          const summaryResponse = await fetch(`http://localhost:8000/summarize?url=${encodedUrl}`);
          const data = await summaryResponse.json();
          return data.Summary || '';
        }
      } catch (err) {
        console.error('Error getting post content:', err);
        return '';
      } finally {
        setLoading(false);
      }
    };

    // Initialize post content when chat is opened
    const initializeChat = async () => {
      currentPostContent = await getCurrentPostContent();
      if (currentPostContent) {
        const contextMsg = document.createElement('div');
        contextMsg.classList.add('message', 'bot-message');
        contextMsg.innerHTML = `
          ${BOT_ICON}
          <div class="message-bubble">I'm ready to discuss this Reddit post with you. What would you like to know?</div>
        `;
        chatMessages.appendChild(contextMsg);
      }
    };
    initializeChat();
  
    // Send a chat message
    sendChatMessageButton.addEventListener('click', async () => {
      const userMessage = chatInput.value.trim();
      if (!userMessage) return;
  
      // Display user message
      const userMsgDiv = document.createElement('div');
      userMsgDiv.classList.add('message', 'user-message');
      userMsgDiv.innerHTML = `
        ${USER_ICON}
        <div class="message-bubble">${userMessage}</div>
      `;
      chatMessages.appendChild(userMsgDiv);
  
      chatInput.value = '';
      chatMessages.scrollTop = chatMessages.scrollHeight;
  
      try {
        setLoading(true);
        
        const response = await fetch('http://localhost:8000/chat', {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify({ 
            user_message: userMessage,
            post_content: currentPostContent 
          })
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data && data.bot_reply) {
          const botMsgDiv = document.createElement('div');
          botMsgDiv.classList.add('message', 'bot-message');
          botMsgDiv.innerHTML = `
            ${BOT_ICON}
            <div class="message-bubble">${data.bot_reply}</div>
          `;
          chatMessages.appendChild(botMsgDiv);
          chatMessages.scrollTop = chatMessages.scrollHeight;
        }
      } catch (err) {
        console.error('Chat error:', err);
        const errorMsgDiv = document.createElement('div');
        errorMsgDiv.classList.add('message', 'bot-message');
        errorMsgDiv.innerHTML = `
          ${BOT_ICON}
          <div class="message-bubble">Error: ${err.message}. Make sure the chat server is running on port 8000.</div>
        `;
        chatMessages.appendChild(errorMsgDiv);
      } finally {
        setLoading(false);
      }
    });

    chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessageButton.click();
      }
    });
});
