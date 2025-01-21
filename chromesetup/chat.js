document.addEventListener('DOMContentLoaded', async () => {
    const chatMessages = document.getElementById('chatMessages');
    const chatInput = document.getElementById('chatInput');
    const sendChatMessageButton = document.getElementById('sendChatMessage');
    const chatLoader = document.querySelector('.chat-loader');
    let currentPostContent = '';

    const BOT_ICON = `<svg width="24" height="24" class="message-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 216 216" xml:space="preserve" xmlns:xlink="http://www.w3.org/1999/xlink">
      <defs>
        <radialGradient id="snoo-radial-gragient" cx="169.75" cy="92.19" fx="169.75" fy="92.19" r="50.98" gradientTransform="translate(0 11.64) scale(1 .87)" gradientUnits="userSpaceOnUse">
          <stop offset="0" stop-color="#feffff"></stop>
          <stop offset=".4" stop-color="#feffff"></stop>
          <stop offset=".51" stop-color="#f9fcfc"></stop>
          <stop offset=".62" stop-color="#edf3f5"></stop>
          <stop offset=".7" stop-color="#dee9ec"></stop>
          <stop offset=".72" stop-color="#d8e4e8"></stop>
          <stop offset=".76" stop-color="#ccd8df"></stop>
          <stop offset=".8" stop-color="#c8d5dd"></stop>
          <stop offset=".83" stop-color="#ccd6de"></stop>
          <stop offset=".85" stop-color="#d8dbe2"></stop>
          <stop offset=".88" stop-color="#ede3e9"></stop>
          <stop offset=".9" stop-color="#ffebef"></stop>
        </radialGradient>
        <radialGradient id="snoo-radial-gragient-2" cx="47.31" fx="47.31" r="50.98" xlink:href="#snoo-radial-gragient"></radialGradient>
        <radialGradient id="snoo-radial-gragient-3" cx="109.61" cy="85.59" fx="109.61" fy="85.59" r="153.78" gradientTransform="translate(0 25.56) scale(1 .7)" xlink:href="#snoo-radial-gragient"></radialGradient>
        <radialGradient id="snoo-radial-gragient-4" cx="-6.01" cy="64.68" fx="-6.01" fy="64.68" r="12.85" gradientTransform="translate(81.08 27.26) scale(1.07 1.55)" gradientUnits="userSpaceOnUse">
          <stop offset="0" stop-color="#f60"></stop>
          <stop offset=".5" stop-color="#ff4500"></stop>
          <stop offset=".7" stop-color="#fc4301"></stop>
          <stop offset=".82" stop-color="#f43f07"></stop>
          <stop offset=".92" stop-color="#e53812"></stop>
          <stop offset="1" stop-color="#d4301f"></stop>
        </radialGradient>
        <radialGradient id="snoo-radial-gragient-5" cx="-73.55" cy="64.68" fx="-73.55" fy="64.68" r="12.85" gradientTransform="translate(62.87 27.26) rotate(-180) scale(1.07 -1.55)" xlink:href="#snoo-radial-gragient-4"></radialGradient>
        <radialGradient id="snoo-radial-gragient-6" cx="107.93" cy="166.96" fx="107.93" fy="166.96" r="45.3" gradientTransform="translate(0 57.4) scale(1 .66)" gradientUnits="userSpaceOnUse">
          <stop offset="0" stop-color="#172e35"></stop>
          <stop offset=".29" stop-color="#0e1c21"></stop>
          <stop offset=".73" stop-color="#030708"></stop>
          <stop offset="1" stop-color="#000"></stop>
        </radialGradient>
        <radialGradient id="snoo-radial-gragient-7" cx="147.88" cy="32.94" fx="147.88" fy="32.94" r="39.77" gradientTransform="translate(0 .54) scale(1 .98)" xlink:href="#snoo-radial-gragient"></radialGradient>
        <radialGradient id="snoo-radial-gragient-8" cx="131.31" cy="73.08" fx="131.31" fy="73.08" r="32.6" gradientUnits="userSpaceOnUse">
          <stop offset=".48" stop-color="#7a9299"></stop>
          <stop offset=".67" stop-color="#172e35"></stop>
          <stop offset=".75" stop-color="#000"></stop>
          <stop offset=".82" stop-color="#172e35"></stop>
        </radialGradient>
      </defs>
      <path class="snoo-cls-10" fill="#ff4500" d="m108,0h0C48.35,0,0,48.35,0,108h0c0,29.82,12.09,56.82,31.63,76.37l-20.57,20.57c-4.08,4.08-1.19,11.06,4.58,11.06h92.36s0,0,0,0c59.65,0,108-48.35,108-108h0C216,48.35,167.65,0,108,0Z"></path>
      <circle class="snoo-cls-1" fill="url(#snoo-radial-gragient)" cx="169.22" cy="106.98" r="25.22"></circle>
      <circle class="snoo-cls-2" fill="url(#snoo-radial-gragient-2)" cx="46.78" cy="106.98" r="25.22"></circle>
      <ellipse class="snoo-cls-3" fill="url(#snoo-radial-gragient-3)" cx="108.06" cy="128.64" rx="72" ry="54"></ellipse>
      <path class="snoo-cls-4" fill="url(#snoo-radial-gragient-4)" d="m86.78,123.48c-.42,9.08-6.49,12.38-13.56,12.38s-12.46-4.93-12.04-14.01c.42-9.08,6.49-15.02,13.56-15.02s12.46,7.58,12.04,16.66Z"></path>
      <path class="snoo-cls-7" fill="url(#snoo-radial-gragient-5)" d="m129.35,123.48c.42,9.08,6.49,12.38,13.56,12.38s12.46-4.93,12.04-14.01c-.42-9.08-6.49-15.02-13.56-15.02s-12.46,7.58-12.04,16.66Z"></path>
      <ellipse class="snoo-cls-11" fill="#ffc49c" cx="79.63" cy="116.37" rx="2.8" ry="3.05"></ellipse>
      <ellipse class="snoo-cls-11" fill="#ffc49c" cx="146.21" cy="116.37" rx="2.8" ry="3.05"></ellipse>
      <path class="snoo-cls-5" fill="url(#snoo-radial-gragient-6)" d="m108.06,142.92c-8.76,0-17.16.43-24.92,1.22-1.33.13-2.17,1.51-1.65,2.74,4.35,10.39,14.61,17.69,26.57,17.69s22.23-7.3,26.57-17.69c.52-1.23-.33-2.61-1.65-2.74-7.77-.79-16.16-1.22-24.92-1.22Z"></path>
      <circle class="snoo-cls-8" fill="url(#snoo-radial-gragient-7)" cx="147.49" cy="49.43" r="17.87"></circle>
      <path class="snoo-cls-6" fill="url(#snoo-radial-gragient-8)" d="m107.8,76.92c-2.14,0-3.87-.89-3.87-2.27,0-16.01,13.03-29.04,29.04-29.04,2.14,0,3.87,1.73,3.87,3.87s-1.73,3.87-3.87,3.87c-11.74,0-21.29,9.55-21.29,21.29,0,1.38-1.73,2.27-3.87,2.27Z"></path>
      <path class="snoo-cls-9" fill="#842123" d="m62.82,122.65c.39-8.56,6.08-14.16,12.69-14.16,6.26,0,11.1,6.39,11.28,14.33.17-8.88-5.13-15.99-12.05-15.99s-13.14,6.05-13.56,15.2c-.42,9.15,4.97,13.83,12.04,13.83.17,0,.35,0,.52,0-6.44-.16-11.3-4.79-10.91-13.2Z"></path>
      <path class="snoo-cls-9" fill="#842123" d="m153.3,122.65c-.39-8.56-6.08-14.16-12.69-14.16-6.26,0-11.1,6.39-11.28,14.33-.17-8.88,5.13-15.99,12.05-15.99,7.07,0,13.14,6.05,13.56,15.2.42,9.15-4.97,13.83-12.04,13.83-.17,0-.35,0-.52,0,6.44-.16,11.3-4.79,10.91-13.2Z"></path>
    </svg>`;
    
    const USER_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 90 90" class="message-icon"><path d="M 45 0 C 20.147 0 0 20.147 0 45 c 0 24.853 20.147 45 45 45 s 45 -20.147 45 -45 C 90 20.147 69.853 0 45 0 z M 45 22.007 c 8.899 0 16.14 7.241 16.14 16.14 c 0 8.9 -7.241 16.14 -16.14 16.14 c -8.9 0 -16.14 -7.24 -16.14 -16.14 C 28.86 29.248 36.1 22.007 45 22.007 z M 45 83.843 c -11.135 0 -21.123 -4.885 -27.957 -12.623 c 3.177 -5.75 8.144 -10.476 14.05 -13.341 c 2.009 -0.974 4.354 -0.958 6.435 0.041 c 2.343 1.126 4.857 1.696 7.473 1.696 c 2.615 0 5.13 -0.571 7.473 -1.696 c 2.083 -1 4.428 -1.015 6.435 -0.041 c 5.906 2.864 10.872 7.591 14.049 13.341 C 66.123 78.957 56.135 83.843 45 83.843 z"/></svg>`;
  
    // Function to escape HTML to prevent XSS
    const escapeHtml = (text) => {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    };

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
            
            // Get the current tab
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (!tab) {
                throw new Error('No active tab found');
            }

            // Get URL directly from tab
            const url = tab.url;
            if (!url) {
                throw new Error('No URL found');
            }

            if (!url.includes('reddit.com') || !url.includes('/comments/')) {
                throw new Error('This page is not a Reddit post');
            }

            // Fetch the summary from the server
            const encodedUrl = encodeURIComponent(url);
            const summaryResponse = await fetch(`http://localhost:8000/summarize?url=${encodedUrl}`);
            if (!summaryResponse.ok) {
                const errorData = await summaryResponse.json();
                throw new Error(errorData.error || 'Failed to get post summary');
            }
            
            const data = await summaryResponse.json();
            if (!data.Summary) {
                throw new Error('No summary available for this post');
            }
            
            return data.Summary;
        } catch (err) {
            throw err;
        } finally {
            setLoading(false);
        }
    };

    // Function to display a message
    const displayMessage = (message, isError = false, isUser = false) => {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', isUser ? 'user-message' : 'bot-message');
        msgDiv.innerHTML = `
            ${isUser ? USER_ICON : BOT_ICON}
            <div class="message-bubble">${isError ? `Error: ${message}` : message}</div>
        `;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    // Function to send chat message
    const sendMessage = async (message) => {
        if (!message.trim()) return;

        // Display user message
        displayMessage(escapeHtml(message), false, true);
        chatInput.value = '';

        try {
            setLoading(true);
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ 
                    user_message: message,
                    post_content: currentPostContent 
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            if (data && data.bot_reply) {
                displayMessage(escapeHtml(data.bot_reply));
            }
        } catch (err) {
            console.error('Chat error:', err);
            displayMessage(err.message + '. Make sure the chat server is running on port 8000.', true);
        } finally {
            setLoading(false);
        }
    };

    // Initialize chat content
    const initializeChat = async () => {
        try {
            currentPostContent = await getCurrentPostContent();
            
            // Send initial message to chat endpoint to set up context
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ 
                    user_message: "Hi",
                    post_content: currentPostContent 
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            displayMessage("I'm ready to discuss this Reddit post with you. What would you like to know?");
        } catch (err) {
            console.error('Error in initializeChat:', err);
            displayMessage(err.message, true);
        }
    };

    // Start loading content immediately
    await initializeChat();

    // Add click event listener to send button
    sendChatMessageButton.addEventListener('click', () => {
        sendMessage(chatInput.value);
    });

    // Add enter key listener
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage(chatInput.value);
        }
    });
});
