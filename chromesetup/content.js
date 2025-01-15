// Notify that content script is loaded
console.log('Reddit ChatBot content script loaded');

// Set up connection as soon as possible
let isReady = false;

function setupMessageListener() {
    if (isReady) return; // Only set up once
    
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        console.log('Content script received message:', request);
        
        if (request.action === "getUrl") {
            const currentUrl = window.location.href;
            console.log('Sending current URL:', currentUrl);
            
            // Check if we're actually on a Reddit post page
            const isRedditPost = currentUrl.match(/reddit\.com\/r\/[^\/]+\/comments\//);
            if (!isRedditPost) {
                console.log('Not a Reddit post page');
                sendResponse({ error: 'Not a Reddit post page' });
                return true;
            }
            
            sendResponse({ url: currentUrl });
        }
        return true; // Keep the message port open for async response
    });
    
    isReady = true;
    console.log('Content script ready to receive messages');
}

// Set up immediately
setupMessageListener();

// Also set up when DOM is ready (backup)
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupMessageListener);
} else {
    setupMessageListener();
}

// Final backup in case the above methods fail
window.addEventListener('load', setupMessageListener);
