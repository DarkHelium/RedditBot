function showLoading() {
  document.getElementById('loader').classList.remove('hidden');
  document.querySelector('.content').classList.add('loading');
}

function hideLoading() {
  document.getElementById('loader').classList.add('hidden');
  document.querySelector('.content').classList.remove('loading');
}

document.addEventListener('DOMContentLoaded', () => {
  const summarizeButton = document.getElementById('summarize');
  const summaryElement = document.getElementById('summary');
  const toggleViewButton = document.getElementById('toggleView');
  const toggleSummarizeButton = document.getElementById('toggleSummarize');
  const summarizerContainer = document.getElementById('summarizerContainer');
  const chatContainer = document.getElementById('chatContainer');
  const titleElement = document.querySelector('.header h1');

  // Toggle between summarizer and chat views
  toggleViewButton.addEventListener('click', () => {
    summarizerContainer.style.display = 'none';
    chatContainer.style.display = 'flex';
    titleElement.textContent = 'Chat';
  });

  toggleSummarizeButton.addEventListener('click', () => {
    summarizerContainer.style.display = 'flex';
    chatContainer.style.display = 'none';
    titleElement.textContent = 'Summarizer';
  });

  // Function to handle response from content.js
  const handleResponse = (response) => {
    if (response && response.url) {
      const encodedUrl = encodeURIComponent(response.url);
      // Make a GET request to your local server
      fetch(`http://localhost:8000/summarize?url=${encodedUrl}`, {
        method: "GET",
        mode: "cors",
      })
        .then((res) => {
          if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
          }
          return res.json();
        })
        .then((data) => {
          // Configure marked.js options for tighter spacing
          marked.setOptions({
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false
          });
          
          // Format the summary with less spacing
          const summaryMarkdown = data.Summary?.replace(/\n\n/g, '\n') || "No summary available.";
          summaryElement.innerHTML = marked.parse(summaryMarkdown);
        })
        .catch((error) => {
          console.error('Fetch error:', error);
          summaryElement.textContent = `Error: ${error.message}. Make sure the server is running and accessible.`;
        })
        .finally(() => {
          // Hide loader
          hideLoading();
          // (Optional) If you want the button gone permanently, do not re-enable or show it again
          // summarizeButton.disabled = false; // ← Removed/Commented out
        });
    } else {
      summaryElement.textContent = "Error: Couldn't retrieve the current Reddit URL.";
      hideLoading();
      // summarizeButton.disabled = false; // ← Removed/Commented out
    }
  };

  // When the Summarize button is clicked
  summarizeButton.addEventListener('click', () => {
    // Disable the button and hide it
    summarizeButton.disabled = true;
    summarizeButton.style.display = "none"; // ← Added line

    // Clear the current summary (optional)
    summaryElement.textContent = "";
    // Show the loader
    showLoading();

    // Get the current active tab
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const currentTab = tabs[0];
      // Quick check if it's a Reddit page
      if (!currentTab || !currentTab.url || !currentTab.url.includes('reddit.com')) {
        hideLoading();
        // If you want to show an error message but keep the button hidden, just set the text:
        summaryElement.textContent = "Error: This is not a Reddit page.";
        return;
      }

      // Send a message to the content script to get the URL
      chrome.tabs.sendMessage(currentTab.id, { action: "getUrl" }, (response) => {
        // Handle any errors
        if (chrome.runtime.lastError) {
          console.error('Runtime error:', chrome.runtime.lastError);
          // Attempt to inject content script if not present
          chrome.scripting.executeScript({
            target: { tabId: currentTab.id },
            files: ["content.js"]
          }, () => {
            if (chrome.runtime.lastError) {
              console.error('Failed to inject content script:', chrome.runtime.lastError);
              hideLoading();
              summaryElement.textContent = "Error: Could not access page content.";
            } else {
              // Retry after injecting content script
              chrome.tabs.sendMessage(currentTab.id, { action: "getUrl" }, handleResponse);
            }
          });
        } else {
          handleResponse(response);
        }
      });
    });
  });
});