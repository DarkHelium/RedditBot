// Utility functions to show/hide the loader
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
          // If your server returns the summary in Markdown, parse it with Marked
          // so you can have bold, italic, headings, etc.
          const summaryMarkdown = data.Summary || "No summary available.";
          summaryElement.innerHTML = marked.parse(summaryMarkdown);

          // If you wanted plain text only, you'd do:
          // summaryElement.textContent = summaryMarkdown;
        })
        .catch((error) => {
          console.error('Fetch error:', error);
          summaryElement.textContent = `Error: ${error.message}. Make sure the server is running and accessible.`;
        })
        .finally(() => {
          // Hide loader & re-enable button
          hideLoading();
          summarizeButton.disabled = false;
        });
    } else {
      summaryElement.textContent = "Error: Couldn't retrieve the current Reddit URL.";
      hideLoading();
      summarizeButton.disabled = false;
    }
  };

  // When the Summarize button is clicked
  summarizeButton.addEventListener('click', () => {
    // Disable the button to prevent multiple requests
    summarizeButton.disabled = true;
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
        summarizeButton.disabled = false;
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
              summarizeButton.disabled = false;
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
