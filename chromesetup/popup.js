document.addEventListener('DOMContentLoaded', function() {
  const summarizeButton = document.getElementById('summarize');
  const summaryElement = document.getElementById('summary');

  summarizeButton.addEventListener('click', function() {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      const currentTab = tabs[0];
      if (!currentTab.url.includes('reddit.com')) {
        summaryElement.textContent = "Error: This is not a Reddit page.";
        return;
      }

      chrome.tabs.sendMessage(currentTab.id, {action: "getUrl"}, function(response) {
        if (chrome.runtime.lastError) {
          console.error(chrome.runtime.lastError);
          // Inject content script if not already present
          chrome.scripting.executeScript({
            target: {tabId: currentTab.id},
            files: ['content.js']
          }, () => {
            if (chrome.runtime.lastError) {
              console.error('Failed to inject content script:', chrome.runtime.lastError);
              summaryElement.textContent = "Error: Failed to access page content.";
            } else {
              // Retry sending the message after injecting the content script
              chrome.tabs.sendMessage(currentTab.id, {action: "getUrl"}, handleResponse);
            }
          });
        } else {
          handleResponse(response);
        }
      });
    });
  });

  const handleResponse = function(response) {
    console.log('Received response:', response);
    if (response && response.url) {
      const encodedUrl = encodeURIComponent(response.url);
      console.log('Encoded URL:', encodedUrl);
      console.log('Fetching from:', `http://localhost:8000/summarize?url=${encodedUrl}`);
      
      fetch(`http://localhost:8000/summarize?url=${encodedUrl}`, {
        method: 'GET',
        mode: 'cors',
      })
      .then(response => {
        console.log('Fetch response status:', response.status);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log('Received data:', data);
        summaryElement.textContent = data.Summary;
      })
      .catch((error) => {
        console.error('Fetch error:', error);
        summaryElement.textContent = `Error: ${error.message}. Make sure the server is running and accessible.`;
      });
    } else {
      console.error('Invalid response:', response);
      summaryElement.textContent = "Error: Couldn't get the current URL.";
    }
  };
});
