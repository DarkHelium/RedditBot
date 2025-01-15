document.addEventListener('DOMContentLoaded', () => {
  const summaryDiv = document.getElementById('summary');
  const summarizeButton = document.getElementById('summarize');
  const loader = document.getElementById('loader');
  const toggleViewButton = document.getElementById('toggleView');
  const toggleSummarizeButton = document.getElementById('toggleSummarize');
  const summarizerContainer = document.getElementById('summarizerContainer');
  const chatContainer = document.getElementById('chatContainer');

  // Function to show/hide loader
  const setLoading = (loading) => {
    loader.classList.toggle('hidden', !loading);
  };

  // Function to summarize the current Reddit post
  const summarizePost = async () => {
    try {
      setLoading(true);
      
      // Get the current tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      // Send message to content.js to get URL
      const response = await chrome.tabs.sendMessage(tab.id, { action: "getUrl" });
      
      if (response && response.url) {
        // Encode the URL
        const encodedUrl = encodeURIComponent(response.url);
        
        // Make request to our Python server
        const summaryResponse = await fetch(`http://localhost:8000/summarize?url=${encodedUrl}`);
        const data = await summaryResponse.json();
        
        // Update the summary div with the response
        if (data.Summary) {
          summaryDiv.innerHTML = marked.parse(data.Summary);
          summarizeButton.style.display = 'none';  // Hide the button after successful summary
        } else {
          summaryDiv.textContent = 'No summary available.';
        }
      }
    } catch (error) {
      console.error('Error:', error);
      summaryDiv.textContent = 'Error getting summary. Please try again.';
    } finally {
      setLoading(false);
    }
  };

  // Add click event listener to the summarize button
  summarizeButton.addEventListener('click', summarizePost);

  // Toggle between summarizer and chat views
  toggleViewButton.addEventListener('click', () => {
    const isSummarizerVisible = summarizerContainer.style.display !== 'none';
    summarizerContainer.style.display = isSummarizerVisible ? 'none' : 'flex';
    chatContainer.style.display = isSummarizerVisible ? 'flex' : 'none';
  });

  // Toggle back to summarizer view
  toggleSummarizeButton.addEventListener('click', () => {
    summarizerContainer.style.display = 'flex';
    chatContainer.style.display = 'none';
  });

  // Show chat container by default and hide summarizer
  summarizerContainer.style.display = 'none';
  chatContainer.style.display = 'flex';
});
