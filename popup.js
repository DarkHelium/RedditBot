document.getElementById('summarize').addEventListener('click', function() {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      var url = tabs[0].url;
      fetch('http://localhost:5000/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({url: url}),
      })
      .then(response => response.json())
      .then(data => {
        document.getElementById('summary').textContent = data.summary;
      })
      .catch((error) => {
        console.error('Error:', error);
      });
    });
  });