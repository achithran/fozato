document.addEventListener('DOMContentLoaded', () => {
  const seoToggle = document.getElementById('seo-toggle');
  const youtubeUrlInput = document.getElementById('youtube-url');
  let selectedOption = '';

  // Load the toggle state from chrome storage
  chrome.storage.local.get(['popupEnabled'], (result) => {
    const isEnabled = result.popupEnabled !== false; // Default to true if not set
    seoToggle.checked = isEnabled;
    console.log(`Initial toggle state: ${isEnabled ? 'enabled' : 'disabled'}`);
  });
   // Listen for toggle changes
   seoToggle.addEventListener('change', () => {
    const isEnabled = seoToggle.checked;
    chrome.storage.local.set({ popupEnabled: isEnabled }, () => {
      console.log(`Popup state saved: ${isEnabled ? 'enabled' : 'disabled'}`);
    });
  });

  // Option click event handlers
  const videoToSeoOption = document.getElementById('video-to-seo');
  
  if (videoToSeoOption) {
    videoToSeoOption.addEventListener('click', () => {
      window.open('https://studio.youtube.com/', '_blank'); // Open in a new tab
      // For the same tab, use:
      // window.location.href = 'https://studio.youtube.com/';
    });
  }

  document.getElementById('url-to-seo').addEventListener('click', () => {
    selectedOption = 'url-to-seo';
    console.log('Selected: URL to SEO');
  });

  document.getElementById('keyword-to-seo').addEventListener('click', () => {
    selectedOption = 'keyword-to-seo';
    console.log('Selected: Keyword to SEO');
  });

  // Handle submit button
  document.getElementById('submit-btn').addEventListener('click', () => {
    const youtubeUrl = youtubeUrlInput.value;
    console.log('Submitted URL:', youtubeUrl);
    console.log('Selected Option for SEO:', selectedOption);

    // Add logic to handle the URL and SEO option here
  });

  // Handle close button
  document.getElementById('close-btn').addEventListener('click', () => {
    window.close();
  });
});
