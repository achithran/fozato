chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === "complete" && tab.url) {
      // Check if the URL is YouTube Studio
      if (tab.url.includes("https://studio.youtube.com")) {
        // Now, the popup is handled by content.js, so we don't need to call anything here
      }
    }
  });
  