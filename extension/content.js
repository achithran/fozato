let generatedTitle = null; // Global variable to store title from backend

window.addEventListener("load", () => {
  const intervalId = setInterval(() => {
    const uploadButton = document.querySelector('a[test-id="upload-icon-url"]');
    if (uploadButton) {
      clearInterval(intervalId); // Stop interval once button is found
      uploadButton.addEventListener("click", () => {
        console.log("Upload button clicked. Showing popup...");
        
        // Hide the dialog element with specific id, role, and class
        // const dialog = document.querySelector('ytcp-uploads-dialog');
        // console.log(dialog)
        // if (dialog) {
        //   dialog.style.display = 'none'; // Hides the dialog
        //   console.log("Dialog hidden.");
        // }
 

        // Show your custom popup (or call your popup function)
       // showEnterKeywordPopup();
         // Check the toggle state before showing the popup
           // Check the toggle state before showing the popup
           chrome.storage.local.get(['popupEnabled'], (result) => {
            const isPopupEnabled = result.popupEnabled !== false; // Default to true if not set
            console.log(`Popup enabled state: ${isPopupEnabled}`);
  
            if (isPopupEnabled) {
              console.log("Showing popup because toggle is ON.");
              showPopup(); // Call your popup function
            } else {
              console.log("Not showing popup because toggle is OFF.");
            }
          });
        
      });
    }
  }, 100); // Check every 100ms
});

// Wait until the DOM is fully loaded

  
// Function to show the popup
function showPopup() {
    const overlay = document.createElement("div");
    overlay.id = "youtube-popup-overlay";
    overlay.innerHTML = `
      <div id="youtube-popup">
        <div class="popup-header">
          <h2>Upload for Lightning SEO</h2>
          <div id="close-btn">&#10005;</div> <!-- Close icon -->
        </div>
        <div class="upload-area">
          <div id="upload-arrow">&#8593;</div>
          <p>Drag and drop video files to upload</p>
        </div>
        <div class="popup-border">
          <p>Your videos will be private until you publish them.</p>
        </div>
        <input type="file" id="file-upload" accept="video/*">
        <input type="text" placeholder="Enter Keyword">
        <button id="submit-btn">Submit File</button>
        <div id="upload-spinner" class="spinner" style="display: none;"></div> <!-- Spinner for upload -->
      </div>
    `;

    const style = document.createElement("style");
    style.textContent = `
      #youtube-popup-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: rgba(0, 0, 0, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 100000;
      }
      #youtube-popup {
        background-color: #333;
        color: #fff;
        padding: 20px;
        border-radius: 8px;
        width: 400px;
        text-align: center;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
      }
      .popup-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
      }
      .popup-header h2 {
        margin: 0;
        font-size: 18px;
        flex-grow: 1;
        text-align: center;
      }
      #close-btn {
        font-size: 24px;
        cursor: pointer;
        color: #fff;
      }
      .upload-area {
        margin-bottom: 20px;
        font-size: 14px;
      }
      #upload-arrow {
        font-size: 36px;
        margin-bottom: 10px;
        transition: transform 0.3s ease-in-out;
      }
      .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
      }
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
      .popup-border {
        border-top: 1px solid #ddd;
        border-bottom: 1px solid #ddd;
        margin: 20px 0;
        padding: 10px 0;
        font-size: 12px;
      }
      #file-upload {
        padding: 10px;
        width: 100%;
        margin-bottom: 20px;
        font-size: 14px;
      }
      #submit-btn {
        padding: 10px;
        width: 100%;
        background-color: #4CAF50;
        color: white;
        cursor: pointer;
        border: none;
        font-size: 16px;
      }
      #submit-btn:hover { opacity: 0.9; }
      #close-btn:hover { color: #f44336; }
      #seo-processing-popup {
    position: fixed;
    top: 20px;
    right: 20px;
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
    z-index: 100001; /* Ensure it's above the overlay */
  }
  #seo-processing-popup .spinner {
    width: 24px;
    height: 24px;
    margin-bottom: 10px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  #progress-container {
    width: 100%;
    background-color: #ddd;
    border-radius: 8px;
    margin-top: 10px;
    overflow: hidden;
    position: relative;
  }
  #progress-bar {
    height: 10px;
    background-color: #76c7c0;
    width: 0%;
  }
  #progress-percentage {
    position: absolute;
    top: -20px;
    right: 5px;
    font-size: 12px;
  }
    `;

    document.head.appendChild(style);
    document.body.appendChild(overlay);

    document.getElementById("close-btn").addEventListener("click", () => {
      overlay.remove();
    });

    document.getElementById("submit-btn").addEventListener("click", () => {
        const fileInput = document.getElementById("file-upload");
        const file = fileInput.files[0];
        if (file) {
            document.getElementById("upload-arrow").style.display = "none";
            document.getElementById("upload-spinner").style.display = "block";
            showSEOProcessingPopup(); // Show the SEO processing popup
            uploadFile(file); // Upload file and get title
        }
        else
        {
            alert("Please select a video file to upload.");
        }
});
}

// Function to upload file and get the generated title from the server
function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "http://127.0.0.1:8000/sampleapp/youtube-url/");

  // Define response outside of the async blocks
  let response = null;

 


  xhr.onload = () => {
      if (xhr.status === 200) {
        
          
          try {
              response = JSON.parse(xhr.responseText);  // Safely parse the response
              print("response:",response)
              generatedTitle = response.title; // Store the title globally
              generatedDescription = response.description; // Store the description globally
              console.log("GENERATED TITLE:", generatedTitle);
              console.log("GENERATED DESCRIPTION:", generatedDescription);
              alert(response.message); // Display success message
              hideSEOProcessingPopup();
              hideshowPopup();
              
              
              // Use response inside setTimeout
              setTimeout(() => {
                  waitForYouTubeStudioPopup(response);  // Pass response to the function
              }, 1000); // Adjust delay as needed
          } catch (e) {
              // Handle JSON parsing errors
              alert("Error parsing response: " + e.message);
              hideSEOProcessingPopup();
             
          }
      } else {
          // If status is not 200, safely assign a response object
          response = { error: "Transcription Failed" };
          alert(response.error || "Transcription Failed");
          hideSEOProcessingPopup();
           // Show the "Enter Keyword" popup
           showEnterKeywordPopup();
           const dialog = document.querySelector('ytcp-uploads-dialog');
           console.log(dialog)
           if (dialog) {
             dialog.style.display = 'none'; // Hides the dialog
             console.log("Dialog hidden.");
           }
           
          
      }
     
  };

  xhr.onerror = () => {
      
      alert("A network error occurred. Please try again.");
      hideSEOProcessingPopup();
      // Hide the YouTube Studio popup
      document.getElementById('youtube-popup').style.display = 'none';
  };

  xhr.send(formData);
}


// Function to show the popup with the textarea
function showEnterKeywordPopup() {
  console.log("Creating popup...");
  const overlay = document.createElement("div");
  overlay.id = "youtube-popup-Keyword";
  // Create the popup structure
  overlay.innerHTML = `
  <div id="div-popup-keyword">
    <div class="popup-header">
      <h2>We'll Generate SEO Keywords for You</h2>
      <div id="close-btn">&#10005;</div> <!-- Close icon -->
    </div>
    <div class="upload-area">
     
      <pEnter your keyword below</p>
    </div>
    <div class="popup-border">
      <textarea placeholder="Enter keyword" id="keyword-textarea"></textarea>
    </div>
   
    <button id="submitbtn">Submit</button>
    <div id="upload-spinner-keyword" class="spinner" style="display: none;"></div> <!-- Spinner for upload -->
  </div>
`;

const style = document.createElement("style");
style.textContent = `
  #youtube-popup-Keyword {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: rgba(0, 0, 0, 0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100000;
  }
  #div-popup-keyword{
    background-color: #333;
    color: #fff;
    padding: 20px;
    border-radius: 8px;
    width: 400px;
    text-align: center;
    box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
  }
  .popup-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }
  .popup-header h2 {
    margin: 0;
    font-size: 18px;
    flex-grow: 1;
    text-align: center;
  }
  #close-btn {
    font-size: 24px;
    cursor: pointer;
    color: #fff;
  }
  .upload-area {
    margin-bottom: 20px;
    font-size: 14px;
  }

  .spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
  }
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  .popup-border {
    border-top: 1px solid #ddd;
    border-bottom: 1px solid #ddd;
    margin: 20px 0;
    padding: 10px 0;
    font-size: 12px;
  }
  
  #submitbtn {
    padding: 10px;
    width: 100%;
    background-color: #4CAF50;
    color: white;
    cursor: pointer;
    border: none;
    font-size: 16px;
  }
  #submitbtn:hover { opacity: 0.9; }
  #close-btn:hover { color: #f44336; }
  #seo-processing-popup {
position: fixed;
top: 20px;
right: 20px;
background-color: #4CAF50;
color: white;
padding: 10px 20px;
border-radius: 8px;
display: flex;
flex-direction: column;
align-items: center;
box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
z-index: 100001; /* Ensure it's above the overlay */
}
#seo-processing-popup .spinner {
width: 24px;
height: 24px;
margin-bottom: 10px;
border: 4px solid #f3f3f3;
border-top: 4px solid #3498db;
border-radius: 50%;
animation: spin 1s linear infinite;
}
@keyframes spin {
0% { transform: rotate(0deg); }
100% { transform: rotate(360deg); }
}
#progress-container {
width: 100%;
background-color: #ddd;
border-radius: 8px;
margin-top: 10px;
overflow: hidden;
position: relative;
}
#progress-bar {
height: 10px;
background-color: #76c7c0;
width: 0%;
}
#progress-percentage {
position: absolute;
top: -20px;
right: 5px;
font-size: 12px;
}
textarea{
 width: 100%;
  height: 150px;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 8px;
  font-size: 16px;
  resize: vertical;
  box-sizing: border-box;
  background-color: #f9f9f9;
  transition: border-color 0.3s ease;}
`;

document.head.appendChild(style);
document.body.appendChild(overlay);

document.getElementById("close-btn").addEventListener("click", () => {

  overlay.remove();

 
});

document.getElementById("submitbtn").addEventListener("click", () => {
    const keyword = document.getElementById("keyword-textarea");
    const keywordvalue = keyword.value;
    if (keywordvalue) {
       console.log("keyword is:",keywordvalue)
        document.getElementById("upload-spinner-keyword").style.display = "block";
        showSEOProcessingPopup(); // Show the SEO processing popup
        uploadkeyword(keywordvalue); // Upload file and get title
    }
    else
    {
        alert("Please Enter a keyword");
    }})
}

// Function to upload file and get the generated title from the server
function uploadkeyword(keyword) {
  const formData = new FormData();
  formData.append("text", keyword);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "http://127.0.0.1:9000/extractionapp/extraction_from_text_api/");

  // Define response outside of the async blocks
  let response = null;

 


  xhr.onload = () => {
      if (xhr.status === 200) {
        
          
          try {
              response = JSON.parse(xhr.responseText);  // Safely parse the response
              alert("Got response successfully")
              hideSEOProcessingPopup();
              hideshowEnterKeywordPopup();
              const dialog = document.querySelector('ytcp-uploads-dialog');
              console.log(dialog)
              if (dialog) {
                dialog.style.display = 'block'; // Hides the dialog
                console.log("Dialog hidden.");
              }
          } catch (e) {
              // Handle JSON parsing errors
              alert("Error parsing response: " + e.message);
              hideSEOProcessingPopup();
             
          }
      } else {
          // If status is not 200, safely assign a response object
          response = { error: "function cant called" };
          alert(response.error || "function cant called");
          hideSEOProcessingPopup();
           // Show the "Enter Keyword" popup
           //showEnterKeywordPopup();
           
          //
      }
     
  };

  xhr.onerror = () => {
      
      alert("A network error occurred. Please try again.");
      hideSEOProcessingPopup();
      // Hide the YouTube Studio popup
      document.getElementById('div-popup-keyword').style.display = 'none';
  };

  xhr.send(formData);
}


// Function to wait for YouTube Studio popup and set the title

function waitForYouTubeStudioPopup(response) {
  console.log("Received response in waitForYouTubeStudioPopup:", response);
    try {
        const popupObserver = new MutationObserver(() => {
            const titleTextbox = document.querySelector('div[id="child-input"]');
            const descriptionTextbox = document.querySelector('div[aria-label="Tell viewers about your video (type @ to mention a channel)"]');
            // const thumbnailContainer = document.querySelector('div[class="autogenerated-thumbnails"]');
            if (titleTextbox && generatedTitle) {
                titleTextbox.textContent = generatedTitle;
                console.log("Title set in YouTube Studio textbox:", generatedTitle);
                
            }
            if (descriptionTextbox && generatedDescription) {
                descriptionTextbox.textContent = generatedDescription;
                 // Simulate input events to mimic manual entry
                descriptionTextbox.dispatchEvent(new Event('input', { bubbles: true }));
                descriptionTextbox.dispatchEvent(new Event('change', { bubbles: true }));
                console.log("Description set in YouTube Studio textbox:", generatedDescription);
            }

            // Add thumbnails to thumbnail container if detected
          //   if (thumbnailContainer && generatedThumbnails.length > 0) {
          //     generatedThumbnails.forEach((thumbnailUrl) => {
          //         const thumbnailImg = document.createElement("img");
          //         thumbnailImg.src = thumbnailUrl;
          //         thumbnailImg.alt = "Generated Thumbnail";
          //         thumbnailImg.style.width = "100px"; // Adjust size as needed
          //         thumbnailImg.style.margin = "5px";
          //         thumbnailContainer.appendChild(thumbnailImg);
          //     });
          //     console.log("Thumbnails added to YouTube Studio.");
          // }
             // Disconnect observer if both fields are set
             if (titleTextbox && descriptionTextbox) {
                popupObserver.disconnect();
            }
        });
        popupObserver.observe(document.body, { childList: true, subtree: true });
    } catch (error) {
        console.error("Error setting title in YouTube Studio popup:", error);
    }
}


// Function to show SEO processing popup with a progress bar at the top-right corner
function showSEOProcessingPopup() {
    const seoPopup = document.createElement("div");
    seoPopup.id = "seo-processing-popup";
    seoPopup.innerHTML = `
        <div class="spinner"></div>
        <span>SEO Processing</span>
        <div id="progress-container">
            <div id="progress-bar"></div>
            <span id="progress-percentage">0%</span>
        </div>
    `;
    document.body.appendChild(seoPopup);

    console.log("SEO Processing popup shown"); // Debugging statement
}

// Function to hide SEO processing popup
function hideSEOProcessingPopup() {
    const seoPopup = document.getElementById("seo-processing-popup");
    if (seoPopup) {
        seoPopup.remove();
        console.log("SEO Processing popup hidden"); // Debugging statement
    }
}
function hideshowEnterKeywordPopup(){
  const seoPopup = document.getElementById("youtube-popup-Keyword");
    if (seoPopup) {
        seoPopup.remove();
        console.log("SEO Processing popup hidden"); // Debugging statement
    }
}
function hideshowPopup(){
  const seoPopup = document.getElementById("youtube-popup-overlay");
    if (seoPopup) {
        seoPopup.remove();
        console.log("SEO Processing popup hidden"); // Debugging statement
    }
}
function minimalPopupTest() {
  const overlay = document.createElement("div");
  overlay.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background-color: rgba(0, 0, 0, 0.8); display: flex; align-items: center;
    justify-content: center; z-index: 99999999; pointer-events: auto;
  `;
  
  const textarea = document.createElement("textarea");
  textarea.style.cssText = `
    width: 90%; height: 150px; padding: 10px; background-color: #555; 
    color: white; border: 1px solid #444; border-radius: 4px; font-size: 14px;
    resize: none; pointer-events: auto;
  `;
  overlay.appendChild(textarea);
  document.body.appendChild(overlay);

  textarea.focus();

  overlay.addEventListener("click", () => overlay.remove());
}

//minimalPopupTest();
//showEnterKeywordPopup()


// Listen for messages from the popup
chrome.runtime.onMessage.addListener((message) => {
  if (message.action === "showPopup") {
    const keyword = message.keyword;
    createCustomPopup(keyword);
  }
});

// Function to create the popup in YouTube Studio
function createCustomPopup(keyword) {
  const overlay = document.createElement("div");
  overlay.id = "youtube-popup-overlay1";

  overlay.innerHTML = `
    <div id="youtube-popup1">
      <div class="popup-header">
        <h2>Keyword Received</h2>
        <span id="close-popup" style="cursor: pointer;">&#10005;</span>
      </div>
      <p>Your keyword: <strong>${keyword}</strong></p>
    </div>
  `;

  const style = document.createElement("style");
  style.textContent = `
    #youtube-popup-overlay1 {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(0, 0, 0, 0.8);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 9999;
    }
    #youtube-popup1 {
      background: white;
      padding: 20px;
      border-radius: 8px;
      text-align: center;
    }
    .popup-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
  `;

  document.body.appendChild(overlay);
  document.head.appendChild(style);

  document.getElementById("close-popup").addEventListener("click", () => {
    overlay.remove();
  });
}




