document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded and parsed. app.js executing.");

    // DOM Element References
    const videoForm = document.getElementById('video-form');
    const videoUrlInput = document.getElementById('video-url');
    const searchIntentInput = document.getElementById('search-intent');
    const processButton = document.getElementById('process-button');
    
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessageDiv = document.getElementById('error-message');
    
    const outputSection = document.getElementById('output-section');
    const summaryContentDiv = document.getElementById('summary-content');
    const tocListUl = document.getElementById('toc-list');
    const fullArticleTextDiv = document.getElementById('full-article-text');
    const copyToClipboardButton = document.getElementById('copy-to-clipboard');

    const videoPlayerSection = document.getElementById('video-player-section');
    const youtubePlayerContainer = document.getElementById('youtube-player-container'); // This IS the element
    const videoPlayerTitle = document.getElementById('video-player-title');

    const sidebarHistory = document.getElementById('sidebar-history');
    const historyListUl = document.getElementById('history-list');
    const noHistoryP = document.getElementById('no-history'); 

    // Configuration
    const API_BASE_URL = 'http://localhost:8000'; 
    const MAX_HISTORY_ITEMS = 5; 

    // State
    const processedArticlesStore = {}; 
    let ytPlayer; 
    let currentVideoId = null; 
    let isYouTubeApiReady = false; 
    let pendingVideoToLoad = null; 

    // --- YouTube IFrame API Setup ---
    window.onYouTubeIframeAPIReady = function() {
        console.log("YouTube IFrame API is ready.");
        isYouTubeApiReady = true;
        if (pendingVideoToLoad && pendingVideoToLoad.videoId) {
            createYouTubePlayer(pendingVideoToLoad.videoId, pendingVideoToLoad.videoTitle);
            pendingVideoToLoad = null; 
        }
    };

    function createYouTubePlayer(videoId, videoTitle) {
        currentVideoId = videoId; 
        videoPlayerSection.style.display = 'block';
        videoPlayerTitle.textContent = videoTitle || "Video Player";
        
        youtubePlayerContainer.innerHTML = ''; 
        console.log(`Creating new YT.Player for videoId: ${videoId}`);
        
        ytPlayer = new YT.Player(youtubePlayerContainer, { 
            height: '100%', 
            width: '100%',  
            videoId: videoId,
            playerVars: {
                'playsinline': 1, 
                'autoplay': 0,    
                'controls': 1     
            },
            events: {
                'onReady': (event) => {
                    console.log("Player ready for video:", videoId, event.target);
                    // Player is ready. Now seeking should be more reliable.
                },
                'onError': (event) => {
                    console.error("YouTube Player Error Code:", event.data);
                    let errorReason = "Unknown player error.";
                    switch(event.data) {
                        case 2: errorReason = "Invalid parameter value. Check video ID."; break;
                        case 5: errorReason = "HTML5 player error."; break;
                        case 100: errorReason = "Video not found or removed."; break;
                        case 101: case 150: errorReason = "Embedding disabled by the video owner."; break;
                    }
                    displayError(`YouTube Player Error: ${errorReason} (Code ${event.data})`);
                }
            }
        });
    }

    function loadOrUpdateYouTubePlayer(videoId, videoTitle) { 
        if (!videoId) {
            console.log("No videoId provided, hiding player.");
            videoPlayerSection.style.display = 'none';
            if(ytPlayer && typeof ytPlayer.destroy === 'function') {
                try { ytPlayer.destroy(); } catch(e) { console.warn("Error destroying player:", e); }
                ytPlayer = null;
            }
            currentVideoId = null;
            return;
        }

        if (isYouTubeApiReady) {
            if (ytPlayer && typeof ytPlayer.loadVideoById === 'function') {
                console.log(`Player exists. Loading new videoId via loadVideoById: ${videoId}`);
                videoPlayerSection.style.display = 'block'; // Ensure section is visible
                videoPlayerTitle.textContent = videoTitle || "Video Player";
                ytPlayer.loadVideoById(videoId); // This changes the video in the existing player
                currentVideoId = videoId; // Update currentVideoId
            } else { // No player instance yet, or it's not fully functional
                createYouTubePlayer(videoId, videoTitle);
            }
        } else {
            console.log("YouTube API not ready yet. Queuing video load:", videoId);
            pendingVideoToLoad = { videoId, videoTitle }; 
        }
    }
    
    // --- Timestamp Link Click Handler ---
    function handleTimestampLinkClick(event) {
        const targetLink = event.target.closest('a.youtube-timestamp-link'); 
        if (targetLink) {
            event.preventDefault(); // Always prevent default for these links if we intend to handle them
            
            const videoIdFromLink = targetLink.dataset.videoId;
            const timeParam = targetLink.dataset.timestamp; // This should be just seconds now
            const timestampInSeconds = parseInt(timeParam);
    
            console.log(`Timestamp link clicked: videoId=${videoIdFromLink}, time=${timestampInSeconds}s`);
    
            if (isNaN(timestampInSeconds)) {
                console.error("Invalid timestamp in link:", timeParam);
                return;
            }
    
            if (ytPlayer && typeof ytPlayer.seekTo === 'function') {
                if (currentVideoId === videoIdFromLink) {
                    console.log(`Seeking current player to: ${timestampInSeconds}`);
                    // Ensure player is in a state where seeking is allowed (e.g., playing, paused, cued)
                    // Sometimes, calling playVideo() right before or after seekTo() helps.
                    ytPlayer.playVideo(); // Attempt to start playing first (might cue it if not loaded)
                    
                    // Give a tiny moment for playVideo to register if the video wasn't playing
                    // This is a common trick for YouTube API.
                    setTimeout(() => {
                        ytPlayer.seekTo(timestampInSeconds, true);
                        // Ensure it continues playing if playVideo() just cued it.
                        if (ytPlayer.getPlayerState() !== YT.PlayerState.PLAYING) {
                            ytPlayer.playVideo();
                        }
                    }, 150); // 150ms delay, adjust if needed
    
                    videoPlayerSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                } else {
                    console.warn(`Clicked timestamp for video ${videoIdFromLink}, but player has ${currentVideoId} loaded. Consider loading new video.`);
                    // Future enhancement: If videoIdFromLink is different, load it into the player:
                    // loadOrUpdateYouTubePlayer(videoIdFromLink, "Video from Link"); // Need a title
                    // And then, once that new video is ready, seek to the timestamp. This is more complex.
                    // For now, we can just log or do nothing if the video ID doesn't match.
                }
            } else {
                console.error("YouTube player (ytPlayer) is not available or seekTo is not a function.");
            }
        }
    }
    fullArticleTextDiv.addEventListener('click', handleTimestampLinkClick);
    


    // --- Helper Functions ---
    function displayError(message) {
        errorMessageDiv.textContent = message;
        errorMessageDiv.style.display = 'block';
        outputSection.style.display = 'none'; 
        videoPlayerSection.style.display = 'none';
        copyToClipboardButton.style.display = 'none';
    }

    function clearError() {
        errorMessageDiv.textContent = '';
        errorMessageDiv.style.display = 'none';
    }

    function showLoading(isLoading) {
        if (isLoading) {
            loadingIndicator.style.display = 'block';
            processButton.disabled = true; 
            processButton.textContent = 'Processing...';
        } else {
            loadingIndicator.style.display = 'none';
            processButton.disabled = false; 
            processButton.textContent = 'Process Video';
        }
    }

    // --- Output Rendering Function ---
    function renderOutput(data) {
        clearError(); 
        outputSection.style.display = 'block'; 

        console.log("Data for player in renderOutput:", data.video_data); // DEBUG line
        if (data.video_data && data.video_data.video_id) {
            loadOrUpdateYouTubePlayer(data.video_data.video_id, data.video_data.title); 
        } else {
            loadOrUpdateYouTubePlayer(null, null); 
        }

        if (data.llm_article_data && data.llm_article_data.summary) {
            summaryContentDiv.innerHTML = ''; 
            const summaryParagraph = document.createElement('p');
            summaryParagraph.innerHTML = data.llm_article_data.summary.replace(/\n- /g, '<br>- ').replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');
            summaryContentDiv.appendChild(summaryParagraph);
        } else {
            summaryContentDiv.innerHTML = '<p>No summary available.</p>';
        }

        tocListUl.innerHTML = ''; 
        if (data.llm_article_data && data.llm_article_data.table_of_contents && data.llm_article_data.table_of_contents.length > 0) {
            data.llm_article_data.table_of_contents.forEach((item, index) => {
                const listItem = document.createElement('li');
                const link = document.createElement('a');
                const sectionId = `section-${encodeURIComponent(item.replace(/\s+/g, '-').toLowerCase())}`;
                link.href = `#${sectionId}`;
                link.textContent = item;
                link.className = "hover:underline text-blue-600 hover:text-blue-800"; 
                listItem.appendChild(link);
                tocListUl.appendChild(listItem);
            });
        } else {
            tocListUl.innerHTML = '<li>No table of contents available.</li>';
        }

        fullArticleTextDiv.innerHTML = ''; 
        if (data.llm_article_data && data.llm_article_data.article_sections && data.llm_article_data.article_sections.length > 0) {
            data.llm_article_data.article_sections.forEach((section, index) => {
                const sectionContainer = document.createElement('div');
                sectionContainer.className = 'mb-6'; 
                const sectionId = `section-${encodeURIComponent(section.heading?.replace(/\s+/g, '-').toLowerCase() || `untitled-section-${index}`)}`;
                sectionContainer.id = sectionId; 

                const heading = document.createElement('h4'); 
                heading.className = 'text-lg font-semibold mb-2 text-gray-800'; 
                heading.textContent = section.heading || 'Untitled Section';
                sectionContainer.appendChild(heading);

                const contentDiv = document.createElement('div'); 
                let rawContentFromLLM = section.content || '';
                let structuredHtmlContent = rawContentFromLLM
                    .split('\n\n') 
                    .map(paragraph => `<p class="mb-2">${paragraph.replace(/\n/g, '<br>')}</p>`) 
                    .join('');

                const timestampRegex = /\[(?:ts:)?([\d:.]+s?)\]/g;
                finalContentHtml = structuredHtmlContent.replace(timestampRegex, (match, timeValue) => {
                    let seconds = 0;
                    const cleanedTimeValue = timeValue.replace('s','');
                    if (cleanedTimeValue.includes(':')) { 
                        const parts = cleanedTimeValue.split(':').map(Number);
                        if (parts.length === 3) { 
                            seconds = parts[0] * 3600 + parts[1] * 60 + parts[2];
                        } else if (parts.length === 2) { 
                            seconds = parts[0] * 60 + parts[1];
                        }
                    } else { 
                        seconds = parseFloat(cleanedTimeValue);
                    }

                    if (!isNaN(seconds) && data.video_data && data.video_data.video_id) {
                        return `<a href="#" 
                                   data-video-id="${data.video_data.video_id}" 
                                   data-timestamp="${Math.floor(seconds)}" 
                                   class="youtube-timestamp-link text-blue-600 hover:text-blue-800 underline" 
                                   title="Play from ${timeValue}">${match}</a>`;
                    }
                    return match; 
                });
                
                contentDiv.innerHTML = finalContentHtml; 
                
                if (section.relevant_to_search_intent) {
                    sectionContainer.classList.add('bg-yellow-100', 'border-l-4', 'border-yellow-500', 'p-3', 'rounded-r-md'); 
                }

                sectionContainer.appendChild(contentDiv);
                fullArticleTextDiv.appendChild(sectionContainer);
            });
        } else {
            fullArticleTextDiv.innerHTML = '<p>No article content available.</p>';
        }
        
        if (data.llm_article_data && (data.llm_article_data.summary || data.llm_article_data.article_sections?.length > 0)) {
            copyToClipboardButton.style.display = 'inline-block';
        } else {
            copyToClipboardButton.style.display = 'none';
        }
    }

    // --- Form Submission Handler ---
    if (videoForm) {
        videoForm.addEventListener('submit', async (event) => {
            event.preventDefault(); 
            clearError();
            showLoading(true);
            outputSection.style.display = 'none'; 
            videoPlayerSection.style.display = 'none'; 
            copyToClipboardButton.style.display = 'none';

            const videoUrl = videoUrlInput.value.trim();
            const searchIntentValue = searchIntentInput.value.trim();

            if (!videoUrl) {
                displayError('Please enter a YouTube video URL.');
                showLoading(false);
                return;
            }
            try {
                new URL(videoUrl);
                if (!videoUrl.includes('youtube.com/') && !videoUrl.includes('youtu.be/')) {
                    displayError('URL must be a valid YouTube video link (e.g., includes youtube.com or youtu.be).');
                    showLoading(false);
                    return; 
                }
            } catch (error) {
                if (error instanceof TypeError) { 
                    displayError('Please enter a valid URL format (e.g., https://example.com).');
                } else {
                    displayError('An unexpected error occurred during URL validation.');
                    console.error("Unexpected validation error:", error);
                }
                showLoading(false);
                return;
            }
            
            try {
                const requestBody = { video_url: videoUrl };
                if (searchIntentValue) { requestBody.search_intent = searchIntentValue; }

                const response = await fetch(`${API_BASE_URL}/process-video`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify(requestBody)
                });
                
                if (!response.ok) {
                    let errorData;
                    try { errorData = await response.json(); } 
                    catch (e) { errorData = { detail: `HTTP error! Status: ${response.status} - ${response.statusText}` }; }
                    console.error('API Error Response:', errorData);
                    displayError(errorData.detail || `An API error occurred: ${response.status}`);
                    showLoading(false); 
                    return;
                }

                const data = await response.json();
                console.log('API Success Response:', data);
                showLoading(false); 

                if (data.message.toLowerCase().includes("failed") || (data.video_data && data.video_data.error)) {
                    displayError(data.message || (data.video_data && data.video_data.error) || "Processing failed, please check the video URL or try again.");
                } else if (data.llm_article_data && data.llm_article_data.summary.toLowerCase().includes("error:")) {
                    displayError(`LLM Processing Error: ${data.llm_article_data.summary}`);
                } else if (data.llm_article_data) { 
                    renderOutput(data); 
                } else {
                    displayError("Received data from backend, but no article content was generated.");
                }

            } catch (error) {
                console.error('Fetch API Call Error:', error);
                showLoading(false);
                displayError('Failed to connect to the server or network error. Please try again.');
            }
        });
    } else {
        console.error("Video form not found!");
    }

    // --- Copy to Clipboard Functionality ---
    if(copyToClipboardButton) {
        copyToClipboardButton.addEventListener('click', () => {
            let textToCopy = "";
            const articleTitle = document.getElementById('video-player-title')?.textContent || "Video Article";
            textToCopy += articleTitle + "\n\n";

            if (summaryContentDiv.innerText.trim() !== "" && !summaryContentDiv.innerText.toLowerCase().includes("no summary")) {
                textToCopy += "Summary:\n" + summaryContentDiv.innerText.trim() + "\n\n";
            }
            if (tocListUl.children.length > 0 && !tocListUl.textContent.toLowerCase().includes("no table of contents")) {
                textToCopy += "Table of Contents:\n";
                Array.from(tocListUl.children).forEach(li => {
                    textToCopy += "- " + li.textContent.trim() + "\n";
                });
                textToCopy += "\n";
            }
            if (fullArticleTextDiv.innerText.trim() !== "" && !fullArticleTextDiv.innerText.toLowerCase().includes("no article content")) {
                textToCopy += "Full Article:\n";
                const sections = fullArticleTextDiv.querySelectorAll('div.mb-6'); 
                sections.forEach(sectionDiv => {
                    const heading = sectionDiv.querySelector('h4');
                    const content = sectionDiv.querySelector('div'); 
                    if (heading) {
                        textToCopy += "\n## " + heading.innerText.trim() + "\n"; 
                    }
                    if (content) {
                        textToCopy += content.innerText.trim() + "\n\n"; 
                    }
                });
            }

            if (textToCopy) {
                navigator.clipboard.writeText(textToCopy.trim()).then(() => {
                    alert('Article content copied to clipboard!');
                }).catch(err => {
                    console.error('Failed to copy text: ', err);
                    alert('Failed to copy text. See console for details.');
                });
            } else {
                alert('No content available to copy.');
            }
        });
    }
});