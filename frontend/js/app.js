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
    const youtubePlayerContainer = document.getElementById('youtube-player-container');
    const videoPlayerTitle = document.getElementById('video-player-title');

    const sidebarHistory = document.getElementById('sidebar-history');
    const historyListUl = document.getElementById('history-list');
    const noHistoryP = document.getElementById('no-history');

    // Configuration
    const API_BASE_URL = 'http://localhost:8000';
    //const API_BASE_URL = 'https://streamzine-service-122411196232.us-west1.run.app';
    const MAX_HISTORY_ITEMS = 5;

    // State
    let ytPlayer;
    let currentVideoId = null;
    let isYouTubeApiReady = false;
    let pendingVideoToLoad = null;
    let history = [];
    let currentActiveHistoryItem = null;

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
        // ... (no changes here) ...
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
        // ... (no changes here) ...
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
            if (ytPlayer && typeof ytPlayer.loadVideoById === 'function' && currentVideoId === videoId) {
                videoPlayerSection.style.display = 'block';
                videoPlayerTitle.textContent = videoTitle || "Video Player";
            } else if (ytPlayer && typeof ytPlayer.loadVideoById === 'function') {
                console.log(`Player exists. Loading new videoId via loadVideoById: ${videoId}`);
                videoPlayerSection.style.display = 'block';
                videoPlayerTitle.textContent = videoTitle || "Video Player";
                ytPlayer.loadVideoById(videoId);
                currentVideoId = videoId;
            } else {
                createYouTubePlayer(videoId, videoTitle);
            }
        } else {
            console.log("YouTube API not ready yet. Queuing video load:", videoId);
            pendingVideoToLoad = { videoId, videoTitle };
        }
    }

    // --- Timestamp Link Click Handler ---
    function handleTimestampLinkClick(event) {
        // ... (no changes here) ...
        const targetLink = event.target.closest('a.youtube-timestamp-link');
        if (targetLink) {
            event.preventDefault();

            const videoIdFromLink = targetLink.dataset.videoId;
            const timeParam = targetLink.dataset.timestamp;
            const timestampInSeconds = parseInt(timeParam);

            console.log(`Timestamp link clicked: videoId=${videoIdFromLink}, time=${timestampInSeconds}s`);

            if (isNaN(timestampInSeconds)) {
                console.error("Invalid timestamp in link:", timeParam);
                return;
            }

            if (ytPlayer && typeof ytPlayer.seekTo === 'function') {
                if (currentVideoId === videoIdFromLink) {
                    console.log(`Seeking current player to: ${timestampInSeconds}`);
                    ytPlayer.playVideo();
                    setTimeout(() => {
                        ytPlayer.seekTo(timestampInSeconds, true);
                        if (ytPlayer.getPlayerState() !== YT.PlayerState.PLAYING) {
                            ytPlayer.playVideo();
                        }
                    }, 150);

                    videoPlayerSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                } else {
                    console.warn(`Clicked timestamp for video ${videoIdFromLink}, but player has ${currentVideoId} loaded. Consider loading new video.`);
                }
            } else {
                console.error("YouTube player (ytPlayer) is not available or seekTo is not a function.");
            }
        }
    }
    fullArticleTextDiv.addEventListener('click', handleTimestampLinkClick);

    // --- Helper Functions ---
    function displayError(message) {
        // ... (no changes here) ...
        errorMessageDiv.textContent = message;
        errorMessageDiv.style.display = 'block';
        outputSection.style.display = 'none';
        videoPlayerSection.style.display = 'none';
        copyToClipboardButton.style.display = 'none';
    }

    function clearError() {
        // ... (no changes here) ...
        errorMessageDiv.textContent = '';
        errorMessageDiv.style.display = 'none';
    }

    function showLoading(isLoading) {
        // ... (no changes here) ...
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

    function updateHistory(articleData) {
        // ... (no changes here) ...
        if (!articleData || !articleData.video_data || !articleData.video_data.video_id) {
            console.warn("Attempted to add invalid article data to history.", articleData);
            return;
        }
        const existingIndex = history.findIndex(item => item.video_data.video_id === articleData.video_data.video_id);
        if (existingIndex > -1) {
            history.splice(existingIndex, 1);
        }
        history.unshift(articleData);
        if (history.length > MAX_HISTORY_ITEMS) {
            history.pop();
        }
        renderHistoryList();
    }

    function renderHistoryList() {
        // ... (no changes here) ...
        historyListUl.innerHTML = '';
        if (history.length === 0) {
            noHistoryP.style.display = 'block';
            historyListUl.style.display = 'none';
        } else {
            noHistoryP.style.display = 'none';
            historyListUl.style.display = 'block';
            history.forEach((articleData, index) => {
                const listItem = document.createElement('li');
                listItem.className = 'p-2 hover:bg-gray-200 rounded cursor-pointer text-sm text-gray-600 truncate';
                listItem.textContent = articleData.video_data.title || 'Untitled Video';
                listItem.title = articleData.video_data.title || 'Untitled Video';
                listItem.dataset.historyIndex = index;
                listItem.addEventListener('click', () => {
                    loadArticleFromHistory(index);
                });
                historyListUl.appendChild(listItem);
            });
        }
    }

    function setActiveHistoryItem(articleVideoId) {
        // ... (no changes here) ...
        const listItems = historyListUl.querySelectorAll('li');
        let itemToActivate = null;
        listItems.forEach(item => {
            item.classList.remove('active-history-item');
            const itemIndex = parseInt(item.dataset.historyIndex);
            if (history[itemIndex] && history[itemIndex].video_data.video_id === articleVideoId) {
                itemToActivate = item;
            }
        });
        if (itemToActivate) {
            itemToActivate.classList.add('active-history-item');
            currentActiveHistoryItem = itemToActivate;
        } else {
            currentActiveHistoryItem = null;
        }
    }

    function loadArticleFromHistory(index) {
        // ... (no changes here) ...
        if (index >= 0 && index < history.length) {
            const articleData = history[index];
            console.log("Loading from history:", articleData.video_data.title);
            renderOutput(articleData, false);
        }
    }

    function renderOutput(data, isNewSubmission = true) {
        clearError();
        outputSection.style.display = 'block';

        // ... (video player and history update logic remain the same) ...
        console.log("Data for player in renderOutput:", data.video_data);
        if (data.video_data && data.video_data.video_id) {
            loadOrUpdateYouTubePlayer(data.video_data.video_id, data.video_data.title);
            if (isNewSubmission) {
                updateHistory(data);
            }
            setActiveHistoryItem(data.video_data.video_id);
        } else {
            loadOrUpdateYouTubePlayer(null, null);
            if (currentActiveHistoryItem) {
                currentActiveHistoryItem.classList.remove('active-history-item');
                currentActiveHistoryItem = null;
            }
        }

        // ... (summary and ToC rendering remain the same) ...
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
                link.className = "hover:underline text-indigo-600 hover:text-indigo-800";
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    const targetElement = document.getElementById(sectionId);
                    if (targetElement) {
                        targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                });
                listItem.appendChild(link);
                tocListUl.appendChild(listItem);
            });
        } else {
            tocListUl.innerHTML = '<li>No table of contents available.</li>';
        }


        fullArticleTextDiv.innerHTML = '';
        if (data.llm_article_data && data.llm_article_data.article_sections && data.llm_article_data.article_sections.length > 0) {
            data.llm_article_data.article_sections.forEach((section_item, index) => { // Renamed 'section' to 'section_item'
                const sectionContainer = document.createElement('div');
                sectionContainer.className = 'mb-6';
                // Use section_item.heading for ID generation
                const sectionId = `section-${encodeURIComponent(section_item.heading?.replace(/\s+/g, '-').toLowerCase() || `untitled-section-${index}`)}`;
                sectionContainer.id = sectionId;

                const heading = document.createElement('h4');
                heading.className = 'text-lg font-semibold mb-2 text-gray-800';
                heading.textContent = section_item.heading || 'Untitled Section'; // Use section_item.heading
                sectionContainer.appendChild(heading);

                // NEW: Process content_block and append timestamp icon
                const contentBlockText = section_item.content_block || '';
                const timestampSeconds = section_item.timestamp_seconds; // This is a float or null

                // Paragraph handling for content_block_text
                const paragraphs = contentBlockText
                                    .replace(/\r\n/g, '\n')
                                    .split(/\n+/)
                                    .filter(p => p.trim() !== '');

                let contentHtml = paragraphs.map(pText => `<p class="mb-2">${pText}</p>`).join('');

                // Append timestamp icon if timestamp_seconds is valid
                if (timestampSeconds !== null && !isNaN(parseFloat(timestampSeconds)) && data.video_data && data.video_data.video_id) {
                    const totalSeconds = parseFloat(timestampSeconds);
                    const displayHours = Math.floor(totalSeconds / 3600);
                    const remainingSecondsAfterHours = totalSeconds % 3600;
                    const displayMinutes = Math.floor(remainingSecondsAfterHours / 60);
                    const displaySeconds = Math.floor(remainingSecondsAfterHours % 60);
                    const formattedTimeForTitle = `${String(displayHours).padStart(2, '0')}:${String(displayMinutes).padStart(2, '0')}:${String(displaySeconds).padStart(2, '0')}`;
                    const playIconSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-4 h-4 inline-block align-middle ml-1 text-indigo-600 hover:text-indigo-800"><path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" /></svg>`;

                    // Append the icon link after the content paragraph(s) for this block
                    // If contentHtml is empty, it will just add the icon. If there are paragraphs, it adds after the last one.
                    // A simple way is to wrap contentHtml and the icon in a new div or just append to the string.
                    // To ensure it's visually after the text, we can add it as a separate element or cleverly append.
                    // For simplicity here, we'll append it to the HTML string.
                    // It might be better to append it to the last <p> tag's innerHTML if one exists, or add as a sibling.
                    // Let's create a paragraph that just holds the icon if contentHtml itself is just text, or append.
                    
                    const timestampLink = ` <a href="#" 
                                               data-video-id="${data.video_data.video_id}" 
                                               data-timestamp="${Math.floor(totalSeconds)}" 
                                               class="youtube-timestamp-link" 
                                               title="Play from ${formattedTimeForTitle}">${playIconSvg}</a>`;
                    
                    // If contentHtml ends with </p>, insert before the last </p> for better flow,
                    // otherwise, just append. This is a heuristic.
                    if (contentHtml.endsWith('</p>')) {
                        contentHtml = contentHtml.substring(0, contentHtml.length - 4) + timestampLink + '</p>';
                    } else {
                        contentHtml += timestampLink;
                    }
                }

                const contentDiv = document.createElement('div');
                contentDiv.innerHTML = contentHtml;

                if (section_item.relevant_to_search_intent) { // Use section_item
                    sectionContainer.classList.add('bg-yellow-100', 'border-l-4', 'border-yellow-500', 'p-3', 'rounded-r-md');
                }

                sectionContainer.appendChild(contentDiv);
                fullArticleTextDiv.appendChild(sectionContainer);
            });
        } else {
            fullArticleTextDiv.innerHTML = '<p>No article content available.</p>';
        }

        // ... (copy to clipboard button logic remains the same) ...
        if (data.llm_article_data && (data.llm_article_data.summary || data.llm_article_data.article_sections?.length > 0)) {
            copyToClipboardButton.style.display = 'inline-block';
        } else {
            copyToClipboardButton.style.display = 'none';
        }

        if (isNewSubmission) {
             outputSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    if (videoForm) {
        // ... (form submission logic remains the same) ...
        videoForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            clearError();
            showLoading(true);

            const videoUrl = videoUrlInput.value.trim();
            const searchIntentValue = searchIntentInput.value.trim();

            if (!videoUrl) {
                displayError('Please enter a YouTube video URL.');
                showLoading(false);
                return;
            }
            try {
                new URL(videoUrl);
                if (!/^(https?:\/\/)?(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/)/.test(videoUrl)) {
                    displayError('URL must be a valid YouTube video link (e.g., youtube.com/watch?v=... or youtu.be/...).');
                    showLoading(false);
                    return;
                }
            } catch (error) {
                displayError('Please enter a valid URL format (e.g., https://www.youtube.com/watch?v=...).');
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

                showLoading(false);

                if (!response.ok) {
                    let errorData;
                    try { errorData = await response.json(); }
                    catch (e) { errorData = { detail: `HTTP error! Status: ${response.status} - ${response.statusText}` }; }
                    console.error('API Error Response:', errorData);
                    displayError(errorData.detail || `An API error occurred: ${response.status}`);
                    outputSection.style.display = 'none';
                    videoPlayerSection.style.display = 'none';
                    copyToClipboardButton.style.display = 'none';
                    return;
                }

                const data = await response.json();
                console.log('API Success Response:', data);

                if (data.message?.toLowerCase().includes("failed") || (data.video_data && data.video_data.error)) {
                    displayError(data.message || (data.video_data && data.video_data.error) || "Processing failed, please check the video URL or try again.");
                    outputSection.style.display = 'none';
                    videoPlayerSection.style.display = 'none';
                    copyToClipboardButton.style.display = 'none';
                } else if (data.llm_article_data && data.llm_article_data.summary?.toLowerCase().includes("error:")) {
                    displayError(`Processing Error: ${data.llm_article_data.summary}`);
                    outputSection.style.display = 'none';
                    videoPlayerSection.style.display = 'none';
                    copyToClipboardButton.style.display = 'none';
                } else if (data.llm_article_data && (data.llm_article_data.summary || data.llm_article_data.article_sections?.length > 0)) {
                    renderOutput(data, true);
                } else {
                    displayError("Received data from backend, but no article content was generated or an unknown error occurred.");
                    outputSection.style.display = 'none';
                    videoPlayerSection.style.display = 'none';
                    copyToClipboardButton.style.display = 'none';
                }

            } catch (error) {
                console.error('Fetch API Call Error:', error);
                showLoading(false);
                displayError('Failed to connect to the server or network error. Please try again.');
                outputSection.style.display = 'none';
                videoPlayerSection.style.display = 'none';
                copyToClipboardButton.style.display = 'none';
            }
        });
    } else {
        console.error("Video form not found!");
    }

    if(copyToClipboardButton) {
        // ... (copy to clipboard logic remains largely the same, might need slight adjustment if icon text is copied) ...
        // For now, this will copy the icon's SVG code if it's inside the paragraph text.
        // A more advanced copy would strip HTML or specifically format.
        copyToClipboardButton.addEventListener('click', () => {
            let textToCopy = "";
            const articleTitleElement = document.getElementById('video-player-title');
            const articleTitle = articleTitleElement ? articleTitleElement.textContent : "Video Article";
            textToCopy += articleTitle + "\n\n";

            const summaryText = summaryContentDiv.innerText?.trim();
            if (summaryText && !summaryText.toLowerCase().includes("no summary")) {
                textToCopy += "Summary:\n" + summaryText + "\n\n";
            }

            const tocItems = tocListUl.querySelectorAll('li a');
            if (tocItems.length > 0 && !tocListUl.textContent.toLowerCase().includes("no table of contents")) {
                textToCopy += "Table of Contents:\n";
                tocItems.forEach(link => {
                    textToCopy += "- " + link.textContent.trim() + "\n";
                });
                textToCopy += "\n";
            }

            const fullArticleElement = document.getElementById('full-article-text');
            if (fullArticleElement && fullArticleElement.innerText?.trim() && !fullArticleElement.innerText.toLowerCase().includes("no article content")) {
                textToCopy += "Full Article:\n";
                const sections = fullArticleElement.querySelectorAll('div.mb-6');
                sections.forEach(sectionDiv => {
                    const heading = sectionDiv.querySelector('h4');
                    const contentParagraphs = sectionDiv.querySelectorAll('div > p'); // Paragraphs within the content div

                    if (heading) {
                        textToCopy += "\n## " + heading.innerText.trim() + "\n";
                    }
                    if (contentParagraphs.length > 0) {
                        contentParagraphs.forEach(p => {
                             // Get innerText of paragraph. If it contains the SVG, this might be clunky.
                             // For a cleaner copy, you might want to clone the paragraph, remove the SVG link, then get innerText.
                             // Simple approach for now:
                            let pText = p.innerText.trim();
                            // Attempt to remove the visual space taken by SVG if it was just an icon.
                            // This is a heuristic. If SVG had actual text, it would be removed too.
                            // A more robust solution would be to have a data-attribute on the link or not include it in innerText.
                            pText = pText.replace(/\s*\u25B6\s*/g, ' ').trim(); // Replace play icon if it got copied as a character
                            if (pText) {
                                textToCopy += pText.replace(/\n+/g, '\n') + "\n\n";
                            }
                        });
                    } else { // Fallback if direct content div without <p> tags
                        const directContentDiv = sectionDiv.querySelector('div');
                        if (directContentDiv && directContentDiv.innerText) {
                           textToCopy += directContentDiv.innerText.trim().replace(/\n+/g, '\n') + "\n\n";
                        }
                    }
                });
            }

            textToCopy = textToCopy.trim();
            if (textToCopy.replace(articleTitle, "").trim()) {
                navigator.clipboard.writeText(textToCopy).then(() => {
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
    renderHistoryList();
});