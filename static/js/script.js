const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileNameDisplay = document.getElementById('file-name');
const analyzeBtn = document.getElementById('analyze-btn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('results-section');
const originalPreview = document.getElementById('original-preview');
const resultPreview = document.getElementById('result-preview');

let currentFile = null;

// --- Drag & Drop Handlers ---
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    
    if (e.dataTransfer.files.length) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (fileInput.files.length) {
        handleFileSelect(fileInput.files[0]);
    }
});

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        alert("Please select an image file.");
        return;
    }

    currentFile = file;
    fileNameDisplay.textContent = `Selected: ${file.name}`;
    analyzeBtn.disabled = false;
    
    // Show preview of original immediately
    const reader = new FileReader();
    reader.onload = (e) => {
        originalPreview.src = e.target.result;
    };
    reader.readAsDataURL(file);
    
    // Hide results if we select a new file
    resultsSection.classList.remove('visible');
}

// --- Analysis Handler ---
analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // Show Loading
    loading.classList.add('active');
    
    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            // Update Result Image
            // We start caching issues for same filenames by adding timestamp
            resultPreview.src = `/${data.result}?t=${new Date().getTime()}`;
            
            // Show Results
            resultsSection.classList.add('visible');
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            alert(`Error: ${data.error}`);
        }

    } catch (error) {
        console.error(error);
        alert("An error occurred during analysis.");
    } finally {
        loading.classList.remove('active');
    }
});
