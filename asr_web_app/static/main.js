document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const recordBtn = document.getElementById('record-btn');
    const statusElem = document.getElementById('status');
    const viFinalElem = document.getElementById('vi-final');
    const viInterimElem = document.getElementById('vi-interim');
    const enFinalElem = document.getElementById('en-final');
    const enInterimElem = document.getElementById('en-interim');
    
    // Audio context and processing variables
    let socket = null;
    let audioContext = null;
    let mediaStream = null;
    let processor = null;
    let isRecording = false;
    
    // Connect to Socket.IO server
    function connectSocket() {
        socket = io();
        
        socket.on('connect', () => {
            console.log('Connected to server');
            statusElem.textContent = 'Kết nối thành công. Nhấn nút để bắt đầu nói.';
        });
        
        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            statusElem.textContent = 'Mất kết nối với máy chủ.';
            stopRecording();
        });
        
        socket.on('asr_result', handleASRResult);
    }
    
    // Handle ASR results from server
    function handleASRResult(result) {
        console.log('ASR Result:', result);
        
        if (result.type === 'vi') {
            if (result.isFinal) {
                appendFinalText(viFinalElem, result.text);
                viInterimElem.textContent = '';
            } else {
                viInterimElem.textContent = result.text;
            }
        } else if (result.type === 'en') {
            if (result.isFinal) {
                appendFinalText(enFinalElem, result.text);
                enInterimElem.textContent = '';
            } else {
                enInterimElem.textContent = result.text;
            }
        }
    }
    
    // Append final text with proper formatting
    function appendFinalText(element, text) {
        if (!text || text.trim() === '') return;
        
        // Capitalize first letter and add period if needed
        let formattedText = text.trim();
        formattedText = formattedText.charAt(0).toUpperCase() + formattedText.slice(1);
        if (!formattedText.endsWith('.') && !formattedText.endsWith('?') && !formattedText.endsWith('!')) {
            formattedText += '.';
        }
        
        // Add space if not empty
        if (element.textContent && !element.textContent.endsWith(' ')) {
            element.textContent += ' ';
        }
        
        element.textContent += formattedText + ' ';
        
        // Scroll to bottom
        const container = element.parentElement;
        container.scrollTop = container.scrollHeight;
    }
    
    // Start recording audio
    async function startRecording() {
        try {
            // Request microphone access
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Create audio context
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            
            // Create audio source from microphone
            const source = audioContext.createMediaStreamSource(mediaStream);
            
            // Create script processor node for audio processing
            processor = audioContext.createScriptProcessor(4096, 1, 1);
            
            // Process audio data
            processor.onaudioprocess = (e) => {
                if (!isRecording) return;
                
                // Get audio data
                const inputData = e.inputBuffer.getChannelData(0);
                
                // Convert float32 to int16
                const int16Data = convertFloat32ToInt16(inputData);
                
                // Send audio data to server
                if (socket && socket.connected) {
                    socket.emit('audio_data', {
                        audio: arrayBufferToBase64(int16Data.buffer)
                    });
                }
            };
            
            // Connect nodes
            source.connect(processor);
            processor.connect(audioContext.destination);
            
            // Update UI
            isRecording = true;
            recordBtn.classList.add('recording');
            statusElem.textContent = 'Đang thu âm... Hãy nói vào microphone.';
            
            console.log('Recording started');
        } catch (error) {
            console.error('Error starting recording:', error);
            statusElem.textContent = 'Lỗi khi truy cập microphone: ' + error.message;
        }
    }
    
    // Stop recording audio
    function stopRecording() {
        if (!isRecording) return;
        
        // Disconnect and clean up audio processing
        if (processor) {
            processor.disconnect();
            processor = null;
        }
        
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
        
        // Update UI
        isRecording = false;
        recordBtn.classList.remove('recording');
        statusElem.textContent = 'Đã dừng thu âm. Nhấn nút để bắt đầu lại.';
        
        console.log('Recording stopped');
    }
    
    // Convert Float32Array to Int16Array for WebSocket
    function convertFloat32ToInt16(float32Array) {
        const int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            // Convert float [-1,1] to int16 [-32768,32767]
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return int16Array;
    }
    
    // Convert ArrayBuffer to Base64 string
    function arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        const binary = bytes.reduce((data, byte) => data + String.fromCharCode(byte), '');
        return window.btoa(binary);
    }
    
    // Toggle recording when button is clicked
    recordBtn.addEventListener('click', () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });
    
    // Clear results button
    document.getElementById('clear-btn')?.addEventListener('click', () => {
        viFinalElem.textContent = '';
        viInterimElem.textContent = '';
        enFinalElem.textContent = '';
        enInterimElem.textContent = '';
    });
    
    // Connect to Socket.IO server when page loads
    connectSocket();
});