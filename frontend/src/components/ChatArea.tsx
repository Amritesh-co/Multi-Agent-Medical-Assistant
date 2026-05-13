import React, { useState, useRef, useEffect } from 'react';
import { Box, TextField, IconButton, Paper, Typography, CircularProgress, Chip, Avatar } from '@mui/material';
import { Send, Image as ImageIcon, Mic, MicOff, StopCircle, User, Bot, Play, Pause, AlertCircle, CheckCircle2, XCircle, Activity } from 'lucide-react';
import type { Message, ApiResponse } from '../types';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { motion, AnimatePresence } from 'framer-motion';

interface ChatAreaProps {
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

const ChatArea: React.FC<ChatAreaProps> = ({ messages, setMessages }) => {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setSelectedImagePreview] = useState<string | null>(null);
  const [isTranscribing, setIsTranscribing] = useState(false);
  
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => setSelectedImagePreview(e.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim() && !selectedImage) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      image: imagePreview || undefined,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setSelectedImage(null);
    setSelectedImagePreview(null);
    setIsLoading(true);

    try {
      let response;
      if (selectedImage) {
        const formData = new FormData();
        formData.append('text', input);
        formData.append('image', selectedImage);
        response = await axios.post<ApiResponse>('/upload', formData);
      } else {
        response = await axios.post<ApiResponse>('/chat', {
          query: input,
          conversation_history: []
        });
      }

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.data.response,
        agent: response.data.agent,
        resultImage: response.data.result_image,
        isHumanValidationRequired: response.data.agent.includes('HUMAN_VALIDATION'),
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'I apologize, but I encountered an error processing your request. Please try again.',
        agent: 'System',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleRecording = async () => {
    if (!isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        mediaRecorderRef.current = mediaRecorder;
        audioChunksRef.current = [];

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) audioChunksRef.current.push(e.data);
        };

        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          await transcribeAudio(audioBlob);
        };

        mediaRecorder.start();
        setIsRecording(true);
      } catch (err) {
        console.error('Mic access denied:', err);
      }
    } else {
      mediaRecorderRef.current?.stop();
      mediaRecorderRef.current?.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };

  const transcribeAudio = async (blob: Blob) => {
    setIsTranscribing(true);
    const formData = new FormData();
    formData.append('audio', blob);

    try {
      const response = await axios.post('/transcribe', formData);
      if (response.data.transcript) {
        setInput(response.data.transcript);
      }
    } catch (err) {
      console.error('Transcription failed:', err);
    } finally {
      setIsTranscribing(false);
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', p: { xs: 2, md: 4 } }}>
      <Box sx={{ mb: 4, display: 'flex', alignItems: 'center' }}>
        <Typography variant="h5" sx={{ fontWeight: 700 }}>
          Clinical Intelligence
        </Typography>
        <Chip 
          label="Live Status: Online" 
          color="success" 
          size="small" 
          variant="outlined" 
          sx={{ ml: 2, height: 24, fontSize: '0.7rem' }} 
        />
      </Box>

      <Box 
        ref={scrollRef}
        sx={{ 
          flexGrow: 1, 
          overflowY: 'auto', 
          display: 'flex', 
          flexDirection: 'column',
          gap: 3,
          pr: 2,
          mb: 2,
          '&::-webkit-scrollbar': { width: '6px' },
          '&::-webkit-scrollbar-thumb': { bgcolor: 'divider', borderRadius: '3px' }
        }}
      >
        <AnimatePresence>
          {messages.map((msg) => (
            <MessageItem key={msg.id} message={msg} />
          ))}
        </AnimatePresence>
        
        {isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            style={{ display: 'flex', gap: 16 }}
          >
            <Avatar sx={{ bgcolor: 'primary.main', width: 32, height: 32 }}>
              <Bot size={18} />
            </Avatar>
            <Paper sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
              <CircularProgress size={16} />
              <Typography variant="body2" color="text.secondary">
                Analyzing clinical data...
              </Typography>
            </Paper>
          </motion.div>
        )}
      </Box>

      {/* Input Area */}
      <Box sx={{ position: 'relative' }}>
        {imagePreview && (
          <Paper sx={{ 
            position: 'absolute', 
            bottom: '100%', 
            left: 0, 
            mb: 2, 
            p: 1, 
            borderRadius: 2, 
            display: 'flex', 
            alignItems: 'center', 
            gap: 1,
            bgcolor: 'background.paper',
            border: '1px solid',
            borderColor: 'divider'
          }}>
            <img src={imagePreview} style={{ width: 40, height: 40, borderRadius: 4, objectFit: 'cover' }} />
            <Typography variant="caption" sx={{ maxWidth: 100, noWrap: true }}>Medical Image</Typography>
            <IconButton size="small" onClick={() => { setSelectedImage(null); setSelectedImagePreview(null); }}>
              <XCircle size={14} />
            </IconButton>
          </Paper>
        )}

        <Paper 
          elevation={0}
          sx={{ 
            p: 1, 
            display: 'flex', 
            alignItems: 'center', 
            gap: 1,
            bgcolor: 'rgba(20, 20, 20, 0.6)',
            backdropFilter: 'blur(16px)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: 4,
            transition: 'all 0.2s',
            '&:focus-within': { 
              borderColor: 'primary.main',
              boxShadow: '0 0 0 2px rgba(59, 130, 246, 0.1)'
            }
          }}
        >
          <input
            type="file"
            hidden
            ref={fileInputRef}
            onChange={handleImageChange}
            accept="image/*"
          />
          <IconButton onClick={() => fileInputRef.current?.click()}>
            <ImageIcon size={20} />
          </IconButton>
          
          <IconButton 
            onClick={toggleRecording} 
            color={isRecording ? "error" : "default"}
          >
            {isRecording ? <StopCircle size={20} /> : <Mic size={20} />}
          </IconButton>

          <TextField
            fullWidth
            multiline
            maxRows={4}
            placeholder={isTranscribing ? "Transcribing voice..." : "Describe symptoms or ask clinical questions..."}
            variant="standard"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isTranscribing}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
              }
            }}
            slotProps={{
              input: {
                disableUnderline: true,
                sx: { py: 1, px: 1 }
              }
            }}
          />

          <IconButton 
            onClick={handleSendMessage} 
            disabled={(!input.trim() && !selectedImage) || isLoading}
            sx={{ 
              bgcolor: 'primary.main', 
              color: 'white',
              '&:hover': { bgcolor: 'primary.dark' },
              '&.Mui-disabled': { bgcolor: 'divider' }
            }}
          >
            <Send size={18} />
          </IconButton>
        </Paper>
        {isRecording && (
          <Typography variant="caption" color="error" sx={{ position: 'absolute', top: '100%', mt: 1, ml: 1, display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: 'red', display: 'inline-block' }}></span>
            Recording audio...
          </Typography>
        )}
      </Box>
    </Box>
  );
};

const MessageItem: React.FC<{ message: Message }> = ({ message }) => {
  const isBot = message.role === 'assistant';
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [audio, setAudio] = useState<HTMLAudioElement | null>(null);

  const handlePlayVoice = async () => {
    if (isSpeaking && audio) {
      audio.pause();
      setIsSpeaking(false);
      return;
    }

    try {
      const response = await axios.post('/generate-speech', {
        text: message.content.substring(0, 1000), // Safety limit
        voice_id: 'XrExE9yKIg1WjnnlVkGX'
      }, { responseType: 'blob' });

      const audioUrl = URL.createObjectURL(response.data);
      const newAudio = new Audio(audioUrl);
      newAudio.onended = () => {
        setIsSpeaking(false);
        URL.revokeObjectURL(audioUrl);
      };
      setAudio(newAudio);
      newAudio.play();
      setIsSpeaking(true);
    } catch (err) {
      console.error('Speech failed:', err);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      style={{ 
        display: 'flex', 
        gap: 16, 
        flexDirection: isBot ? 'row' : 'row-reverse' 
      }}
    >
      <Avatar sx={{ 
        bgcolor: isBot ? 'primary.main' : 'secondary.main',
        width: 36,
        height: 36
      }}>
        {isBot ? <Bot size={20} /> : <User size={20} />}
      </Avatar>

      <Box sx={{ 
        maxWidth: '75%', 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: isBot ? 'flex-start' : 'flex-end' 
      }}>
        {message.agent && (
          <Typography variant="caption" sx={{ mb: 0.5, color: 'text.secondary', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            {message.agent}
          </Typography>
        )}
        
        <Paper 
          elevation={0}
          sx={{ 
            p: 2, 
            borderRadius: 3, 
            bgcolor: isBot ? 'background.paper' : 'primary.main',
            color: isBot ? 'text.primary' : 'primary.foreground',
            border: isBot ? '1px solid' : 'none',
            borderColor: 'divider',
            position: 'relative'
          }}
        >
          {message.image && (
            <Box sx={{ mb: 2 }}>
              <img src={message.image} style={{ width: '100%', borderRadius: 8, maxHeight: 300, objectFit: 'contain' }} />
            </Box>
          )}

          <Box className="markdown-content" sx={{ 
            '& p': { m: 0 },
            '& table': { borderCollapse: 'collapse', my: 2, width: '100%' },
            '& th, & td': { border: '1px solid', borderColor: 'divider', p: 1 },
            '& th': { bgcolor: 'rgba(255,255,255,0.05)' }
          }}>
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </Box>

          {message.resultImage && (
            <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Box>
                <img src={message.resultImage} style={{ maxWidth: '100%', borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)' }} />
                <Typography variant="caption" align="center" sx={{ display: 'block', mt: 0.5, color: 'text.secondary' }}>
                  Analysis Result
                </Typography>
              </Box>
            </Box>
          )}

          {isBot && (
            <Box sx={{ mt: 2, pt: 1, borderTop: '1px solid', borderColor: 'divider', display: 'flex', gap: 1 }}>
              <IconButton size="small" onClick={handlePlayVoice} sx={{ color: 'primary.main' }}>
                {isSpeaking ? <Pause size={14} /> : <Play size={14} />}
              </IconButton>
              <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', color: 'text.secondary' }}>
                {isSpeaking ? 'Speaking...' : 'Play Voice'}
              </Typography>
            </Box>
          )}
        </Paper>

        {message.isHumanValidationRequired && (
          <Paper sx={{ 
            mt: 1.5, 
            p: 2, 
            borderRadius: 3, 
            bgcolor: 'rgba(59, 130, 246, 0.05)', 
            border: '1px dashed',
            borderColor: 'primary.main'
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
              <AlertCircle size={16} color="#3b82f6" />
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>Clinical Validation Required</Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Chip icon={<CheckCircle2 size={14} />} label="Approve" clickable color="success" size="small" variant="filled" />
              <Chip icon={<XCircle size={14} />} label="Flag Discrepancy" clickable color="error" size="small" variant="outlined" />
            </Box>
          </Paper>
        )}
      </Box>
    </motion.div>
  );
};

export default ChatArea;
