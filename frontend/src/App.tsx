import React, { useState, useEffect, useRef } from 'react';
import { ThemeProvider, createTheme, CssBaseline, Box, Container, Paper } from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import type { Message } from './types';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#3b82f6',
      dark: '#2563eb',
      light: '#60a5fa',
    },
    background: {
      default: '#050505',
      paper: '#0a0a0a',
    },
    text: {
      primary: '#f5f5f5',
      secondary: '#a3a3a3',
    },
    divider: 'rgba(255, 255, 255, 0.08)',
  },
  typography: {
    fontFamily: '"Inter", "system-ui", "sans-serif"',
    h5: {
      fontFamily: '"Space Grotesk", sans-serif',
      fontWeight: 700,
    },
    h6: {
      fontFamily: '"Space Grotesk", sans-serif',
      fontWeight: 600,
    },
    overline: {
      fontFamily: '"Space Grotesk", sans-serif',
      letterSpacing: '0.1em',
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: '#0a0a0a',
          border: '1px solid rgba(255, 255, 255, 0.05)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
        },
      },
    },
  },
});

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: "Hello! I'm your advanced medical assistant. I can analyze medical images (MRI, X-ray, Skin Lesions) or answer health-related questions using verified sources. How can I help you today?",
      agent: 'System',
      timestamp: new Date(),
    }
  ]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ 
        display: 'flex', 
        height: '100vh', 
        width: '100vw', 
        bgcolor: 'background.default',
        overflow: 'hidden' 
      }}>
        <Sidebar isOpen={isSidebarOpen} toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)} />
        
        <Box component="main" sx={{ 
          flexGrow: 1, 
          display: 'flex', 
          flexDirection: 'column',
          height: '100%',
          position: 'relative',
          transition: 'margin 0.3s'
        }}>
          <ChatArea messages={messages} setMessages={setMessages} />
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
