import React from 'react';
import { Box, Typography, Divider, List, ListItem, ListItemIcon, ListItemText, IconButton, Paper, Button } from '@mui/material';
import { 
  Bot, 
  Stethoscope, 
  Database, 
  Search, 
  Brain, 
  Activity, 
  User, 
  ShieldCheck, 
  Trash2, 
  Menu,
  ChevronLeft
} from 'lucide-react';
import { motion } from 'framer-motion';

interface SidebarProps {
  isOpen: boolean;
  toggleSidebar: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, toggleSidebar }) => {
  return (
    <motion.div
      initial={false}
      animate={{ width: isOpen ? 320 : 0 }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      style={{ overflow: 'hidden', height: '100%' }}
    >
      <Box sx={{ 
        width: 320, 
        height: '100%', 
        bgcolor: 'rgba(10, 10, 10, 0.8)', 
        backdropFilter: 'blur(12px)',
        borderRight: '1px solid rgba(255, 255, 255, 0.05)',
        display: 'flex',
        flexDirection: 'column',
        p: 3
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
          <Stethoscope size={28} color="#3b82f6" />
          <Typography variant="h6" sx={{ ml: 2, fontWeight: 700, color: 'primary.main' }}>
            MedAssist AI
          </Typography>
        </Box>

        <Box sx={{ mb: 4 }}>
          <Typography variant="overline" sx={{ color: 'text.secondary', fontWeight: 600 }}>
            Core Agents
          </Typography>
          <List size="small">
            <ListItem sx={{ px: 0 }}>
              <ListItemIcon sx={{ minWidth: 40 }}><User size={18} /></ListItemIcon>
              <ListItemText primary="Conversation Agent" />
            </ListItem>
            <ListItem sx={{ px: 0 }}>
              <ListItemIcon sx={{ minWidth: 40 }}><Database size={18} /></ListItemIcon>
              <ListItemText primary="Medical RAG Agent" />
            </ListItem>
            <ListItem sx={{ px: 0 }}>
              <ListItemIcon sx={{ minWidth: 40 }}><Search size={18} /></ListItemIcon>
              <ListItemText primary="Web Search Agent" />
            </ListItem>
          </List>
        </Box>

        <Box sx={{ mb: 4 }}>
          <Typography variant="overline" sx={{ color: 'text.secondary', fontWeight: 600 }}>
            Medical Vision
          </Typography>
          <List size="small">
            <ListItem sx={{ px: 0 }}>
              <ListItemIcon sx={{ minWidth: 40 }}><Brain size={18} /></ListItemIcon>
              <ListItemText primary="Brain MRI Analysis" />
            </ListItem>
            <ListItem sx={{ px: 0 }}>
              <ListItemIcon sx={{ minWidth: 40 }}><Activity size={18} /></ListItemIcon>
              <ListItemText primary="Chest X-Ray Classification" />
            </ListItem>
            <ListItem sx={{ px: 0 }}>
              <ListItemIcon sx={{ minWidth: 40 }}><Stethoscope size={18} /></ListItemIcon>
              <ListItemText primary="Skin Lesion Detection" />
            </ListItem>
          </List>
        </Box>

        <Box sx={{ mb: 4, flexGrow: 1 }}>
          <Typography variant="overline" sx={{ color: 'text.secondary', fontWeight: 600 }}>
            Trust & Safety
          </Typography>
          <List size="small">
            <ListItem sx={{ px: 0 }}>
              <ListItemIcon sx={{ minWidth: 40 }}><ShieldCheck size={18} color="#10b981" /></ListItemIcon>
              <ListItemText primary="Verified Sources Only" />
            </ListItem>
            <ListItem sx={{ px: 0 }}>
              <ListItemIcon sx={{ minWidth: 40 }}><ShieldCheck size={18} color="#10b981" /></ListItemIcon>
              <ListItemText primary="Human-in-the-loop" />
            </ListItem>
          </List>
        </Box>

        <Button 
          fullWidth 
          variant="outlined" 
          color="error" 
          startIcon={<Trash2 size={18} />}
          onClick={() => window.location.reload()}
          sx={{ borderRadius: 2 }}
        >
          Clear Session
        </Button>
      </Box>
    </motion.div>
  );
};

export default Sidebar;
