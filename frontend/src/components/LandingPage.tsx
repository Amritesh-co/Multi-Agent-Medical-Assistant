import React from 'react';
import { Box, Typography, Button, Container, Grid, Paper, Stack, IconButton } from '@mui/material';
import { motion } from 'framer-motion';
import { 
  Bot, 
  Stethoscope, 
  Brain, 
  Activity, 
  ShieldCheck, 
  Search, 
  Database, 
  ChevronRight,
  Sparkles,
  Zap,
  Globe,
  Lock
} from 'lucide-react';

interface LandingPageProps {
  onStart: () => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onStart }) => {
  return (
    <Box sx={{ 
      height: '100vh', 
      width: '100vw', 
      overflowY: 'auto', 
      bgcolor: '#050505',
      color: '#f5f5f5',
      position: 'relative',
      '&::-webkit-scrollbar': { width: '8px' },
      '&::-webkit-scrollbar-thumb': { bgcolor: '#222', borderRadius: '4px' }
    }}>
      {/* Background Decorative Elements */}
      <Box sx={{ 
        position: 'absolute', 
        top: '10%', 
        left: '20%', 
        width: '40vw', 
        height: '40vw', 
        bgcolor: 'rgba(59, 130, 246, 0.03)', 
        filter: 'blur(120px)', 
        borderRadius: '50%',
        zIndex: 0 
      }} />

      {/* Header */}
      <Container maxWidth="lg" sx={{ pt: 4, pb: 2, position: 'relative', zIndex: 1 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Stack direction="row" alignItems="center" spacing={1.5}>
            <Box sx={{ 
              p: 1, 
              bgcolor: 'primary.main', 
              borderRadius: 2, 
              display: 'flex', 
              boxShadow: '0 0 20px rgba(59, 130, 246, 0.3)' 
            }}>
              <Stethoscope size={24} color="white" />
            </Box>
            <Typography variant="h6" sx={{ fontWeight: 800, letterSpacing: '-0.02em', fontFamily: '"Space Grotesk", sans-serif' }}>
              MEDASSIST AI
            </Typography>
          </Stack>
          <Stack direction="row" spacing={3} sx={{ display: { xs: 'none', md: 'flex' } }}>
            <Typography variant="body2" sx={{ color: 'text.secondary', cursor: 'pointer', '&:hover': { color: 'primary.main' } }}>Features</Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary', cursor: 'pointer', '&:hover': { color: 'primary.main' } }}>Capabilities</Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary', cursor: 'pointer', '&:hover': { color: 'primary.main' } }}>Security</Typography>
          </Stack>
          <Button 
            variant="contained" 
            size="small" 
            onClick={onStart}
            sx={{ 
              borderRadius: '20px', 
              px: 3, 
              textTransform: 'none', 
              fontWeight: 600,
              boxShadow: '0 4px 15px rgba(59, 130, 246, 0.2)'
            }}
          >
            Launch Agent
          </Button>
        </Stack>
      </Container>

      {/* Hero Section */}
      <Container maxWidth="md" sx={{ pt: { xs: 10, md: 15 }, pb: 10, textAlign: 'center', position: 'relative', zIndex: 1 }}>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <Box sx={{ mb: 3, display: 'inline-flex', alignItems: 'center', px: 2, py: 0.5, borderRadius: '20px', border: '1px solid rgba(59, 130, 246, 0.2)', bgcolor: 'rgba(59, 130, 246, 0.05)' }}>
            <Sparkles size={14} color="#3b82f6" style={{ marginRight: 8 }} />
            <Typography variant="caption" sx={{ fontWeight: 600, color: 'primary.main', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
              Next-Gen Multi-Agent Medical Intelligence
            </Typography>
          </Box>
          <Typography variant="h1" sx={{ 
            fontSize: { xs: '3rem', md: '4.5rem' }, 
            fontWeight: 800, 
            lineHeight: 1.1, 
            mb: 3,
            fontFamily: '"Space Grotesk", sans-serif',
            background: 'linear-gradient(180deg, #fff 0%, #a3a3a3 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            Clinical Decisions <br /> 
            Powered by AI Agents
          </Typography>
          <Typography variant="h6" sx={{ color: 'text.secondary', mb: 5, fontWeight: 400, maxWidth: '600px', mx: 'auto', lineHeight: 1.6 }}>
            A professional ecosystem of specialized medical agents for diagnostic imaging, 
            verified research, and clinical knowledge retrieval.
          </Typography>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} justifyContent="center">
            <Button 
              variant="contained" 
              size="large" 
              onClick={onStart}
              endIcon={<ChevronRight size={20} />}
              sx={{ 
                height: 56, 
                px: 4, 
                fontSize: '1.1rem', 
                borderRadius: 3, 
                fontWeight: 700, 
                textTransform: 'none' 
              }}
            >
              Get Started for Free
            </Button>
            <Button 
              variant="outlined" 
              size="large" 
              sx={{ 
                height: 56, 
                px: 4, 
                fontSize: '1.1rem', 
                borderRadius: 3, 
                fontWeight: 700, 
                textTransform: 'none',
                borderColor: 'rgba(255, 255, 255, 0.1)',
                color: 'white',
                '&:hover': { borderColor: 'rgba(255, 255, 255, 0.2)', bgcolor: 'rgba(255, 255, 255, 0.05)' }
              }}
            >
              View Documentation
            </Button>
          </Stack>
        </motion.div>
      </Container>

      {/* Feature Grid */}
      <Container maxWidth="lg" sx={{ py: 10, position: 'relative', zIndex: 1 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <FeatureCard 
              icon={<Database size={24} color="#3b82f6" />}
              title="Medical RAG"
              description="Semantic retrieval from clinical documentation with automated structural parsing of tables and images."
              delay={0.2}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <FeatureCard 
              icon={<Brain size={24} color="#a855f7" />}
              title="Vision Diagnostics"
              description="State-of-the-art inference for Brain MRI, Chest X-rays, and Skin Lesion segmentation."
              delay={0.4}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <FeatureCard 
              icon={<Globe size={24} color="#10b981" />}
              title="Verified Search"
              description="Integrated Tavily search for real-time medical guidelines from trusted global health authorities."
              delay={0.6}
            />
          </Grid>
        </Grid>
      </Container>

      {/* Capabilities Section */}
      <Box sx={{ py: 15, bgcolor: 'rgba(255, 255, 255, 0.01)', borderY: '1px solid rgba(255, 255, 255, 0.03)' }}>
        <Container maxWidth="lg">
          <Grid container spacing={8} alignItems="center">
            <Grid item xs={12} md={6}>
              <Typography variant="overline" sx={{ color: 'primary.main', fontWeight: 700 }}>
                Specialized Agents
              </Typography>
              <Typography variant="h3" sx={{ fontWeight: 800, mb: 3, fontFamily: '"Space Grotesk", sans-serif' }}>
                Multi-Agent Orchestration
              </Typography>
              <Typography variant="body1" sx={{ color: 'text.secondary', mb: 4, lineHeight: 1.8 }}>
                MedAssist AI doesn't rely on a single model. It uses a LangGraph-powered orchestration 
                layer to route your queries to the most qualified agent.
              </Typography>
              <Stack spacing={2}>
                <CapabilityItem icon={<Zap size={18} />} text="Dynamic Triage & Routing System" />
                <CapabilityItem icon={<Lock size={18} />} text="Built-in Medical Guardrails & Safety" />
                <CapabilityItem icon={<ShieldCheck size={18} />} text="Human-in-the-Loop Validation Workflow" />
              </Stack>
            </Grid>
            <Grid item xs={12} md={6}>
              <Paper sx={{ 
                p: 4, 
                bgcolor: '#0a0a0a', 
                border: '1px solid rgba(255, 255, 255, 0.05)',
                borderRadius: 4,
                boxShadow: '0 20px 40px rgba(0, 0, 0, 0.4)'
              }}>
                <Box sx={{ mb: 4 }}>
                  <Typography variant="caption" color="text.secondary">AGENT STATUS</Typography>
                  <Stack direction="row" spacing={1} mt={1}>
                    <Chip label="VISION: ONLINE" size="small" sx={{ bgcolor: 'rgba(168, 85, 247, 0.1)', color: '#a855f7', fontWeight: 600 }} />
                    <Chip label="RAG: ACTIVE" size="small" sx={{ bgcolor: 'rgba(59, 130, 246, 0.1)', color: '#3b82f6', fontWeight: 600 }} />
                  </Stack>
                </Box>
                <Typography variant="body2" sx={{ fontFamily: 'monospace', color: '#10b981', mb: 2 }}>
                  {">"} INITIALIZING NVIDIA NIM... <br />
                  {">"} LOADING MEDICAL KNOWLEDGE BASE... <br />
                  {">"} CONNECTING TAVILY ENGINE... <br />
                  {">"} SYSTEM READY.
                </Typography>
                <Box sx={{ height: 120, display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: '#050505', borderRadius: 2, border: '1px dashed #222' }}>
                  <Bot size={40} className="animate-pulse" />
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Footer */}
      <Container maxWidth="lg" sx={{ py: 6, textAlign: 'center', borderTop: '1px solid rgba(255, 255, 255, 0.05)' }}>
        <Typography variant="body2" color="text.secondary">
          © 2026 MedAssist AI Clinical Intelligence. For educational and professional assistance only.
        </Typography>
      </Container>
    </Box>
  );
};

const FeatureCard: React.FC<{ icon: React.ReactNode, title: string, description: string, delay: number }> = ({ icon, title, description, delay }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, delay }}
    viewport={{ once: true }}
  >
    <Paper sx={{ 
      p: 4, 
      height: '100%', 
      bgcolor: 'rgba(20, 20, 20, 0.4)', 
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(255, 255, 255, 0.05)',
      borderRadius: 4,
      transition: 'transform 0.2s',
      '&:hover': { transform: 'translateY(-5px)', borderColor: 'rgba(59, 130, 246, 0.2)' }
    }}>
      <Box sx={{ mb: 2 }}>{icon}</Box>
      <Typography variant="h6" sx={{ fontWeight: 700, mb: 1.5, fontFamily: '"Space Grotesk", sans-serif' }}>{title}</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.6 }}>{description}</Typography>
    </Paper>
  </motion.div>
);

const CapabilityItem: React.FC<{ icon: React.ReactNode, text: string }> = ({ icon, text }) => (
  <Stack direction="row" spacing={1.5} alignItems="center">
    <Box sx={{ color: 'primary.main', display: 'flex' }}>{icon}</Box>
    <Typography variant="body2" sx={{ fontWeight: 500 }}>{text}</Typography>
  </Stack>
);

const Chip: React.FC<{ label: string, size?: "small" | "medium", sx?: any }> = ({ label, size = "medium", sx }) => (
  <Box sx={{ 
    display: 'inline-flex', 
    px: 1.5, 
    py: 0.5, 
    borderRadius: '10px', 
    fontSize: '0.7rem', 
    letterSpacing: '0.05em',
    ...sx 
  }}>
    {label}
  </Box>
);

export default LandingPage;
