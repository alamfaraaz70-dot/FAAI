/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { GoogleGenAI, Type, Modality } from "@google/genai";
import { 
  Brain, 
  Zap, 
  ShieldCheck, 
  Search, 
  History, 
  Settings, 
  Image as ImageIcon, 
  Mic, 
  Volume2, 
  Download, 
  ThumbsUp, 
  ThumbsDown, 
  AlertTriangle, 
  CheckCircle2, 
  ChevronRight, 
  Loader2, 
  Trophy, 
  BarChart3, 
  Layers, 
  HelpCircle,
  X,
  Plus,
  Table,
  ArrowRight,
  Sparkles,
  Info
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  ResponsiveContainer, 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  Radar, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip as RechartsTooltip,
  Cell
} from 'recharts';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

import { TRANSLATIONS } from './translations';
import { FAQ_DATA } from './faqData';

// --- Utility ---
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Types ---
type Domain = 'General' | 'Math' | 'Coding' | 'Theory' | 'Creative';
type Mode = 'Fast' | 'Deep';
type Language = 'English' | 'Hindi' | 'Urdu' | 'Spanish' | 'French' | 'German';

const LANGUAGES: { name: Language; flag: string }[] = [
  { name: 'English', flag: '🇺🇸' },
  { name: 'Hindi', flag: '🇮🇳' },
  { name: 'Urdu', flag: '🇵🇰' },
  { name: 'Spanish', flag: '🇪🇸' },
  { name: 'French', flag: '🇫🇷' },
  { name: 'German', flag: '🇩🇪' },
];

interface ModelScore {
  accuracy: number;
  logic: number;
  completeness: number;
  clarity: number;
  creativity: number;
  speed: number;
  total: number;
}

interface ModelDebate {
  defense: string;
  critique: string;
}

interface ConsensusResult {
  query_analysis: {
    type: string;
    complexity: string;
    intent: string;
  };
  debate: {
    GPT: ModelDebate;
    Gemini: ModelDebate;
    Claude: ModelDebate;
  };
  scores: {
    GPT: ModelScore;
    Gemini: ModelScore;
    Claude: ModelScore;
  };
  source_verification: {
    GPT: string;
    Gemini: string;
    Claude: string;
  };
  truth_check: string;
  hallucination_flags: string;
  speed_winner: string;
  best_model: string;
  justification: string;
  final_answer: string;
  summary: {
    short: string;
    medium: string;
    detailed: string;
  };
  learning_mode: {
    beginner: string;
    intermediate: string;
    expert: string;
  };
  improvements: {
    GPT: string;
    Gemini: string;
    Claude: string;
  };
  confidence_score: string;
  read_aloud_script: string;
}

interface FileAttachment {
  name: string;
  type: string;
  data: string; // base64 without prefix
  preview?: string; // with prefix for display
}

interface HistoryItem {
  id: string;
  timestamp: number;
  question: string;
  domain: Domain;
  data: ConsensusResult;
}

// --- Constants ---
const MODELS = ['Gemini 3.1 Pro', 'GPT-4o (Simulated)', 'Claude 3.5 Sonnet (Simulated)'];

const SYSTEM_PROMPT = (domain: Domain, mode: Mode, language: Language, useTabular: boolean) => `
You are an Advanced Multi-Agent AI System designed to evaluate, compare, and reason across multiple LLM outputs (GPT, Gemini, Claude, etc.).
Your goal is to deliver the MOST accurate answer, explain WHY it is correct, and provide deep analytical insights using multiple intelligent modules.

DOMAIN: ${domain}
MODE: ${mode}
LANGUAGE: ${language}
TABULAR_PREFERENCE: ${useTabular ? 'YES (Use Markdown Tables for structured data)' : 'AUTO (Detect if structured data/comparison/statistics is needed)'}

CRITICAL: You MUST respond in ${language}. All analysis, explanations, and the final answer must be in ${language}.

${useTabular ? 'IMPORTANT: Focus on providing data in a clear TABULAR format whenever possible. Use Markdown Tables.' : 'AUTOMATIC DETECTION: If the query involves comparisons, statistical data, lists, or analytics, you MUST use Markdown Tables to present this structured data clearly.'}

---
🔍 INPUT:
USER_QUERY: {The user's query}
USER_PREFERENCE: { "priority": "${mode === 'Fast' ? 'speed' : 'accuracy'}" }

---
🧠 SYSTEM MODULES (EXECUTE IN ORDER):

1️⃣ QUERY ANALYSIS: Identify type, complexity, and intent.
2️⃣ MULTI-AGENT DEBATE MODE: Simulate GPT, Gemini, and Claude. For EACH: DEFENSE (why correct) and CRITIQUE (flaws in others).
3️⃣ CRITERIA-WISE EVALUATION: Score 0-100 on Accuracy (40%), Logic (20%), Completeness (15%), Clarity (10%), Creativity (10%), Speed (5%).
4️⃣ SOURCE VERIFICATION: Label "Verified", "Weak", or "None".
5️⃣ TRUTH CHECKER AI: Independent fact-check. Output verification result and explanation.
6️⃣ SPEED COMPARISON: Identify fastest model.
7️⃣ HALLUCINATION DETECTOR: Flag fabricated facts or overconfidence.
8️⃣ FINAL MODEL SELECTION: Choose BEST model and justify.
9️⃣ FINAL ANSWER SYNTHESIS: Improve the best answer for accuracy, clarity, and completeness.
🔟 SMART SUMMARY GENERATOR: Short (2 lines), Medium, and Detailed.
1️⃣1️⃣ LEARNING MODE: Explain in Beginner, Intermediate, and Expert levels.
1️⃣2️⃣ AI IMPROVEMENT SUGGESTIONS: How each model could improve.
1️⃣3️⃣ CONFIDENCE METER: Overall score (0-100%).
1️⃣4️⃣ READ ALOUD SCRIPT: Conversational and clear format.

---
📊 OUTPUT FORMAT (STRICT JSON):
{
  "query_analysis": { "type": "...", "complexity": "...", "intent": "..." },
  "debate": {
    "GPT": { "defense": "...", "critique": "..." },
    "Gemini": { "defense": "...", "critique": "..." },
    "Claude": { "defense": "...", "critique": "..." }
  },
  "scores": {
    "GPT": { "accuracy": 0, "logic": 0, "completeness": 0, "clarity": 0, "creativity": 0, "speed": 0, "total": 0 },
    "Gemini": { "accuracy": 0, "logic": 0, "completeness": 0, "clarity": 0, "creativity": 0, "speed": 0, "total": 0 },
    "Claude": { "accuracy": 0, "logic": 0, "completeness": 0, "clarity": 0, "creativity": 0, "speed": 0, "total": 0 }
  },
  "source_verification": { "GPT": "...", "Gemini": "...", "Claude": "..." },
  "truth_check": "...",
  "hallucination_flags": "...",
  "speed_winner": "...",
  "best_model": "...",
  "justification": "...",
  "final_answer": "...",
  "summary": { "short": "...", "medium": "...", "detailed": "..." },
  "learning_mode": { "beginner": "...", "intermediate": "...", "expert": "..." },
  "improvements": { "GPT": "...", "Gemini": "...", "Claude": "..." },
  "confidence_score": "...%",
  "read_aloud_script": "..."
}
`;

// --- Components ---

export default function App() {
  // State
  const [query, setQuery] = useState('');
  const [files, setFiles] = useState<FileAttachment[]>([]);
  const [domain, setDomain] = useState<Domain>('General');
  const [mode, setMode] = useState<Mode>('Deep');
  const [language, setLanguage] = useState<Language>('English');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<HistoryItem | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [activeTab, setActiveTab] = useState<'compare' | 'consensus' | 'history'>('compare');
  const [showHistory, setShowHistory] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [useTabular, setUseTabular] = useState(false);
  const [settingsView, setSettingsView] = useState<'main' | 'language' | 'faq'>('main');
  const [expandedFaq, setExpandedFaq] = useState<number | null>(null);

  const t = TRANSLATIONS[language];

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load History
  useEffect(() => {
    const saved = localStorage.getItem('omnijudge_history');
    if (saved) setHistory(JSON.parse(saved));
  }, []);

  // Save History
  useEffect(() => {
    localStorage.setItem('omnijudge_history', JSON.stringify(history));
  }, [history]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files;
    if (selectedFiles) {
      Array.from(selectedFiles).forEach(file => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = reader.result as string;
          setFiles(prev => [...prev, {
            name: file.name,
            type: file.type,
            data: base64.split(',')[1],
            preview: base64.startsWith('data:image/') ? base64 : undefined
          }]);
        };
        reader.readAsDataURL(file);
      });
    }
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const [isReadingAloud, setIsReadingAloud] = useState(false);

  const downloadCSV = (markdown: string) => {
    const tableRegex = /\|(.+)\|/g;
    const rows = markdown.match(tableRegex);
    if (!rows) return;

    const csvContent = rows
      .map(row => {
        // Remove pipes and split by pipes, then clean up
        return row
          .split('|')
          .filter(cell => cell.trim() !== '')
          .map(cell => `"${cell.trim().replace(/"/g, '""')}"`)
          .join(',');
      })
      .filter(row => !row.includes('---')) // Filter out markdown table separators
      .join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    if (link.download !== undefined) {
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', 'analysis_data.csv');
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const handleReadAloud = async () => {
    console.log("Read Aloud triggered");
    if (!result || isReadingAloud) {
      console.log("Read Aloud skipped:", { hasResult: !!result, isReadingAloud });
      return;
    }

    setIsReadingAloud(true);
    try {
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        throw new Error('GEMINI_API_KEY is not defined');
      }
      console.log("Initializing Gemini TTS...");
      const ai = new GoogleGenAI({ apiKey });
      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash-preview-tts",
        contents: [{ parts: [{ text: result.data.read_aloud_script }] }],
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: 'Kore' },
            },
          },
        },
      });

      const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (base64Audio) {
        console.log("Audio data received, length:", base64Audio.length);
        
        // Convert base64 to ArrayBuffer
        const binaryString = window.atob(base64Audio);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        
        // Gemini TTS returns 16-bit PCM, mono, 24kHz
        const AudioContextClass = (window as any).AudioContext || (window as any).webkitAudioContext;
        const audioContext = new AudioContextClass();
        const sampleRate = 24000;
        const numChannels = 1;
        
        // Create Int16Array from the bytes
        const int16Data = new Int16Array(bytes.buffer);
        
        // Create Float32Array for AudioBuffer
        const float32Data = new Float32Array(int16Data.length);
        for (let i = 0; i < int16Data.length; i++) {
          // Convert 16-bit PCM to float [-1.0, 1.0]
          float32Data[i] = int16Data[i] / 32768.0;
        }
        
        const audioBuffer = audioContext.createBuffer(numChannels, float32Data.length, sampleRate);
        audioBuffer.getChannelData(0).set(float32Data);
        
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        
        source.onended = () => {
          console.log("Audio playback ended");
          setIsReadingAloud(false);
          audioContext.close();
        };
        
        console.log("Starting audio playback...");
        source.start(0);
      } else {
        throw new Error('No audio data received from model');
      }
    } catch (err) {
      console.error('TTS Error:', err);
      setError('Failed to read aloud. Please try again.');
      setIsReadingAloud(false);
    }
  };

  const runConsensus = async () => {
    if (!query && files.length === 0) return;

    setLoading(true);
    setResult(null);
    setError(null);
    setActiveTab('compare');

    try {
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        throw new Error("API Key is missing. Please check your environment variables.");
      }

      const ai = new GoogleGenAI({ apiKey });

      let contents: any[] = [];
      if (query) {
        contents.push({ text: query });
      }

      if (files.length > 0) {
        if (!query) {
          contents.push({ text: "Analyze these files and provide a consensus report." });
        }
        
        files.forEach(file => {
          contents.push({
            inlineData: {
              mimeType: file.type || "application/octet-stream",
              data: file.data
            }
          });
        });
      }

      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: { parts: contents },
        config: {
          systemInstruction: SYSTEM_PROMPT(domain, mode, language, useTabular),
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              query_analysis: {
                type: Type.OBJECT,
                properties: {
                  type: { type: Type.STRING },
                  complexity: { type: Type.STRING },
                  intent: { type: Type.STRING }
                },
                required: ["type", "complexity", "intent"]
              },
              debate: {
                type: Type.OBJECT,
                properties: {
                  GPT: {
                    type: Type.OBJECT,
                    properties: {
                      defense: { type: Type.STRING },
                      critique: { type: Type.STRING }
                    },
                    required: ["defense", "critique"]
                  },
                  Gemini: {
                    type: Type.OBJECT,
                    properties: {
                      defense: { type: Type.STRING },
                      critique: { type: Type.STRING }
                    },
                    required: ["defense", "critique"]
                  },
                  Claude: {
                    type: Type.OBJECT,
                    properties: {
                      defense: { type: Type.STRING },
                      critique: { type: Type.STRING }
                    },
                    required: ["defense", "critique"]
                  }
                },
                required: ["GPT", "Gemini", "Claude"]
              },
              scores: {
                type: Type.OBJECT,
                properties: {
                  GPT: {
                    type: Type.OBJECT,
                    properties: {
                      accuracy: { type: Type.NUMBER },
                      logic: { type: Type.NUMBER },
                      completeness: { type: Type.NUMBER },
                      clarity: { type: Type.NUMBER },
                      creativity: { type: Type.NUMBER },
                      speed: { type: Type.NUMBER },
                      total: { type: Type.NUMBER }
                    },
                    required: ["accuracy", "logic", "completeness", "clarity", "creativity", "speed", "total"]
                  },
                  Gemini: {
                    type: Type.OBJECT,
                    properties: {
                      accuracy: { type: Type.NUMBER },
                      logic: { type: Type.NUMBER },
                      completeness: { type: Type.NUMBER },
                      clarity: { type: Type.NUMBER },
                      creativity: { type: Type.NUMBER },
                      speed: { type: Type.NUMBER },
                      total: { type: Type.NUMBER }
                    },
                    required: ["accuracy", "logic", "completeness", "clarity", "creativity", "speed", "total"]
                  },
                  Claude: {
                    type: Type.OBJECT,
                    properties: {
                      accuracy: { type: Type.NUMBER },
                      logic: { type: Type.NUMBER },
                      completeness: { type: Type.NUMBER },
                      clarity: { type: Type.NUMBER },
                      creativity: { type: Type.NUMBER },
                      speed: { type: Type.NUMBER },
                      total: { type: Type.NUMBER }
                    },
                    required: ["accuracy", "logic", "completeness", "clarity", "creativity", "speed", "total"]
                  }
                },
                required: ["GPT", "Gemini", "Claude"]
              },
              source_verification: {
                type: Type.OBJECT,
                properties: {
                  GPT: { type: Type.STRING },
                  Gemini: { type: Type.STRING },
                  Claude: { type: Type.STRING }
                },
                required: ["GPT", "Gemini", "Claude"]
              },
              truth_check: { type: Type.STRING },
              hallucination_flags: { type: Type.STRING },
              speed_winner: { type: Type.STRING },
              best_model: { type: Type.STRING },
              justification: { type: Type.STRING },
              final_answer: { type: Type.STRING },
              summary: {
                type: Type.OBJECT,
                properties: {
                  short: { type: Type.STRING },
                  medium: { type: Type.STRING },
                  detailed: { type: Type.STRING }
                },
                required: ["short", "medium", "detailed"]
              },
              learning_mode: {
                type: Type.OBJECT,
                properties: {
                  beginner: { type: Type.STRING },
                  intermediate: { type: Type.STRING },
                  expert: { type: Type.STRING }
                },
                required: ["beginner", "intermediate", "expert"]
              },
              improvements: {
                type: Type.OBJECT,
                properties: {
                  GPT: { type: Type.STRING },
                  Gemini: { type: Type.STRING },
                  Claude: { type: Type.STRING }
                },
                required: ["GPT", "Gemini", "Claude"]
              },
              confidence_score: { type: Type.STRING },
              read_aloud_script: { type: Type.STRING }
            },
            required: [
              "query_analysis", "debate", "scores", "source_verification", 
              "truth_check", "hallucination_flags", "speed_winner", 
              "best_model", "justification", "final_answer", "summary", 
              "learning_mode", "improvements", "confidence_score", "read_aloud_script"
            ]
          }
        }
      });

      if (!response.text) {
        throw new Error("The model returned an empty response.");
      }

      const data = JSON.parse(response.text);
      
      const newItem: HistoryItem = {
        id: Math.random().toString(36).substr(2, 9),
        timestamp: Date.now(),
        question: query || t.fileQuery,
        domain,
        data: data
      };

      setResult(newItem);
      setHistory(prev => [newItem, ...prev].slice(0, 20));
    } catch (err: any) {
      console.error("Consensus Error:", err);
      setError(err.message || "An unexpected error occurred while generating the solution.");
    } finally {
      setLoading(false);
    }
  };

  const exportResult = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `omnijudge-report-${result.id}.json`;
    a.click();
  };

  return (
    <div className="min-h-screen bg-[#050505] text-slate-200 font-sans selection:bg-indigo-500/30">
      {/* Background Orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-600/10 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/10 blur-[120px] rounded-full" />
      </div>

      {/* Navigation */}
      <nav className="sticky top-0 z-50 border-b border-white/5 bg-black/50 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <span className="font-display font-bold text-xl tracking-tight text-white">FA<span className="text-indigo-500">AI</span></span>
          </div>

          <div className="flex items-center gap-4 relative">
            <button 
              onClick={() => setShowHistory(!showHistory)}
              className="p-2 hover:bg-white/5 rounded-full transition-colors relative"
            >
              <History className="w-5 h-5" />
              {history.length > 0 && <span className="absolute top-0 right-0 w-2 h-2 bg-indigo-500 rounded-full" />}
            </button>
            <div className="relative">
              <button 
                onClick={() => {
                  setShowSettings(!showSettings);
                  setSettingsView('main');
                }}
                className={cn(
                  "p-2 rounded-full transition-colors",
                  showSettings ? "bg-white/10 text-white" : "hover:bg-white/5 text-slate-400"
                )}
              >
                <Settings className="w-5 h-5" />
              </button>

              <AnimatePresence>
                {showSettings && (
                  <motion.div
                    initial={{ opacity: 0, y: 10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 10, scale: 0.95 }}
                    className="absolute right-0 mt-2 w-64 bg-[#0a0a0f] border border-white/10 shadow-2xl rounded-3xl p-2 z-[100] backdrop-blur-2xl overflow-hidden"
                  >
                    {settingsView === 'main' ? (
                      <div className="space-y-1">
                        <div className="px-3 py-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest">{t.settings}</div>
                        <button 
                          onClick={() => setSettingsView('language')}
                          className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-white/5 rounded-xl transition-colors group"
                        >
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center">
                              <Volume2 className="w-4 h-4 text-indigo-400" />
                            </div>
                            <span className="text-sm font-medium text-slate-200">{t.language}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-slate-500">{language}</span>
                            <ChevronRight className="w-4 h-4 text-slate-600 group-hover:translate-x-0.5 transition-transform" />
                          </div>
                        </button>
                        <button 
                          onClick={() => setSettingsView('faq')}
                          className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-white/5 rounded-xl transition-colors group"
                        >
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center">
                              <HelpCircle className="w-4 h-4 text-blue-400" />
                            </div>
                            <span className="text-sm font-medium text-slate-200">{t.faq}</span>
                          </div>
                          <ChevronRight className="w-4 h-4 text-slate-600 group-hover:translate-x-0.5 transition-transform" />
                        </button>
                        <button className="w-full flex items-center gap-3 px-3 py-2.5 hover:bg-white/5 rounded-xl transition-colors">
                          <div className="w-8 h-8 rounded-lg bg-purple-500/10 flex items-center justify-center">
                            <Zap className="w-4 h-4 text-purple-400" />
                          </div>
                          <span className="text-sm font-medium text-slate-200">{t.systemStatus}</span>
                        </button>
                      </div>
                    ) : settingsView === 'language' ? (
                      <div className="space-y-1">
                        <div className="flex items-center gap-2 px-1 mb-1">
                          <button 
                            onClick={() => setSettingsView('main')}
                            className="p-2 hover:bg-white/5 rounded-lg transition-colors"
                          >
                            <ChevronRight className="w-4 h-4 rotate-180" />
                          </button>
                          <span className="text-xs font-bold text-white uppercase tracking-widest">{t.selectLanguage}</span>
                        </div>
                        <div className="max-h-64 overflow-y-auto scrollbar-hide">
                          {LANGUAGES.map((lang) => (
                            <button
                              key={lang.name}
                              onClick={() => {
                                setLanguage(lang.name);
                                setShowSettings(false);
                              }}
                              className={cn(
                                "w-full flex items-center justify-between px-3 py-2.5 rounded-xl transition-colors",
                                language === lang.name ? "bg-indigo-600/20 text-indigo-400" : "hover:bg-white/5 text-slate-300"
                              )}
                            >
                              <div className="flex items-center gap-3">
                                <span className="text-lg">{lang.flag}</span>
                                <span className="text-sm font-medium">{lang.name}</span>
                              </div>
                              {language === lang.name && <CheckCircle2 className="w-4 h-4" />}
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-1">
                        <div className="flex items-center gap-2 px-1 mb-1">
                          <button 
                            onClick={() => {
                              setSettingsView('main');
                              setExpandedFaq(null);
                            }}
                            className="p-2 hover:bg-white/5 rounded-lg transition-colors"
                          >
                            <ChevronRight className="w-4 h-4 rotate-180" />
                          </button>
                          <span className="text-xs font-bold text-white uppercase tracking-widest">{t.faq}</span>
                        </div>
                        <div className="max-h-80 overflow-y-auto scrollbar-hide space-y-2 pr-1 p-1">
                          {FAQ_DATA.map((item, idx) => (
                            <div key={idx} className="border border-white/5 rounded-xl overflow-hidden bg-white/[0.02]">
                              <button
                                onClick={() => setExpandedFaq(expandedFaq === idx ? null : idx)}
                                className="w-full flex items-center justify-between px-3 py-3 hover:bg-white/5 transition-colors text-left gap-3"
                              >
                                <span className="text-xs font-medium text-slate-200 leading-tight">{item.question}</span>
                                <ChevronRight className={cn(
                                  "w-3 h-3 text-slate-500 transition-transform flex-shrink-0",
                                  expandedFaq === idx ? "rotate-90" : ""
                                )} />
                              </button>
                              <AnimatePresence>
                                {expandedFaq === idx && (
                                  <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: "auto", opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    className="overflow-hidden"
                                  >
                                    <div className="px-3 pb-3 text-[11px] text-slate-400 leading-relaxed border-t border-white/5 pt-2">
                                      {item.answer}
                                    </div>
                                  </motion.div>
                                )}
                              </AnimatePresence>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
            <div className="h-6 w-[1px] bg-white/10 mx-2" />
            <div className="flex items-center gap-2 bg-white/5 px-3 py-1.5 rounded-full border border-white/10">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
              <span className="text-xs font-medium text-slate-400">{t.systemOnline}</span>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-12 relative">
        {/* Hero / Input Section */}
        {!result && !loading && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-3xl mx-auto text-center mb-16"
          >
            <h1 className="text-5xl md:text-7xl font-display font-bold text-white mb-6 tracking-tight leading-[1.1]">
              {t.heroTitle} <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">{t.heroSubtitle}</span>
            </h1>
            <p className="text-slate-400 text-lg md:text-xl mb-12 leading-relaxed">
              {t.heroDescription}
            </p>
          </motion.div>
        )}

        {/* Main Interface */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Controls & Input */}
          <div className={cn("lg:col-span-12 transition-all duration-500", result || loading ? "lg:col-span-4" : "lg:col-span-8 lg:col-start-3")}>
            <div className="glass-card p-8 neon-border">
              {/* Domain & Mode Selectors */}
              <div className="flex flex-wrap items-center gap-4 mb-8">
                <div className="flex bg-black/40 p-1 rounded-xl border border-white/5">
                  {(['General', 'Math', 'Coding', 'Theory'] as Domain[]).map((d) => (
                    <button
                      key={d}
                      onClick={() => setDomain(d)}
                      className={cn(
                        "px-4 py-2 rounded-lg text-sm font-medium transition-all",
                        domain === d ? "bg-indigo-600 text-white shadow-lg" : "text-slate-500 hover:text-slate-300"
                      )}
                    >
                      {d === 'General' ? t.domainGeneral : d === 'Math' ? t.domainMath : d === 'Coding' ? t.domainCoding : t.domainTheory}
                    </button>
                  ))}
                </div>
                <div className="flex bg-black/40 p-1 rounded-xl border border-white/5">
                  {(['Fast', 'Deep'] as Mode[]).map((m) => (
                    <button
                      key={m}
                      onClick={() => setMode(m)}
                      className={cn(
                        "px-4 py-2 rounded-lg text-sm font-medium transition-all",
                        mode === m ? "bg-purple-600 text-white shadow-lg" : "text-slate-500 hover:text-slate-300"
                      )}
                    >
                      {m === 'Fast' ? t.modeFast : t.modeDeep}
                    </button>
                  ))}
                </div>
                <div className="flex bg-black/40 p-1 rounded-xl border border-white/5">
                  <button
                    onClick={() => setUseTabular(!useTabular)}
                    className={cn(
                      "px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                      useTabular ? "bg-pink-600 text-white shadow-lg" : "text-slate-500 hover:text-slate-300"
                    )}
                  >
                    <Table className="w-4 h-4" />
                    {t.tabularMode}
                  </button>
                </div>
              </div>

              {/* Input Area */}
              <div className="relative group">
                <textarea
                  value={query}
                  onChange={(e) => {
                  setQuery(e.target.value);
                  if (error) setError(null);
                }}
                  placeholder={t.inputPlaceholder}
                  className="w-full bg-black/40 border border-white/10 rounded-2xl p-6 h-48 text-lg text-white placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all resize-none"
                />
                
                {files.length > 0 && (
                  <div className="absolute bottom-4 left-4 flex flex-wrap gap-2 max-w-[80%]">
                    {files.map((file, idx) => (
                      <div key={idx} className="relative group/img">
                        {file.preview ? (
                          <img src={file.preview} alt="Upload" className="w-16 h-16 object-cover rounded-xl border-2 border-indigo-500/50 shadow-xl" />
                        ) : (
                          <div className="w-16 h-16 bg-white/10 rounded-xl border-2 border-white/10 flex flex-col items-center justify-center p-1 overflow-hidden">
                            <Layers className="w-6 h-6 text-indigo-400 mb-1" />
                            <span className="text-[8px] text-slate-400 truncate w-full text-center">{file.name}</span>
                          </div>
                        )}
                        <button 
                          onClick={() => removeFile(idx)}
                          className="absolute -top-2 -right-2 bg-red-500 text-white p-1 rounded-full opacity-0 group-hover/img:opacity-100 transition-opacity z-10"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                <div className="absolute bottom-4 right-4 flex items-center gap-2">
                  <button 
                    onClick={() => fileInputRef.current?.click()}
                    className="p-3 bg-white/5 hover:bg-white/10 rounded-xl transition-colors text-slate-400 hover:text-white"
                  >
                    <Plus className="w-5 h-5" />
                  </button>
                  <button className="p-3 bg-white/5 hover:bg-white/10 rounded-xl transition-colors text-slate-400 hover:text-white">
                    <Mic className="w-5 h-5" />
                  </button>
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    onChange={handleFileUpload} 
                    className="hidden" 
                    multiple
                    accept="image/*,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain" 
                  />
                </div>
              </div>

              <button
                onClick={runConsensus}
                disabled={loading || (!query && files.length === 0)}
                className="w-full mt-6 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white rounded-2xl font-bold text-lg shadow-xl shadow-indigo-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-3 group"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-6 h-6 animate-spin" />
                    {t.reachingConsensus}
                  </>
                ) : (
                  <>
                    <Sparkles className="w-6 h-6 group-hover:rotate-12 transition-transform" />
                    {t.runOmniJudge}
                  </>
                )}
              </button>
            </div>

            {/* Quick Tips */}
            {!result && !loading && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
                {[
                  { icon: ShieldCheck, title: t.factVerification, desc: t.factVerificationDesc },
                  { icon: BarChart3, title: t.logicalAnalysis, desc: t.logicalAnalysisDesc },
                  { icon: Trophy, title: t.bestAnswerSelection, desc: t.bestAnswerSelectionDesc }
                ].map((tip, i) => (
                  <div key={i} className="glass-card p-6 glass-card-hover">
                    <tip.icon className="w-8 h-8 text-indigo-400 mb-4" />
                    <h3 className="font-bold text-white mb-2">{tip.title}</h3>
                    <p className="text-sm text-slate-500">{tip.desc}</p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Right Column: Results */}
          {(result || loading || error) && (
            <div className="lg:col-span-8 space-y-8">
              {error && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-6 bg-red-500/10 border border-red-500/20 rounded-3xl flex items-start gap-4 text-red-400"
                >
                  <div className="p-3 bg-red-500/20 rounded-2xl">
                    <AlertTriangle className="w-6 h-6 shrink-0" />
                  </div>
                  <div className="flex-1">
                    <p className="font-black uppercase tracking-widest text-[10px] mb-1">System Error</p>
                    <p className="text-sm font-medium leading-relaxed">{error}</p>
                    <button 
                      onClick={() => runConsensus()}
                      className="mt-4 px-6 py-2 bg-red-500 text-white rounded-xl text-xs font-black uppercase tracking-widest hover:bg-red-600 transition-all hover:scale-105 active:scale-95"
                    >
                      Try Again
                    </button>
                  </div>
                </motion.div>
              )}

              {/* Tabs */}
              {(result || loading) && (
                <div className="flex items-center gap-2 bg-white/5 p-1 rounded-2xl border border-white/10 w-fit">
                  <button
                    onClick={() => setActiveTab('compare')}
                    className={cn(
                      "px-6 py-2.5 rounded-xl text-sm font-bold transition-all flex items-center gap-2",
                      activeTab === 'compare' ? "bg-white text-black shadow-lg" : "text-slate-400 hover:text-slate-200"
                    )}
                  >
                    <Layers className="w-4 h-4" />
                    {t.modelComparison}
                  </button>
                  <button
                    onClick={() => setActiveTab('consensus')}
                    className={cn(
                      "px-6 py-2.5 rounded-xl text-sm font-bold transition-all flex items-center gap-2",
                      activeTab === 'consensus' ? "bg-white text-black shadow-lg" : "text-slate-400 hover:text-slate-200"
                    )}
                  >
                    <ShieldCheck className="w-4 h-4" />
                    {t.finalConsensus}
                  </button>
                </div>
              )}

              <AnimatePresence mode="wait">
                {loading && (
                  <motion.div 
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="space-y-8"
                  >
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      {[1, 2, 3].map(i => (
                        <div key={i} className="glass-card p-6 h-[400px] animate-pulse bg-white/5" />
                      ))}
                    </div>
                    <div className="glass-card p-10 h-[300px] animate-pulse bg-white/5" />
                  </motion.div>
                )}

                {activeTab === 'compare' && result && !loading && (
                  <motion.div 
                    key="compare"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    className="space-y-8"
                  >
                    {/* Radar Chart for Scores */}
                    <div className="glass-card p-8">
                      <h3 className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-8 flex items-center gap-2">
                        <BarChart3 className="w-4 h-4" />
                        Performance Comparison
                      </h3>
                      <div className="h-[400px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <RadarChart cx="50%" cy="50%" outerRadius="80%" data={[
                            { subject: 'Accuracy', GPT: result.data.scores.GPT.accuracy, Gemini: result.data.scores.Gemini.accuracy, Claude: result.data.scores.Claude.accuracy },
                            { subject: 'Logic', GPT: result.data.scores.GPT.logic, Gemini: result.data.scores.Gemini.logic, Claude: result.data.scores.Claude.logic },
                            { subject: 'Completeness', GPT: result.data.scores.GPT.completeness, Gemini: result.data.scores.Gemini.completeness, Claude: result.data.scores.Claude.completeness },
                            { subject: 'Clarity', GPT: result.data.scores.GPT.clarity, Gemini: result.data.scores.Gemini.clarity, Claude: result.data.scores.Claude.clarity },
                            { subject: 'Creativity', GPT: result.data.scores.GPT.creativity, Gemini: result.data.scores.Gemini.creativity, Claude: result.data.scores.Claude.creativity },
                            { subject: 'Speed', GPT: result.data.scores.GPT.speed, Gemini: result.data.scores.Gemini.speed, Claude: result.data.scores.Claude.speed },
                          ]}>
                            <PolarGrid stroke="#ffffff10" />
                            <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                            <Radar name="GPT" dataKey="GPT" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} />
                            <Radar name="Gemini" dataKey="Gemini" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                            <Radar name="Claude" dataKey="Claude" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.3} />
                            <RechartsTooltip 
                              contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #ffffff10', borderRadius: '12px' }}
                              itemStyle={{ fontSize: '12px' }}
                            />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="flex flex-wrap justify-center gap-6 mt-4">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full bg-indigo-500" />
                          <span className="text-xs text-slate-400">GPT</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full bg-emerald-500" />
                          <span className="text-xs text-slate-400">Gemini</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full bg-amber-500" />
                          <span className="text-xs text-slate-400">Claude</span>
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      {(['GPT', 'Gemini', 'Claude'] as const).map((model) => (
                        <div key={model} className="glass-card p-6 flex flex-col h-full relative overflow-hidden">
                          {/* Total Score Badge */}
                          <div className="absolute top-0 right-0 p-4">
                            <div className={cn(
                              "px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest",
                              result.data.scores[model].total > 85 ? "bg-emerald-500/20 text-emerald-400" : 
                              result.data.scores[model].total > 70 ? "bg-amber-500/20 text-amber-400" : 
                              "bg-red-500/20 text-red-400"
                            )}>
                              {result.data.scores[model].total}% Overall
                            </div>
                          </div>

                          <div className="flex items-center gap-3 mb-6">
                            <div className={cn(
                              "w-8 h-8 rounded-lg flex items-center justify-center",
                              model === 'GPT' ? "bg-indigo-500/20 text-indigo-400" :
                              model === 'Gemini' ? "bg-emerald-500/20 text-emerald-400" :
                              "bg-amber-500/20 text-amber-400"
                            )}>
                              <Zap className="w-4 h-4" />
                            </div>
                            <h3 className="font-display font-bold text-white">{model}</h3>
                          </div>

                          <div className="space-y-6 flex-grow">
                            <div>
                              <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-2">Defense</h4>
                              <p className="text-sm text-slate-300 leading-relaxed italic">"{result.data.debate[model].defense}"</p>
                            </div>
                            <div>
                              <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-2">Critique</h4>
                              <p className="text-sm text-slate-400 leading-relaxed italic">"{result.data.debate[model].critique}"</p>
                            </div>
                            <div>
                              <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-2">Source Verification</h4>
                              <div className={cn(
                                "inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-bold",
                                result.data.source_verification[model] === 'Verified' ? "bg-emerald-500/10 text-emerald-400" :
                                result.data.source_verification[model] === 'Weak' ? "bg-amber-500/10 text-amber-400" :
                                "bg-red-500/10 text-red-400"
                              )}>
                                {result.data.source_verification[model] === 'Verified' ? <CheckCircle2 className="w-3 h-3" /> : <AlertTriangle className="w-3 h-3" />}
                                {result.data.source_verification[model]}
                              </div>
                            </div>
                          </div>

                          <div className="pt-6 border-t border-white/5 mt-6">
                            <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-3">Improvement Suggestion</h4>
                            <p className="text-xs text-slate-500 leading-relaxed">{result.data.improvements[model]}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}

                {activeTab === 'consensus' && result && !loading && (
                  <motion.div 
                    key="consensus"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    className="space-y-8"
                  >
                    {/* Query Analysis */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="p-4 bg-white/5 rounded-2xl border border-white/5">
                        <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-1">Type</h4>
                        <p className="text-sm font-bold text-white">{result.data.query_analysis.type}</p>
                      </div>
                      <div className="p-4 bg-white/5 rounded-2xl border border-white/5">
                        <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-1">Complexity</h4>
                        <p className="text-sm font-bold text-white">{result.data.query_analysis.complexity}</p>
                      </div>
                      <div className="p-4 bg-white/5 rounded-2xl border border-white/5">
                        <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-1">Intent</h4>
                        <p className="text-sm font-bold text-white">{result.data.query_analysis.intent}</p>
                      </div>
                    </div>

                    {/* Final Answer Hero */}
                    <div className="glass-card p-10 bg-gradient-to-br from-indigo-600/10 to-purple-600/10 border-indigo-500/20 relative overflow-hidden">
                      <div className="absolute top-0 right-0 p-6">
                        <div className="flex flex-col items-end">
                          <span className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-1">Confidence</span>
                          <span className="text-2xl font-black text-indigo-400">{result.data.confidence_score}</span>
                        </div>
                      </div>

                      <div className="flex items-center gap-3 mb-8">
                        <div className="w-12 h-12 bg-indigo-500 rounded-2xl flex items-center justify-center shadow-xl shadow-indigo-500/20">
                          <Trophy className="w-6 h-6 text-white" />
                        </div>
                        <div>
                          <h2 className="text-2xl font-display font-bold text-white">{t.verifiedConsensus}</h2>
                          <div className="flex items-center gap-2 text-indigo-400 text-sm font-medium">
                            <CheckCircle2 className="w-4 h-4" />
                            {result.data.best_model} selected as best
                          </div>
                        </div>
                      </div>

                      <div className="markdown-body text-lg text-white leading-relaxed mb-6">
                        <Markdown remarkPlugins={[remarkGfm]}>{result.data.final_answer}</Markdown>
                      </div>

                      {result.data.final_answer.includes('|') && (
                        <div className="flex justify-end mb-10">
                          <button 
                            onClick={() => downloadCSV(result.data.final_answer)}
                            className="flex items-center gap-2 px-4 py-2 bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-400 rounded-xl text-sm font-bold transition-all border border-indigo-500/20 group"
                          >
                            <Download className="w-4 h-4 group-hover:translate-y-0.5 transition-transform" />
                            Download CSV Analysis
                          </button>
                        </div>
                      )}

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="p-6 bg-white/5 rounded-2xl border border-white/5">
                          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Truth Check</h4>
                          <p className="text-sm text-slate-300 leading-relaxed">
                            {result.data.truth_check}
                          </p>
                        </div>
                        <div className="p-6 bg-white/5 rounded-2xl border border-white/5">
                          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Selection Justification</h4>
                          <p className="text-sm text-slate-400 leading-relaxed italic">
                            "{result.data.justification}"
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Summaries */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      <div className="glass-card p-8">
                        <div className="flex items-center gap-2 mb-4 text-emerald-400">
                          <Zap className="w-4 h-4" />
                          <h4 className="text-xs font-bold uppercase tracking-widest">Short Summary</h4>
                        </div>
                        <p className="text-sm text-slate-400 leading-relaxed">{result.data.summary.short}</p>
                      </div>
                      <div className="glass-card p-8">
                        <div className="flex items-center gap-2 mb-4 text-indigo-400">
                          <Layers className="w-4 h-4" />
                          <h4 className="text-xs font-bold uppercase tracking-widest">Medium Explanation</h4>
                        </div>
                        <p className="text-sm text-slate-400 leading-relaxed">{result.data.summary.medium}</p>
                      </div>
                      <div className="glass-card p-8">
                        <div className="flex items-center gap-2 mb-4 text-purple-400">
                          <Sparkles className="w-4 h-4" />
                          <h4 className="text-xs font-bold uppercase tracking-widest">Detailed Explanation</h4>
                        </div>
                        <p className="text-sm text-slate-400 leading-relaxed">{result.data.summary.detailed}</p>
                      </div>
                    </div>

                    {/* Learning Modes */}
                    <div className="glass-card p-8">
                      <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-6 flex items-center gap-2">
                        <Brain className="w-4 h-4" />
                        Learning Modes
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        <div>
                          <span className="text-[10px] font-black uppercase tracking-widest text-indigo-400 mb-2 block">Beginner</span>
                          <p className="text-sm text-slate-400 leading-relaxed">{result.data.learning_mode.beginner}</p>
                        </div>
                        <div>
                          <span className="text-[10px] font-black uppercase tracking-widest text-purple-400 mb-2 block">Intermediate</span>
                          <p className="text-sm text-slate-400 leading-relaxed">{result.data.learning_mode.intermediate}</p>
                        </div>
                        <div>
                          <span className="text-[10px] font-black uppercase tracking-widest text-pink-400 mb-2 block">Expert</span>
                          <p className="text-sm text-slate-400 leading-relaxed">{result.data.learning_mode.expert}</p>
                        </div>
                      </div>
                    </div>

                    {/* Flags & Speed */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="glass-card p-8 border-red-500/10">
                        <div className="flex items-center gap-3 mb-6">
                          <AlertTriangle className="w-5 h-5 text-amber-500" />
                          <h4 className="text-sm font-bold text-white uppercase tracking-widest">Hallucination Flags</h4>
                        </div>
                        <p className="text-slate-400 leading-relaxed">{result.data.hallucination_flags}</p>
                      </div>
                      <div className="glass-card p-8 border-emerald-500/10">
                        <div className="flex items-center gap-3 mb-6">
                          <Zap className="w-5 h-5 text-emerald-500" />
                          <h4 className="text-sm font-bold text-white uppercase tracking-widest">Speed Winner</h4>
                        </div>
                        <p className="text-slate-400 leading-relaxed">{result.data.speed_winner}</p>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center justify-between pt-8">
                      <div className="flex items-center gap-4">
                        <button className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 rounded-xl transition-all text-sm font-medium">
                          <ThumbsUp className="w-4 h-4" />
                          {t.helpful}
                        </button>
                        <button className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 rounded-xl transition-all text-sm font-medium">
                          <ThumbsDown className="w-4 h-4" />
                          {t.incorrect}
                        </button>
                      </div>
                      <div className="flex items-center gap-4">
                        <button 
                          onClick={exportResult}
                          className="flex items-center gap-2 px-6 py-3 bg-indigo-600 hover:bg-indigo-500 rounded-xl transition-all text-sm font-bold text-white shadow-lg shadow-indigo-500/20"
                        >
                          <Download className="w-4 h-4" />
                          {t.exportReport}
                        </button>
                        <button 
                          onClick={handleReadAloud}
                          disabled={isReadingAloud}
                          className={cn(
                            "flex items-center gap-2 px-6 py-3 bg-white/5 hover:bg-white/10 rounded-xl transition-all text-sm font-bold text-slate-300 disabled:opacity-50",
                            isReadingAloud && "animate-pulse"
                          )}
                        >
                          {isReadingAloud ? <Loader2 className="w-4 h-4 animate-spin" /> : <Volume2 className="w-4 h-4" />}
                          {isReadingAloud ? t.reading : t.readAloud}
                        </button>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}
        </div>
      </main>

      {/* History Sidebar */}
      <AnimatePresence>
        {showHistory && (
          <>
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowHistory(false)}
              className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[60]"
            />
            <motion.div 
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              className="fixed top-0 right-0 bottom-0 w-full max-w-md bg-[#0a0a0a] border-l border-white/10 z-[70] p-8 shadow-2xl"
            >
              <div className="flex items-center justify-between mb-12">
                <h2 className="text-2xl font-display font-bold text-white">{t.queryHistory}</h2>
                <button 
                  onClick={() => setShowHistory(false)}
                  className="p-2 hover:bg-white/5 rounded-full transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="space-y-4 overflow-y-auto max-h-[calc(100vh-200px)] scrollbar-hide">
                {history.length === 0 ? (
                  <div className="text-center py-20">
                    <History className="w-12 h-12 text-slate-700 mx-auto mb-4" />
                    <p className="text-slate-500">{t.noQueriesYet}</p>
                  </div>
                ) : (
                  history.map((item) => (
                    <button
                      key={item.id}
                      onClick={() => {
                        setResult(item);
                        setShowHistory(false);
                      }}
                      className="w-full text-left p-6 glass-card glass-card-hover group"
                    >
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest">{item.domain === 'General' ? t.domainGeneral : item.domain === 'Math' ? t.domainMath : item.domain === 'Coding' ? t.domainCoding : t.domainTheory}</span>
                        <span className="text-[10px] text-slate-500">{new Date(item.timestamp).toLocaleDateString()}</span>
                      </div>
                      <p className="text-sm text-white font-medium line-clamp-2 mb-4 group-hover:text-indigo-300 transition-colors">
                        {item.question}
                      </p>
                      <div className="flex items-center gap-2">
                        <div className="flex -space-x-2">
                          {[1, 2, 3].map(i => (
                            <div key={i} className="w-6 h-6 rounded-full bg-slate-800 border-2 border-[#0a0a0a] flex items-center justify-center">
                              <Zap className="w-3 h-3 text-indigo-400" />
                            </div>
                          ))}
                        </div>
                        <span className="text-[10px] text-slate-500 font-bold uppercase tracking-tighter">
                          {item.data.confidence_score} {t.agreement}
                        </span>
                      </div>
                    </button>
                  ))
                )}
              </div>

              {history.length > 0 && (
                <button 
                  onClick={() => {
                    setHistory([]);
                    localStorage.removeItem('omnijudge_history');
                  }}
                  className="absolute bottom-8 left-8 right-8 py-4 border border-red-500/20 text-red-500 rounded-2xl text-sm font-bold hover:bg-red-500/5 transition-all"
                >
                  {t.clearAllHistory}
                </button>
              )}
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Footer */}
      <footer className="max-w-7xl mx-auto px-6 py-12 border-t border-white/5 mt-20">
        <div className="flex flex-col md:flex-row items-center justify-between gap-8">
          <div className="flex items-center gap-3 opacity-50">
            <Brain className="w-5 h-5" />
            <span className="text-sm font-medium">FAAI v2.4.0</span>
          </div>
          <div className="flex items-center gap-8 text-sm text-slate-500 font-medium">
            <a href="#" className="hover:text-white transition-colors">{t.documentation}</a>
            <a href="#" className="hover:text-white transition-colors">{t.apiAccess}</a>
            <a href="#" className="hover:text-white transition-colors">{t.privacy}</a>
            <a href="#" className="hover:text-white transition-colors">{t.terms}</a>
          </div>
          <div className="flex items-center gap-4">
            <div className="px-3 py-1 bg-white/5 rounded-lg border border-white/10 text-[10px] font-bold text-slate-400 uppercase tracking-widest">
              {t.poweredBy}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
