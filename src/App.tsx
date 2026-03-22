/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { GoogleGenAI, Type } from "@google/genai";
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
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

import { TRANSLATIONS } from './translations';

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

interface ModelResponse {
  modelName: string;
  content: string;
  accuracyScore: number;
  confidence: 'High' | 'Medium' | 'Low';
  reasoning: string;
  steps: { title: string; content: string; status: 'correct' | 'flawed' | 'neutral' }[];
}

interface ConsensusReport {
  finalAnswer: string;
  agreementScore: number; // 0-100
  agreementCount: string; // e.g. "2/3"
  judgeReasoning: string;
  bestModel: string;
  simplifiedExplanation: string;
  detailedExplanation: string;
  realLifeExample: string;
  disagreementAnalysis: string;
}

interface HistoryItem {
  id: string;
  timestamp: number;
  question: string;
  domain: Domain;
  responses: ModelResponse[];
  consensus: ConsensusReport;
}

// --- Constants ---
const MODELS = ['Gemini 3.1 Pro', 'GPT-4o (Simulated)', 'Claude 3.5 Sonnet (Simulated)'];

const SYSTEM_PROMPT = (domain: Domain, mode: Mode, language: Language) => `
You are the "FAAI Consensus Engine". Your goal is to provide a multi-perspective analysis of a user's query.

DOMAIN: ${domain}
MODE: ${mode}
LANGUAGE: ${language} (IMPORTANT: All responses and the consensus report MUST be in this language)

TASK:
1. Generate 3 distinct responses to the user's query as if they were from different top-tier AI models:
   - "Gemini 3.1 Pro": Focus on multimodal reasoning and factual precision.
   - "GPT-4o (Simulated)": Focus on conversational flow and broad knowledge.
   - "Claude 3.5 Sonnet (Simulated)": Focus on safety, nuance, and code/theory depth.

2. Act as an "AI Judge" to evaluate these 3 responses:
   - Assign an Accuracy Score (0-100) to each.
   - Assign a Confidence Level (High/Medium/Low).
   - Break each down into logical steps.
   - Identify any logical flaws or errors.

3. Formulate a "Consensus Report":
   - Provide the single most verified "Final Answer".
   - Calculate an Agreement Score (how much do the models overlap?).
   - Explain WHY they might disagree (assumptions, training data bias).
   - Provide a simplified explanation, a detailed one, and a real-life example.

OUTPUT FORMAT (STRICT JSON):
{
  "responses": [
    {
      "modelName": "string",
      "content": "string",
      "accuracyScore": number,
      "confidence": "High" | "Medium" | "Low",
      "reasoning": "string",
      "steps": [{ "title": "string", "content": "string", "status": "correct" | "flawed" | "neutral" }]
    }
  ],
  "consensus": {
    "finalAnswer": "string",
    "agreementScore": number,
    "agreementCount": "string",
    "judgeReasoning": "string",
    "bestModel": "string",
    "simplifiedExplanation": "string",
    "detailedExplanation": "string",
    "realLifeExample": "string",
    "disagreementAnalysis": "string"
  }
}
`;

// --- Components ---

export default function App() {
  // State
  const [query, setQuery] = useState('');
  const [image, setImage] = useState<string | null>(null);
  const [domain, setDomain] = useState<Domain>('General');
  const [mode, setMode] = useState<Mode>('Deep');
  const [language, setLanguage] = useState<Language>('English');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<HistoryItem | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [activeTab, setActiveTab] = useState<'compare' | 'consensus' | 'history'>('compare');
  const [showHistory, setShowHistory] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [settingsView, setSettingsView] = useState<'main' | 'language'>('main');

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

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const runConsensus = async () => {
    if (!query && !image) return;

    setLoading(true);
    setResult(null);
    setActiveTab('compare');

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

      let contents: any[] = [{ text: query }];
      if (image) {
        contents.push({
          inlineData: {
            mimeType: "image/jpeg",
            data: image.split(',')[1]
          }
        });
      }

      const response = await ai.models.generateContent({
        model: "gemini-3.1-pro-preview",
        contents: { parts: contents },
        config: {
          systemInstruction: SYSTEM_PROMPT(domain, mode, language),
          responseMimeType: "application/json",
        }
      });

      const data = JSON.parse(response.text);
      
      const newItem: HistoryItem = {
        id: Math.random().toString(36).substr(2, 9),
        timestamp: Date.now(),
        question: query || t.imageQuery,
        domain,
        responses: data.responses,
        consensus: data.consensus
      };

      setResult(newItem);
      setHistory(prev => [newItem, ...prev].slice(0, 20));
    } catch (error) {
      console.error("Consensus Error:", error);
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
                    className="absolute right-0 mt-2 w-64 glass-card p-2 z-[100] border-white/10 shadow-2xl overflow-hidden"
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
                        <button className="w-full flex items-center gap-3 px-3 py-2.5 hover:bg-white/5 rounded-xl transition-colors">
                          <div className="w-8 h-8 rounded-lg bg-purple-500/10 flex items-center justify-center">
                            <Zap className="w-4 h-4 text-purple-400" />
                          </div>
                          <span className="text-sm font-medium text-slate-200">{t.systemStatus}</span>
                        </button>
                      </div>
                    ) : (
                      <div className="space-y-1">
                        <div className="flex items-center gap-2 px-1 mb-1">
                          <button 
                            onClick={() => setSettingsView('main')}
                            className="p-2 hover:bg-white/5 rounded-lg transition-colors"
                          >
                            <X className="w-4 h-4 rotate-180" />
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
              </div>

              {/* Input Area */}
              <div className="relative group">
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={t.inputPlaceholder}
                  className="w-full bg-black/40 border border-white/10 rounded-2xl p-6 h-48 text-lg text-white placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all resize-none"
                />
                
                {image && (
                  <div className="absolute bottom-4 left-4 group/img">
                    <img src={image} alt="Upload" className="w-20 h-20 object-cover rounded-xl border-2 border-indigo-500/50 shadow-xl" />
                    <button 
                      onClick={() => setImage(null)}
                      className="absolute -top-2 -right-2 bg-red-500 text-white p-1 rounded-full opacity-0 group-hover/img:opacity-100 transition-opacity"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                )}

                <div className="absolute bottom-4 right-4 flex items-center gap-2">
                  <button 
                    onClick={() => fileInputRef.current?.click()}
                    className="p-3 bg-white/5 hover:bg-white/10 rounded-xl transition-colors text-slate-400 hover:text-white"
                  >
                    <ImageIcon className="w-5 h-5" />
                  </button>
                  <button className="p-3 bg-white/5 hover:bg-white/10 rounded-xl transition-colors text-slate-400 hover:text-white">
                    <Mic className="w-5 h-5" />
                  </button>
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    onChange={handleImageUpload} 
                    className="hidden" 
                    accept="image/*" 
                  />
                </div>
              </div>

              <button
                onClick={runConsensus}
                disabled={loading || (!query && !image)}
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
          {(result || loading) && (
            <div className="lg:col-span-8 space-y-8">
              {/* Tabs */}
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

              <AnimatePresence mode="wait">
                {activeTab === 'compare' && result && (
                  <motion.div 
                    key="compare"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    className="grid grid-cols-1 md:grid-cols-3 gap-6"
                  >
                    {result.responses.map((resp, i) => (
                      <div key={i} className="glass-card p-6 flex flex-col h-full relative overflow-hidden">
                        {/* Accuracy Badge */}
                        <div className="absolute top-0 right-0 p-4">
                          <div className={cn(
                            "px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest",
                            resp.accuracyScore > 85 ? "bg-emerald-500/20 text-emerald-400" : 
                            resp.accuracyScore > 70 ? "bg-amber-500/20 text-amber-400" : 
                            "bg-red-500/20 text-red-400"
                          )}>
                            {resp.accuracyScore}% {t.accuracy}
                          </div>
                        </div>

                        <div className="flex items-center gap-3 mb-6">
                          <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center">
                            <Zap className="w-4 h-4 text-indigo-400" />
                          </div>
                          <h3 className="font-display font-bold text-white">{resp.modelName}</h3>
                        </div>

                        <div className="flex-grow overflow-y-auto max-h-[300px] scrollbar-hide mb-6">
                          <div className="markdown-body text-sm">
                            <Markdown>{resp.content}</Markdown>
                          </div>
                        </div>

                        <div className="pt-6 border-t border-white/5 space-y-4">
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-slate-500 font-medium">{t.confidence}</span>
                            <span className={cn(
                              "font-bold uppercase tracking-tighter",
                              resp.confidence === 'High' ? "text-emerald-400" : 
                              resp.confidence === 'Medium' ? "text-amber-400" : 
                              "text-red-400"
                            )}>
                              {resp.confidence === 'High' ? t.confidenceHigh : resp.confidence === 'Medium' ? t.confidenceMedium : t.confidenceLow}
                            </span>
                          </div>
                          
                          {/* Mini Steps */}
                          <div className="space-y-2">
                            {resp.steps.slice(0, 3).map((step, si) => (
                              <div key={si} className="flex items-center gap-2 text-[10px]">
                                {step.status === 'correct' ? <CheckCircle2 className="w-3 h-3 text-emerald-500" /> :
                                 step.status === 'flawed' ? <AlertTriangle className="w-3 h-3 text-red-500" /> :
                                 <HelpCircle className="w-3 h-3 text-slate-500" />}
                                <span className="text-slate-400 truncate">{step.title}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))}
                  </motion.div>
                )}

                {activeTab === 'consensus' && result && (
                  <motion.div 
                    key="consensus"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    className="space-y-8"
                  >
                    {/* Final Answer Hero */}
                    <div className="glass-card p-10 bg-gradient-to-br from-indigo-600/10 to-purple-600/10 border-indigo-500/20">
                      <div className="flex items-center gap-3 mb-8">
                        <div className="w-12 h-12 bg-indigo-500 rounded-2xl flex items-center justify-center shadow-xl shadow-indigo-500/20">
                          <Trophy className="w-6 h-6 text-white" />
                        </div>
                        <div>
                          <h2 className="text-2xl font-display font-bold text-white">{t.verifiedConsensus}</h2>
                          <div className="flex items-center gap-2 text-indigo-400 text-sm font-medium">
                            <CheckCircle2 className="w-4 h-4" />
                            {result.consensus.agreementCount} {t.modelsAgree}
                          </div>
                        </div>
                      </div>

                      <div className="markdown-body text-lg text-white leading-relaxed mb-10">
                        <Markdown>{result.consensus.finalAnswer}</Markdown>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="p-6 bg-white/5 rounded-2xl border border-white/5">
                          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">{t.agreementMeter}</h4>
                          <div className="flex items-end gap-4">
                            <div className="text-5xl font-black text-white">{result.consensus.agreementScore}%</div>
                            <div className="flex-grow h-3 bg-white/5 rounded-full overflow-hidden mb-2">
                              <motion.div 
                                initial={{ width: 0 }}
                                animate={{ width: `${result.consensus.agreementScore}%` }}
                                transition={{ duration: 1.5, ease: "easeOut" }}
                                className="h-full bg-gradient-to-r from-indigo-500 to-purple-500"
                              />
                            </div>
                          </div>
                        </div>
                        <div className="p-6 bg-white/5 rounded-2xl border border-white/5">
                          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">{t.judgeAnalysis}</h4>
                          <p className="text-sm text-slate-400 leading-relaxed italic">
                            "{result.consensus.judgeReasoning}"
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Enhanced Explanations */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      <div className="glass-card p-8">
                        <div className="flex items-center gap-2 mb-4 text-emerald-400">
                          <Zap className="w-4 h-4" />
                          <h4 className="text-xs font-bold uppercase tracking-widest">{t.simplified}</h4>
                        </div>
                        <p className="text-sm text-slate-400 leading-relaxed">{result.consensus.simplifiedExplanation}</p>
                      </div>
                      <div className="glass-card p-8">
                        <div className="flex items-center gap-2 mb-4 text-indigo-400">
                          <Layers className="w-4 h-4" />
                          <h4 className="text-xs font-bold uppercase tracking-widest">{t.detailed}</h4>
                        </div>
                        <p className="text-sm text-slate-400 leading-relaxed">{result.consensus.detailedExplanation}</p>
                      </div>
                      <div className="glass-card p-8">
                        <div className="flex items-center gap-2 mb-4 text-purple-400">
                          <Sparkles className="w-4 h-4" />
                          <h4 className="text-xs font-bold uppercase tracking-widest">{t.realLife}</h4>
                        </div>
                        <p className="text-sm text-slate-400 leading-relaxed">{result.consensus.realLifeExample}</p>
                      </div>
                    </div>

                    {/* Disagreement Analysis */}
                    <div className="glass-card p-8 border-red-500/10">
                      <div className="flex items-center gap-3 mb-6">
                        <AlertTriangle className="w-5 h-5 text-amber-500" />
                        <h4 className="text-sm font-bold text-white uppercase tracking-widest">{t.whyModelsDisagree}</h4>
                      </div>
                      <p className="text-slate-400 leading-relaxed">{result.consensus.disagreementAnalysis}</p>
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
                        <button className="flex items-center gap-2 px-6 py-3 bg-white/5 hover:bg-white/10 rounded-xl transition-all text-sm font-bold text-slate-300">
                          <Volume2 className="w-4 h-4" />
                          {t.readAloud}
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
                          {item.consensus.agreementScore}% {t.agreement}
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
