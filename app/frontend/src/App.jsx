import React, { useState, useEffect } from "react";
import {
  Shield, Search, Brain, BarChart3, Users, Clock, MapPin,
  Flame, Award, RefreshCw, ChevronRight, CheckCircle2, XCircle, Info
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, PieChart, Pie, Legend, LineChart, Line
} from "recharts";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

// Soft peach-red and mist-white custom chart colors
const COLORS = [
  "#FA5252", "#FF8787", "#FFA8A8", "#FFC9C9", "#FFE3E3",
  "#E03131", "#C92A2A", "#FF6B6B", "#FF8787", "#FFA8A8"
];

function App() {
  const [activeTab, setActiveTab] = useState("overview");

  // Data states
  const [metrics, setMetrics] = useState(null);
  const [analytics, setAnalytics] = useState(null);

  // Search state
  const [searchId, setSearchId] = useState("");
  const [searchSuggestions, setSearchSuggestions] = useState([]);
  const [searchCity, setSearchCity] = useState("All");
  const [searchWeapon, setSearchWeapon] = useState("All");
  const [searchTopK, setSearchTopK] = useState(10);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [selectedSearchCase, setSelectedSearchCase] = useState(null);
  const [baseCaseDetails, setBaseCaseDetails] = useState(null);

  // Ranking state
  const [rankingId, setRankingId] = useState("");
  const [rankingSuggestions, setRankingSuggestions] = useState([]);
  const [rankingTopK, setRankingTopK] = useState(10);
  const [rankingLoading, setRankingLoading] = useState(false);
  const [rankingResults, setRankingResults] = useState([]);
  const [selectedRankCase, setSelectedRankCase] = useState(null);
  const [baseCaseRankDetails, setBaseCaseRankDetails] = useState(null);

  // Training state
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const [showTrainingPanel, setShowTrainingPanel] = useState(false);

  // Fetch initial metrics and analytics
  useEffect(() => {
    fetchMetrics();
    fetchAnalytics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch (err) {
      console.error("Error fetching metrics:", err);
    }
  };

  const fetchAnalytics = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/analytics`);
      if (res.ok) {
        const data = await res.json();
        setAnalytics(data);
      }
    } catch (err) {
      console.error("Error fetching analytics:", err);
    }
  };

  // Case ID suggestions lookup
  const handleLookup = async (query, type) => {
    if (type === "search") {
      setSearchId(query);
      if (query.length < 3) {
        setSearchSuggestions([]);
        return;
      }
      try {
        const res = await fetch(`${API_BASE}/api/cases/search/lookup?q=${query}`);
        if (res.ok) {
          const list = await res.json();
          setSearchSuggestions(list);
        }
      } catch (err) {
        console.error(err);
      }
    } else {
      setRankingId(query);
      if (query.length < 3) {
        setRankingSuggestions([]);
        return;
      }
      try {
        const res = await fetch(`${API_BASE}/api/cases/search/lookup?q=${query}`);
        if (res.ok) {
          const list = await res.json();
          setRankingSuggestions(list);
        }
      } catch (err) {
        console.error(err);
      }
    }
  };

  // Run Similarity Search
  const runSimilaritySearch = async () => {
    if (!searchId) return;
    setSearchLoading(true);
    setSelectedSearchCase(null);
    try {
      // First fetch details of base case
      const baseRes = await fetch(`${API_BASE}/api/cases/${searchId}`);
      if (baseRes.ok) {
        const baseData = await baseRes.json();
        setBaseCaseDetails(baseData);
      } else {
        alert("Case ID not found in database.");
        setSearchLoading(false);
        return;
      }

      // Fetch similar cases
      const res = await fetch(
        `${API_BASE}/api/similarity?case_id=${searchId}&top_k=${searchTopK}&city=${searchCity}&weapon=${searchWeapon}`
      );
      if (res.ok) {
        const list = await res.json();
        setSearchResults(list);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setSearchLoading(false);
    }
  };

  // Run Suspect Ranking
  const runSuspectRanking = async () => {
    if (!rankingId) return;
    setRankingLoading(true);
    setSelectedRankCase(null);
    try {
      // First fetch base case details
      const baseRes = await fetch(`${API_BASE}/api/cases/${rankingId}`);
      if (baseRes.ok) {
        const baseData = await baseRes.json();
        setBaseCaseRankDetails(baseData);
      } else {
        alert("Case ID not found in database.");
        setRankingLoading(false);
        return;
      }

      // Fetch rankings
      const res = await fetch(`${API_BASE}/api/ranking?case_id=${rankingId}&top_k=${rankingTopK}`);
      if (res.ok) {
        const list = await res.json();
        setRankingResults(list);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setRankingLoading(false);
    }
  };

  // Train Ensemble Model
  const trainEnsembleModel = async () => {
    setTrainingLoading(true);
    setTrainingResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/model/train`, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        setTrainingResult(data);
        fetchMetrics(); // Refresh stats
      }
    } catch (err) {
      console.error(err);
    } finally {
      setTrainingLoading(false);
    }
  };

  // Unique lists for filtering dropdowns (taken from metrics)
  const cities = metrics?.top_areas?.map(a => a.area) || [];
  const weapons = metrics?.top_weapons?.map(w => w.weapon) || [];

  return (
    <div className="min-h-screen bg-mist text-text-primary">
      {/* Top Navbar */}
      <header className="bg-surface border-b border-border shadow-sm sticky top-0 z-50 transition-material">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-primary p-2 rounded-lg text-white elevation-1">
              <Shield className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-text-primary">Crime Analytics & Suspect Ranking</h1>
              <p className="text-xs text-text-muted">Ensemble Learning Modus Operandi Matching System</p>
            </div>
          </div>

          <nav className="flex space-x-1">
            <button
              onClick={() => setActiveTab("overview")}
              className={`px-4 py-2 text-sm font-semibold rounded-md transition-material ${activeTab === "overview"
                  ? "bg-primary-light text-primary"
                  : "text-text-secondary hover:bg-mist-dark"
                }`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab("similarity")}
              className={`px-4 py-2 text-sm font-semibold rounded-md transition-material ${activeTab === "similarity"
                  ? "bg-primary-light text-primary"
                  : "text-text-secondary hover:bg-mist-dark"
                }`}
            >
              Similar Cases
            </button>
            <button
              onClick={() => setActiveTab("ranking")}
              className={`px-4 py-2 text-sm font-semibold rounded-md transition-material ${activeTab === "ranking"
                  ? "bg-primary-light text-primary"
                  : "text-text-secondary hover:bg-mist-dark"
                }`}
            >
              Suspect Ranking
            </button>
            <button
              onClick={() => setActiveTab("analytics")}
              className={`px-4 py-2 text-sm font-semibold rounded-md transition-material ${activeTab === "analytics"
                  ? "bg-primary-light text-primary"
                  : "text-text-secondary hover:bg-mist-dark"
                }`}
            >
              Analytics
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto p-6 space-y-6">

        {/* TAB 1: OVERVIEW */}
        {activeTab === "overview" && (
          <div className="space-y-6">
            {/* KPI Cards */}
            {metrics?.summary ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-surface p-6 rounded-xl border border-border elevation-1 hover:elevation-2 transition-material">
                  <div className="flex justify-between items-start text-text-muted mb-2">
                    <span className="text-sm font-medium uppercase tracking-wider">Total Cases</span>
                    <Shield className="h-5 w-5 text-primary" />
                  </div>
                  <h2 className="text-3xl font-extrabold text-text-primary">{metrics.summary.total_cases.toLocaleString()}</h2>
                  <p className="text-xs text-text-muted mt-1">Processed crimes in database</p>
                </div>

                <div className="bg-surface p-6 rounded-xl border border-border elevation-1 hover:elevation-2 transition-material">
                  <div className="flex justify-between items-start text-text-muted mb-2">
                    <span className="text-sm font-medium uppercase tracking-wider">Distinct Areas</span>
                    <MapPin className="h-5 w-5 text-primary" />
                  </div>
                  <h2 className="text-3xl font-extrabold text-text-primary">{metrics.summary.distinct_areas}</h2>
                  <p className="text-xs text-text-muted mt-1">LA policing divisions</p>
                </div>

                <div className="bg-surface p-6 rounded-xl border border-border elevation-1 hover:elevation-2 transition-material">
                  <div className="flex justify-between items-start text-text-muted mb-2">
                    <span className="text-sm font-medium uppercase tracking-wider">Crime Types</span>
                    <Flame className="h-5 w-5 text-primary" />
                  </div>
                  <h2 className="text-3xl font-extrabold text-text-primary">{metrics.summary.crime_types}</h2>
                  <p className="text-xs text-text-muted mt-1">Distinct crime descriptions</p>
                </div>

                <div className="bg-surface p-6 rounded-xl border border-border elevation-1 hover:elevation-2 transition-material">
                  <div className="flex justify-between items-start text-text-muted mb-2">
                    <span className="text-sm font-medium uppercase tracking-wider">Time Span</span>
                    <Clock className="h-5 w-5 text-primary" />
                  </div>
                  <h2 className="text-3xl font-extrabold text-text-primary">{metrics.summary.time_span_days.toLocaleString()} Days</h2>
                  <p className="text-xs text-text-muted mt-1">
                    {metrics.summary.min_date} to {metrics.summary.max_date}
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-text-muted">Loading metrics...</div>
            )}

            {/* Quick Insights Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Top Areas */}
              <div className="bg-surface p-6 rounded-xl border border-border elevation-1">
                <h3 className="text-lg font-bold mb-4 text-text-primary flex items-center space-x-2">
                  <span>Top 5 Crime Areas</span>
                </h3>
                <div className="space-y-3">
                  {metrics?.top_areas?.map((item, idx) => (
                    <div key={idx} className="flex justify-between items-center py-2 border-b border-mist-dark last:border-0">
                      <div className="flex items-center space-x-3">
                        <span className="bg-primary-light text-primary w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold">
                          {idx + 1}
                        </span>
                        <span className="font-medium text-text-secondary">{item.area}</span>
                      </div>
                      <span className="text-sm font-bold text-text-primary bg-mist px-2.5 py-1 rounded-md">
                        {item.count.toLocaleString()} cases
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Top Weapons */}
              <div className="bg-surface p-6 rounded-xl border border-border elevation-1">
                <h3 className="text-lg font-bold mb-4 text-text-primary flex items-center space-x-2">
                  <span>Top 5 Weapons/MODUS Used</span>
                </h3>
                <div className="space-y-3">
                  {metrics?.top_weapons?.map((item, idx) => (
                    <div key={idx} className="flex justify-between items-center py-2 border-b border-mist-dark last:border-0">
                      <div className="flex items-center space-x-3 col-span-2">
                        <span className="bg-primary-light text-primary w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0">
                          {idx + 1}
                        </span>
                        <span className="font-medium text-text-secondary truncate max-w-[240px]">{item.weapon}</span>
                      </div>
                      <span className="text-sm font-bold text-text-primary bg-mist px-2.5 py-1 rounded-md shrink-0">
                        {item.count.toLocaleString()} cases
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Recent Cases Preview */}
            <div className="bg-surface rounded-xl border border-border elevation-1 overflow-hidden">
              <div className="p-6 border-b border-border">
                <h3 className="text-lg font-bold text-text-primary">Recent Identity Theft Cases</h3>
                <p className="text-xs text-text-muted mt-1">Showing 10 most recent identity theft occurrences</p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-mist text-text-secondary text-xs uppercase font-bold border-b border-border">
                      <th className="p-4">Case ID</th>
                      <th className="p-4">Date</th>
                      <th className="p-4">Area Name</th>
                      <th className="p-4">Weapon/MODUS</th>
                      <th className="p-4">Vict Age</th>
                      <th className="p-4">Vict Sex</th>
                      <th className="p-4">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border text-sm">
                    {metrics?.recent_id_theft?.map((item) => (
                      <tr key={item.dr_no} className="hover:bg-mist transition-material">
                        <td className="p-4 font-mono font-bold text-primary">{item.dr_no}</td>
                        <td className="p-4">{item.datetime}</td>
                        <td className="p-4">{item.area_name}</td>
                        <td className="p-4 truncate max-w-[200px]">{item.weapon_desc}</td>
                        <td className="p-4">{item.vict_age}</td>
                        <td className="p-4">{item.vict_sex}</td>
                        <td className="p-4">
                          <button
                            onClick={() => {
                              setSearchId(item.dr_no.toString());
                              setRankingId(item.dr_no.toString());
                              setActiveTab("similarity");
                            }}
                            className="text-xs font-semibold bg-primary-light text-primary px-3 py-1.5 rounded-md hover:bg-primary hover:text-white transition-material elevation-1"
                          >
                            Analyze
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: SIMILAR CASES */}
        {activeTab === "similarity" && (
          <div className="space-y-6">
            {/* Search Controls Card */}
            <div className="bg-surface p-6 rounded-xl border border-border elevation-1">
              <h3 className="text-lg font-bold mb-4 text-text-primary flex items-center space-x-2">
                <Search className="h-5 w-5 text-primary" />
                <span>Search Similar MODUS OPERANDI Cases</span>
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 items-end">
                {/* Case ID Input */}
                <div className="relative">
                  <label className="block text-xs font-bold text-text-secondary mb-1.5 uppercase">Base Case ID (DR_NO)</label>
                  <input
                    type="text"
                    value={searchId}
                    onChange={(e) => handleLookup(e.target.value, "search")}
                    placeholder="Type numeric ID (e.g. 211507896)"
                    className="w-full bg-mist border border-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-primary transition-material font-mono font-bold"
                  />
                  {searchSuggestions.length > 0 && (
                    <div className="absolute left-0 right-0 bg-surface border border-border mt-1 rounded-lg elevation-2 max-h-48 overflow-y-auto z-10 font-mono text-sm">
                      {searchSuggestions.map((sug) => (
                        <div
                          key={sug}
                          onClick={() => {
                            setSearchId(sug.toString());
                            setRankingId(sug.toString());
                            setSearchSuggestions([]);
                          }}
                          className="px-4 py-2 hover:bg-primary-light hover:text-primary cursor-pointer border-b border-mist-dark last:border-0 font-bold"
                        >
                          {sug}
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* City Area dropdown */}
                <div>
                  <label className="block text-xs font-bold text-text-secondary mb-1.5 uppercase">Filter Area</label>
                  <select
                    value={searchCity}
                    onChange={(e) => setSearchCity(e.target.value)}
                    className="w-full bg-mist border border-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-primary transition-material font-semibold"
                  >
                    <option value="All">All Areas</option>
                    {cities.map((city) => (
                      <option key={city} value={city}>{city}</option>
                    ))}
                  </select>
                </div>

                {/* Weapon dropdown */}
                <div>
                  <label className="block text-xs font-bold text-text-secondary mb-1.5 uppercase">Filter Weapon</label>
                  <select
                    value={searchWeapon}
                    onChange={(e) => setSearchWeapon(e.target.value)}
                    className="w-full bg-mist border border-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-primary transition-material font-semibold"
                  >
                    <option value="All">All Weapons</option>
                    {weapons.map((wpn) => (
                      <option key={wpn} value={wpn}>{wpn}</option>
                    ))}
                  </select>
                </div>

                {/* Submit button */}
                <button
                  onClick={runSimilaritySearch}
                  disabled={searchLoading}
                  className="w-full bg-primary hover:bg-primary-hover text-white font-bold py-2.5 px-4 rounded-lg flex items-center justify-center space-x-2 transition-material disabled:bg-text-muted elevation-1 hover:elevation-2"
                >
                  <Search className="h-4 w-4" />
                  <span>{searchLoading ? "Searching..." : "Find Similar Cases"}</span>
                </button>
              </div>

              {/* Slider for Top K */}
              <div className="mt-6 border-t border-border pt-4 flex flex-col md:flex-row md:items-center space-y-2 md:space-y-0 md:space-x-6">
                <span className="text-xs font-bold text-text-secondary uppercase">Results Size (Top K): {searchTopK}</span>
                <input
                  type="range"
                  min="5"
                  max="50"
                  step="5"
                  value={searchTopK}
                  onChange={(e) => setSearchTopK(parseInt(e.target.value))}
                  className="w-full md:max-w-xs accent-primary"
                />
              </div>
            </div>

            {/* Base Case Profile Card */}
            {baseCaseDetails && (
              <div className="bg-surface p-6 rounded-xl border border-border elevation-1">
                <h4 className="text-sm font-bold text-text-muted uppercase mb-3">Target Base Case Details</h4>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6 text-sm">
                  <div>
                    <span className="text-xs font-semibold text-text-muted block">Case ID</span>
                    <span className="font-mono font-bold text-primary text-base">{baseCaseDetails.dr_no}</span>
                  </div>
                  <div>
                    <span className="text-xs font-semibold text-text-muted block">Crime Type</span>
                    <span className="font-bold text-text-primary">{baseCaseDetails.crm_cd_desc}</span>
                  </div>
                  <div>
                    <span className="text-xs font-semibold text-text-muted block">Location / Area</span>
                    <span className="font-bold text-text-primary">{baseCaseDetails.area_name}</span>
                  </div>
                  <div>
                    <span className="text-xs font-semibold text-text-muted block">Date</span>
                    <span className="font-bold text-text-primary">{baseCaseDetails.datetime}</span>
                  </div>
                  <div>
                    <span className="text-xs font-semibold text-text-muted block">Weapon Description</span>
                    <span className="font-bold text-text-primary">{baseCaseDetails.weapon_desc}</span>
                  </div>
                  <div>
                    <span className="text-xs font-semibold text-text-muted block">Victim Details</span>
                    <span className="font-bold text-text-primary">{baseCaseDetails.vict_age} yrs old / {baseCaseDetails.vict_sex}</span>
                  </div>
                  <div className="md:col-span-2">
                    <span className="text-xs font-semibold text-text-muted block">Modus Operandi Text</span>
                    <span className="text-xs font-mono bg-mist p-2.5 rounded border border-border mt-1 block italic">{baseCaseDetails.mo_text}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Search Results */}
            {searchResults.length > 0 && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Data Table */}
                <div className="bg-surface rounded-xl border border-border elevation-1 overflow-hidden lg:col-span-2 flex flex-col">
                  <div className="p-6 border-b border-border">
                    <h3 className="text-lg font-bold text-text-primary">Matching Modus Operandi Cases</h3>
                    <p className="text-xs text-text-muted mt-1">Found {searchResults.length} similar cases. Select any case to inspect details.</p>
                  </div>
                  <div className="overflow-x-auto flex-grow">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="bg-mist text-text-secondary text-xs uppercase font-bold border-b border-border">
                          <th className="p-4">Case ID</th>
                          <th className="p-4">Area</th>
                          <th className="p-4">Crime Category</th>
                          <th className="p-4">Similarity</th>
                          <th className="p-4">Inspection</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border text-sm">
                        {searchResults.map((row) => (
                          <tr
                            key={row.dr_no}
                            onClick={() => setSelectedSearchCase(row)}
                            className={`cursor-pointer transition-material ${selectedSearchCase?.dr_no === row.dr_no
                                ? "bg-primary-light"
                                : "hover:bg-mist"
                              }`}
                          >
                            <td className="p-4 font-mono font-bold text-text-primary">{row.dr_no}</td>
                            <td className="p-4">{row.area_name}</td>
                            <td className="p-4 truncate max-w-[200px]">{row.crm_cd_desc}</td>
                            <td className="p-4 font-bold text-primary">{row.similarity.toFixed(4)}</td>
                            <td className="p-4">
                              <span className="text-xs font-bold text-primary flex items-center space-x-1">
                                <span>Inspect</span>
                                <ChevronRight className="h-3 w-3" />
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Side Inspection Card & Mini Distribution Chart */}
                <div className="space-y-6">
                  {/* Selected Inspect Case details */}
                  <div className="bg-surface p-6 rounded-xl border border-border elevation-1">
                    <h3 className="text-lg font-bold mb-4 text-text-primary flex items-center space-x-2">
                      <Info className="h-5 w-5 text-primary" />
                      <span>Case Details Inspector</span>
                    </h3>

                    {selectedSearchCase ? (
                      <div className="space-y-4 text-sm">
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <span className="text-xs text-text-muted font-bold block">Case ID</span>
                            <span className="font-mono font-bold text-base text-primary">{selectedSearchCase.dr_no}</span>
                          </div>
                          <div>
                            <span className="text-xs text-text-muted font-bold block">Match Similarity</span>
                            <span className="font-bold text-base text-green-600">{selectedSearchCase.similarity.toFixed(4)}</span>
                          </div>
                          <div>
                            <span className="text-xs text-text-muted font-bold block">Date Occurred</span>
                            <span className="font-semibold text-text-secondary">{selectedSearchCase.datetime}</span>
                          </div>
                          <div>
                            <span className="text-xs text-text-muted font-bold block">Area</span>
                            <span className="font-semibold text-text-secondary">{selectedSearchCase.area_name}</span>
                          </div>
                        </div>

                        <div>
                          <span className="text-xs text-text-muted font-bold block">Crime Type</span>
                          <span className="font-semibold text-text-secondary block">{selectedSearchCase.crm_cd_desc}</span>
                        </div>
                        <div>
                          <span className="text-xs text-text-muted font-bold block">Weapon / MODUS Used</span>
                          <span className="font-semibold text-text-secondary block">{selectedSearchCase.weapon_desc}</span>
                        </div>
                        <div>
                          <span className="text-xs text-text-muted font-bold block">Victim Details</span>
                          <span className="font-semibold text-text-secondary block">{selectedSearchCase.vict_age} yrs old / {selectedSearchCase.vict_sex}</span>
                        </div>
                        <div className="border-t border-border pt-3">
                          <span className="text-xs text-text-muted font-bold block">Modus Operandi Text</span>
                          <span className="text-xs font-mono italic bg-mist p-2.5 rounded border border-border block mt-1 leading-relaxed">
                            {selectedSearchCase.mo_text}
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-12 text-text-muted border border-dashed border-border rounded-lg">
                        Select a case from the list on the left to see details here.
                      </div>
                    )}
                  </div>

                  {/* Histogram Chart */}
                  <div className="bg-surface p-6 rounded-xl border border-border elevation-1">
                    <h4 className="text-sm font-bold mb-3 text-text-primary uppercase tracking-wider">Similarity Score Distribution</h4>
                    <div className="h-44">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={searchResults}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E9ECEF" />
                          <XAxis dataKey="dr_no" hide />
                          <YAxis domain={[0, 1]} tickLine={false} axisLine={false} style={{ fontSize: 10, fill: "#868E96" }} />
                          <Tooltip formatter={(value) => [value.toFixed(4), "Similarity"]} />
                          <Bar dataKey="similarity" fill="#FA5252" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

              </div>
            )}
          </div>
        )}

        {/* TAB 3: SUSPECT RANKING */}
        {activeTab === "ranking" && (
          <div className="space-y-6">
            {/* Control Inputs */}
            <div className="bg-surface p-6 rounded-xl border border-border elevation-1">
              <h3 className="text-lg font-bold mb-4 text-text-primary flex items-center space-x-2">
                <Brain className="h-5 w-5 text-primary" />
                <span>Evaluate Suspect Rankings (Ensemble Model)</span>
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-end">
                {/* Input case */}
                <div className="relative">
                  <label className="block text-xs font-bold text-text-secondary mb-1.5 uppercase">Base Case ID (DR_NO) to Rank</label>
                  <input
                    type="text"
                    value={rankingId}
                    onChange={(e) => handleLookup(e.target.value, "ranking")}
                    placeholder="Type numeric ID (e.g. 211507896)"
                    className="w-full bg-mist border border-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-primary transition-material font-mono font-bold"
                  />
                  {rankingSuggestions.length > 0 && (
                    <div className="absolute left-0 right-0 bg-surface border border-border mt-1 rounded-lg elevation-2 max-h-48 overflow-y-auto z-10 font-mono text-sm">
                      {rankingSuggestions.map((sug) => (
                        <div
                          key={sug}
                          onClick={() => {
                            setRankingId(sug.toString());
                            setSearchId(sug.toString());
                            setRankingSuggestions([]);
                          }}
                          className="px-4 py-2 hover:bg-primary-light hover:text-primary cursor-pointer border-b border-mist-dark last:border-0 font-bold"
                        >
                          {sug}
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Slider */}
                <div>
                  <div className="flex justify-between items-center mb-1.5">
                    <label className="text-xs font-bold text-text-secondary uppercase">Top K Ranked Suspects: {rankingTopK}</label>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="50"
                    step="5"
                    value={rankingTopK}
                    onChange={(e) => setRankingTopK(parseInt(e.target.value))}
                    className="w-full accent-primary"
                  />
                </div>

                {/* Submit button */}
                <button
                  onClick={runSuspectRanking}
                  disabled={rankingLoading}
                  className="w-full bg-primary hover:bg-primary-hover text-white font-bold py-2.5 px-4 rounded-lg flex items-center justify-center space-x-2 transition-material disabled:bg-text-muted elevation-1 hover:elevation-2"
                >
                  <Brain className="h-4 w-4" />
                  <span>{rankingLoading ? "Analyzing..." : "⚡ Run Suspect Ranking"}</span>
                </button>
              </div>

              {/* Collapsable Model training drawer button */}
              <div className="mt-6 border-t border-border pt-4 flex justify-between items-center">
                <p className="text-xs text-text-muted">
                  The ranker uses Random Forest + Gradient Boosting classifiers to calculate probabilities of crime pattern replication.
                </p>
                <button
                  onClick={() => setShowTrainingPanel(!showTrainingPanel)}
                  className="text-xs font-bold bg-mist hover:bg-mist-dark text-text-secondary border border-border px-3 py-1.5 rounded-lg transition-material flex items-center space-x-1.5"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  <span>{showTrainingPanel ? "Hide Training Tools" : "Retrain & Model Metrics"}</span>
                </button>
              </div>
            </div>

            {/* Model Retraining Metrics panel */}
            {showTrainingPanel && (
              <div className="bg-surface p-6 rounded-xl border border-border elevation-1 space-y-6">
                <div className="flex flex-col sm:flex-row justify-between sm:items-center border-b border-border pb-4 gap-4">
                  <div>
                    <h4 className="text-base font-bold text-text-primary">Ensemble Model Retraining & Accuracy Diagnostics</h4>
                    <p className="text-xs text-text-muted mt-1">Triggers Random Forest + Gradient Boosting fitting on case pairwise combinations and reports metrics.</p>
                  </div>
                  <button
                    onClick={trainEnsembleModel}
                    disabled={trainingLoading}
                    className="bg-primary hover:bg-primary-hover text-white font-bold py-2 px-4 rounded-lg text-sm flex items-center space-x-1.5 transition-material disabled:bg-text-muted elevation-1"
                  >
                    <RefreshCw className={`h-4 w-4 ${trainingLoading ? "animate-spin" : ""}`} />
                    <span>{trainingLoading ? "Fitting Model..." : "Retrain Model Now"}</span>
                  </button>
                </div>

                {trainingResult && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 text-sm">
                    {/* Metrics Comparison */}
                    <div className="space-y-4">
                      <h5 className="font-bold text-text-primary uppercase tracking-wider text-xs">Accuracy Metrics Comparison (Baseline Similarity vs Ensemble Ranker)</h5>
                      <div className="overflow-x-auto border border-border rounded-lg">
                        <table className="w-full text-left border-collapse">
                          <thead>
                            <tr className="bg-mist text-text-secondary text-xs uppercase font-bold border-b border-border">
                              <th className="p-3">K</th>
                              <th className="p-3">Base Precision</th>
                              <th className="p-3 text-primary">Ensemble Precision</th>
                              <th className="p-3">Base NDCG</th>
                              <th className="p-3 text-primary">Ensemble NDCG</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-border font-medium text-xs">
                            {trainingResult.metrics?.map((m) => (
                              <tr key={m.k} className="hover:bg-mist">
                                <td className="p-3 font-bold">K={m.k}</td>
                                <td className="p-3 text-text-secondary">{m.base_precision.toFixed(4)}</td>
                                <td className="p-3 text-primary font-bold">{m.ensemble_precision.toFixed(4)}</td>
                                <td className="p-3 text-text-secondary">{m.base_ndcg.toFixed(4)}</td>
                                <td className="p-3 text-primary font-bold">{m.ensemble_ndcg.toFixed(4)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <p className="text-xs text-text-muted">
                        * Baseline similarity uses MO text cosine similarity. The Ensemble Model combines location, time decay, weapon types, demographics, and text matches.
                      </p>
                    </div>

                    {/* Feature Importances */}
                    <div className="space-y-4">
                      <h5 className="font-bold text-text-primary uppercase tracking-wider text-xs">Ensemble Feature Importance Weights</h5>
                      <div className="space-y-2 max-h-56 overflow-y-auto pr-2">
                        {trainingResult.importances?.map((feat) => (
                          <div key={feat.feature} className="space-y-1">
                            <div className="flex justify-between items-center text-xs">
                              <span className="font-bold text-text-secondary uppercase">{feat.feature.replace("_", " ")}</span>
                              <span className="font-bold text-primary">{(feat.avg * 100).toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-mist-dark h-2 rounded overflow-hidden">
                              <div className="bg-primary h-full rounded" style={{ width: `${feat.avg * 100}%` }}></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Base Case Profile Card */}
            {baseCaseRankDetails && (
              <div className="bg-surface p-6 rounded-xl border border-border elevation-1">
                <h4 className="text-sm font-bold text-text-muted uppercase mb-3">Base Case for Ranking</h4>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6 text-sm">
                  <div>
                    <span className="text-xs font-semibold text-text-muted block">Case ID</span>
                    <span className="font-mono font-bold text-primary text-base">{baseCaseRankDetails.dr_no}</span>
                  </div>
                  <div>
                    <span className="text-xs font-semibold text-text-muted block">Crime Type</span>
                    <span className="font-bold text-text-primary">{baseCaseRankDetails.crm_cd_desc}</span>
                  </div>
                  <div>
                    <span className="text-xs font-semibold text-text-muted block">Location / Area</span>
                    <span className="font-bold text-text-primary">{baseCaseRankDetails.area_name}</span>
                  </div>
                  <div>
                    <span className="text-xs font-semibold text-text-muted block">Date</span>
                    <span className="font-bold text-text-primary">{baseCaseRankDetails.datetime}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Suspect Rankings list */}
            {rankingResults.length > 0 && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Ranked Data Table */}
                <div className="bg-surface rounded-xl border border-border elevation-1 overflow-hidden lg:col-span-2 flex flex-col">
                  <div className="p-6 border-b border-border">
                    <h3 className="text-lg font-bold text-text-primary">Ranked Potential Suspect Cases</h3>
                    <p className="text-xs text-text-muted mt-1">Top matching cases sorted by Ensemble Probability score.</p>
                  </div>
                  <div className="overflow-x-auto flex-grow">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="bg-mist text-text-secondary text-xs uppercase font-bold border-b border-border">
                          <th className="p-4">Rank</th>
                          <th className="p-4">Case ID</th>
                          <th className="p-4">Area</th>
                          <th className="p-4">Base Text Sim</th>
                          <th className="p-4 text-primary">Ensemble Score</th>
                          <th className="p-4">Inspection</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border text-sm">
                        {rankingResults.map((row, index) => (
                          <tr
                            key={row.dr_no}
                            onClick={() => setSelectedRankCase(row)}
                            className={`cursor-pointer transition-material ${selectedRankCase?.dr_no === row.dr_no
                                ? "bg-primary-light"
                                : "hover:bg-mist"
                              }`}
                          >
                            <td className="p-4 font-bold text-text-secondary">#{index + 1}</td>
                            <td className="p-4 font-mono font-bold text-text-primary">{row.dr_no}</td>
                            <td className="p-4">{row.area_name}</td>
                            <td className="p-4 text-text-secondary font-mono">{row.similarity.toFixed(4)}</td>
                            <td className="p-4 font-bold text-primary text-base">{(row.score * 100).toFixed(1)}%</td>
                            <td className="p-4">
                              <span className="text-xs font-bold text-primary flex items-center space-x-1">
                                <span>Explain</span>
                                <ChevronRight className="h-3 w-3" />
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Inspector side panel */}
                <div className="space-y-6">
                  {/* Detailed features inspection */}
                  <div className="bg-surface p-6 rounded-xl border border-border elevation-1">
                    <h3 className="text-lg font-bold mb-4 text-text-primary flex items-center space-x-2">
                      <Award className="h-5 w-5 text-primary" />
                      <span>Rank Score Explainer</span>
                    </h3>

                    {selectedRankCase ? (
                      <div className="space-y-4 text-sm">
                        <div className="flex justify-between items-center border-b border-border pb-3">
                          <div>
                            <span className="text-xs text-text-muted font-bold block">Case ID</span>
                            <span className="font-mono font-bold text-base text-primary">{selectedRankCase.dr_no}</span>
                          </div>
                          <div className="text-right">
                            <span className="text-xs text-text-muted font-bold block">Ensemble Prob</span>
                            <span className="font-bold text-xl text-primary">{(selectedRankCase.score * 100).toFixed(1)}%</span>
                          </div>
                        </div>

                        {/* Feature Match checklist */}
                        <div className="space-y-2.5">
                          <span className="text-xs text-text-muted font-bold uppercase tracking-wider block">Decision Feature Signals:</span>

                          <div className="flex items-center justify-between py-1 border-b border-mist-dark">
                            <span className="text-text-secondary">Text Modus Operandi Cosine Sim</span>
                            <span className="font-bold text-text-primary font-mono">{selectedRankCase.similarity.toFixed(4)}</span>
                          </div>

                          <div className="flex items-center justify-between py-1 border-b border-mist-dark">
                            <span className="text-text-secondary">Time Proximity Decay Score</span>
                            <span className="font-bold text-text-primary font-mono">{selectedRankCase.features.time_decay.toFixed(4)}</span>
                          </div>

                          <div className="flex items-center justify-between py-1 border-b border-mist-dark">
                            <span className="text-text-secondary">Same Area Code</span>
                            {selectedRankCase.features.area_match ? (
                              <CheckCircle2 className="h-5 w-5 text-green-500" />
                            ) : (
                              <XCircle className="h-5 w-5 text-text-muted" />
                            )}
                          </div>

                          <div className="flex items-center justify-between py-1 border-b border-mist-dark">
                            <span className="text-text-secondary">Same Weapon / MO category</span>
                            {selectedRankCase.features.weapon_match ? (
                              <CheckCircle2 className="h-5 w-5 text-green-500" />
                            ) : (
                              <XCircle className="h-5 w-5 text-text-muted" />
                            )}
                          </div>

                          <div className="flex items-center justify-between py-1 border-b border-mist-dark">
                            <span className="text-text-secondary">Same Crime Code Type</span>
                            {selectedRankCase.features.crime_code_match ? (
                              <CheckCircle2 className="h-5 w-5 text-green-500" />
                            ) : (
                              <XCircle className="h-5 w-5 text-text-muted" />
                            )}
                          </div>

                          <div className="flex items-center justify-between py-1 border-b border-mist-dark">
                            <span className="text-text-secondary">Same Victim Sex</span>
                            {selectedRankCase.features.sex_match ? (
                              <CheckCircle2 className="h-5 w-5 text-green-500" />
                            ) : (
                              <XCircle className="h-5 w-5 text-text-muted" />
                            )}
                          </div>

                          <div className="flex items-center justify-between py-1 border-b border-mist-dark">
                            <span className="text-text-secondary">Victim Age Difference</span>
                            <span className="font-bold text-text-primary">{selectedRankCase.features.age_difference.toFixed(0)} yrs</span>
                          </div>
                        </div>

                        {/* Modus operandi text */}
                        <div className="border-t border-border pt-3">
                          <span className="text-xs text-text-muted font-bold block">Modus Operandi Text</span>
                          <span className="text-xs font-mono italic bg-mist p-2.5 rounded border border-border block mt-1 leading-relaxed">
                            {selectedRankCase.mo_text}
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-12 text-text-muted border border-dashed border-border rounded-lg">
                        Select a ranked case from the list on the left to see feature breakdown here.
                      </div>
                    )}
                  </div>

                  {/* Top scores Bar chart */}
                  <div className="bg-surface p-6 rounded-xl border border-border elevation-1">
                    <h4 className="text-sm font-bold mb-3 text-text-primary uppercase tracking-wider">Top Suspect Scores Chart</h4>
                    <div className="h-44">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={rankingResults.slice(0, 8)}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E9ECEF" />
                          <XAxis dataKey="dr_no" hide />
                          <YAxis domain={[0, 1]} tickLine={false} axisLine={false} style={{ fontSize: 10, fill: "#868E96" }} />
                          <Tooltip formatter={(value) => [`${(value * 100).toFixed(1)}%`, "Probability"]} />
                          <Bar dataKey="score" fill="#FA5252" radius={[4, 4, 0, 0]}>
                            {rankingResults.slice(0, 8).map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={index === 0 ? "#E03131" : "#FF8787"} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

              </div>
            )}
          </div>
        )}

        {/* TAB 4: ANALYTICS */}
        {activeTab === "analytics" && (
          <div className="space-y-6">
            {analytics ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

                {/* 1. Monthly Trend Area Chart */}
                <div className="bg-surface p-6 rounded-xl border border-border elevation-1 md:col-span-2">
                  <h3 className="text-base font-bold mb-4 text-text-primary uppercase tracking-wider flex items-center space-x-2">
                    <BarChart3 className="h-5 w-5 text-primary" />
                    <span>Monthly Historical Crime Trend</span>
                  </h3>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={analytics.monthly_trend}>
                        <defs>
                          <linearGradient id="peachGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#FA5252" stopOpacity={0.4} />
                            <stop offset="95%" stopColor="#FA5252" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E9ECEF" />
                        <XAxis dataKey="month" style={{ fontSize: 11, fill: "#868E96" }} tickLine={false} />
                        <YAxis style={{ fontSize: 11, fill: "#868E96" }} tickLine={false} axisLine={false} />
                        <Tooltip />
                        <Area type="monotone" dataKey="count" stroke="#FA5252" strokeWidth={2} fillOpacity={1} fill="url(#peachGrad)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* 2. Top 10 Areas Bar Chart */}
                <div className="bg-surface p-6 rounded-xl border border-border elevation-1">
                  <h3 className="text-base font-bold mb-4 text-text-primary uppercase tracking-wider">Top 10 Crime Areas</h3>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={analytics.top_areas} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E9ECEF" />
                        <XAxis type="number" style={{ fontSize: 10, fill: "#868E96" }} tickLine={false} />
                        <YAxis dataKey="name" type="category" width={100} style={{ fontSize: 10, fill: "#495057", fontWeight: "bold" }} tickLine={false} />
                        <Tooltip />
                        <Bar dataKey="count" fill="#FA5252" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* 3. Top 10 Weapons Bar Chart */}
                <div className="bg-surface p-6 rounded-xl border border-border elevation-1">
                  <h3 className="text-base font-bold mb-4 text-text-primary uppercase tracking-wider">Top 10 Weapons/MODUS Types</h3>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={analytics.top_weapons} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E9ECEF" />
                        <XAxis type="number" style={{ fontSize: 10, fill: "#868E96" }} tickLine={false} />
                        <YAxis dataKey="name" type="category" width={100} style={{ fontSize: 10, fill: "#495057", fontWeight: "bold" }} tickLine={false} />
                        <Tooltip />
                        <Bar dataKey="count" fill="#FF8787" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* 4. Age groups distribution */}
                <div className="bg-surface p-6 rounded-xl border border-border elevation-1">
                  <h3 className="text-base font-bold mb-4 text-text-primary uppercase tracking-wider">Victim Age Groups Distribution</h3>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={analytics.age_groups}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E9ECEF" />
                        <XAxis dataKey="name" style={{ fontSize: 11, fill: "#868E96" }} tickLine={false} />
                        <YAxis style={{ fontSize: 11, fill: "#868E96" }} tickLine={false} axisLine={false} />
                        <Tooltip />
                        <Bar dataKey="count" fill="#FA5252" radius={[4, 4, 0, 0]}>
                          {analytics.age_groups.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* 5. Sex breakdown Pie Chart */}
                <div className="bg-surface p-6 rounded-xl border border-border elevation-1">
                  <h3 className="text-base font-bold mb-4 text-text-primary uppercase tracking-wider">Victim Sex Breakdown</h3>
                  <div className="h-72 flex justify-center items-center">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={analytics.sex_breakdown}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={85}
                          paddingAngle={3}
                          dataKey="count"
                          label={(entry) => `${entry.name}: ${(entry.percent * 100).toFixed(1)}%`}
                          style={{ fontSize: 11, fontWeight: "bold" }}
                        >
                          {analytics.sex_breakdown.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                        <Legend verticalAlign="bottom" height={36} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* 6. Day of Week counts */}
                <div className="bg-surface p-6 rounded-xl border border-border elevation-1">
                  <h3 className="text-base font-bold mb-4 text-text-primary uppercase tracking-wider">Crime Count by Day of Week</h3>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={analytics.day_of_week}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E9ECEF" />
                        <XAxis dataKey="name" style={{ fontSize: 11, fill: "#868E96" }} tickLine={false} />
                        <YAxis style={{ fontSize: 11, fill: "#868E96" }} tickLine={false} axisLine={false} />
                        <Tooltip />
                        <Bar dataKey="count" fill="#FA5252" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* 7. Hourly counts */}
                <div className="bg-surface p-6 rounded-xl border border-border elevation-1">
                  <h3 className="text-base font-bold mb-4 text-text-primary uppercase tracking-wider">Crimes Occurrence by Hour (0-23)</h3>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={analytics.hour_of_day}>
                        <defs>
                          <linearGradient id="hourGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#FA5252" stopOpacity={0.4} />
                            <stop offset="95%" stopColor="#FA5252" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E9ECEF" />
                        <XAxis dataKey="name" style={{ fontSize: 11, fill: "#868E96" }} tickLine={false} />
                        <YAxis style={{ fontSize: 11, fill: "#868E96" }} tickLine={false} axisLine={false} />
                        <Tooltip />
                        <Area type="monotone" dataKey="count" stroke="#FA5252" strokeWidth={2} fillOpacity={1} fill="url(#hourGrad)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* 8. YoY line trends */}
                <div className="bg-surface p-6 rounded-xl border border-border elevation-1 md:col-span-2">
                  <h3 className="text-base font-bold mb-4 text-text-primary uppercase tracking-wider">Year-Over-Year Monthly Crime Trend</h3>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={analytics.yoy_trend}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E9ECEF" />
                        <XAxis dataKey="month" style={{ fontSize: 11, fill: "#868E96" }} tickLine={false} />
                        <YAxis style={{ fontSize: 11, fill: "#868E96" }} tickLine={false} axisLine={false} />
                        <Tooltip />
                        <Legend />
                        {/* Dynamically draw lines for years present in data */}
                        {analytics.yoy_trend.length > 0 &&
                          Object.keys(analytics.yoy_trend[0])
                            .filter(key => key !== "month")
                            .map((year, idx) => (
                              <Line
                                key={year}
                                type="monotone"
                                dataKey={year}
                                stroke={COLORS[idx % COLORS.length]}
                                strokeWidth={2}
                                dot={{ r: 4 }}
                                activeDot={{ r: 6 }}
                              />
                            ))
                        }
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

              </div>
            ) : (
              <div className="text-center py-12 text-text-muted">Loading analytics data...</div>
            )}
          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="bg-surface border-t border-border mt-12 py-6 text-center text-xs text-text-muted transition-material">
        <div className="max-w-7xl mx-auto px-6">
          <p>© 2026 Crime Analytics & Suspect Ranking System. Built with FastAPI and React. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
