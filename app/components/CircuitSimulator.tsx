"use client";

import React, { useState, useEffect } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import {
  Panel,
  PanelGroup,
  PanelResizeHandle
} from 'react-resizable-panels';
import styles from './CircuitSimulator.module.css';

// Add visualization type enum
type VisualizationType = 'nyquist' | 'bode' | 'magnitude' | 'phase';

// Add available colors for states
const availableColors = [
  '#10B981', // Green
  '#F59E0B', // Yellow
  '#EF4444', // Red
  '#8B5CF6', // Purple
  '#EC4899', // Pink
  '#3B82F6', // Blue
  '#14B8A6', // Teal
  '#F97316', // Orange
  '#6366F1', // Indigo
  '#A855F7'  // Violet
];

interface ImpedancePoint {
  real: number;
  imaginary: number;
  frequency: number;
  magnitude: number;
  phase: number;
}

interface ControlProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (value: number) => void;
  multiplier?: number;
}

interface ModelSnapshot {
  id: string;
  name: string;
  timestamp: number;
  parameters: {
    rBlank: number;
    rs: number;
    ra: number;
    ca: number;
    rb: number;
    cb: number;
  };
  ter: number;
  data: ImpedancePoint[];
  color: string;
  isVisible: boolean;
}

interface ParameterDiff {
  param: string | React.ReactNode;
  value: string;
  initialValue: string;
  percentChange: number;
}

// Update the formatter function to use superscripts
const formatValue = (value: number, unit: string): string => {
  if (Math.abs(value) >= 1000) {
    const exponent = Math.floor(Math.log10(value));
    const base = value / Math.pow(10, exponent);
    return `${base.toFixed(1)}×10${superscript(exponent)} ${unit}`;
  }
  return `${value.toFixed(1)} ${unit}`;
};

// Helper function to convert numbers to superscript
const superscript = (num: number): string => {
  const digits = num.toString().split('');
  const superscriptDigits = digits.map(d => {
    const superscripts: { [key: string]: string } = {
      '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
      '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
      '-': '⁻'
    };
    return superscripts[d] || d;
  });
  return superscriptDigits.join('');
};

const ControlSlider = ({ label, value, min, max, step, unit, onChange, multiplier = 1 }: ControlProps) => {
  const displayValue = value * multiplier;
  const displayMin = min * multiplier;
  const displayMax = max * multiplier;
  const displayStep = step * multiplier;
  const [error, setError] = useState<boolean>(false);
  const [inputValue, setInputValue] = useState<string>(displayValue.toString());

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setInputValue(val);
    
    if (val === '') {
      setError(true);
      return;
    }

    const newValue = Number(val) / multiplier;
    if (!isNaN(newValue) && newValue >= min && newValue <= max) {
      setError(false);
      onChange(newValue);
    } else {
      setError(true);
    }
  };

  const getFormattedLabel = (label: string) => {
    if (label === "Rblank") return <InlineMath>{`R_{blank}`}</InlineMath>;
    if (label === "Rs") return <InlineMath>{`R_s`}</InlineMath>;
    if (label === "Ra") return <InlineMath>{`R_a`}</InlineMath>;
    if (label === "Rb") return <InlineMath>{`R_b`}</InlineMath>;
    if (label === "Ca") return <InlineMath>{`C_a`}</InlineMath>;
    if (label === "Cb") return <InlineMath>{`C_b`}</InlineMath>;
    return label;
  };

  // Update input value when the prop changes
  useEffect(() => {
    setInputValue(displayValue.toFixed(1));
    setError(false);
  }, [displayValue]);

  return (
    <div className="bg-[#FAFAFA] p-3 rounded border border-[#E0E0E0]">
      <div className="flex flex-col space-y-2">
        <div className="flex justify-between items-center">
          <label className="text-sm font-medium text-[#112E51]">
            {getFormattedLabel(label)}
          </label>
          <div className="flex items-center gap-2">
            <div className="relative">
              <input
                type="text"
                value={inputValue}
                onChange={handleInputChange}
                className={`w-24 px-2 py-1 text-right text-sm border rounded focus:outline-none focus:ring-1 
                  ${error 
                    ? 'border-[#981B1E] focus:ring-[#981B1E] focus:border-[#981B1E] text-[#981B1E]' 
                    : 'border-[#E0E0E0] focus:ring-[#205493] focus:border-[#205493] text-[#112E51]'
                  }`}
              />
              {error && (
                <div className="absolute left-0 top-full mt-1 text-xs text-[#981B1E]">
                  Invalid value
                </div>
              )}
            </div>
            <span className="text-sm text-[#5B616B] min-w-[20px]">{unit}</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-[#5B616B]">{formatValue(displayMin, unit)}</span>
          <input
            type="range"
            min={displayMin}
            max={displayMax}
            step={displayStep}
            value={displayValue}
            onChange={(e) => {
              const newValue = Number(e.target.value) / multiplier;
              setError(false);
              setInputValue((newValue * multiplier).toFixed(1));
              onChange(newValue);
            }}
            className="flex-1 h-2 bg-[#E0E0E0] rounded-lg appearance-none cursor-pointer accent-[#205493]"
          />
          <span className="text-xs text-[#5B616B]">{formatValue(displayMax, unit)}</span>
        </div>
      </div>
    </div>
  );
};

// Update CompactControl component
const CompactControl = ({ label, value, unit, onChange, multiplier = 1 }: Omit<ControlProps, 'min' | 'max' | 'step'>) => {
  const [inputValue, setInputValue] = useState<string>((value * multiplier).toFixed(1));
  const [error, setError] = useState<boolean>(false);

  useEffect(() => {
    setInputValue((value * multiplier).toFixed(1));
    setError(false);
  }, [value, multiplier]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setInputValue(val);
    
    const newValue = Number(val) / multiplier;
    if (!isNaN(newValue)) {
      setError(false);
      onChange(newValue);
    } else {
      setError(true);
    }
  };

  const getFormattedLabel = (label: string) => {
    if (label === "Rblank") return <InlineMath>{`R_{blank}`}</InlineMath>;
    if (label === "Rs") return <InlineMath>{`R_s`}</InlineMath>;
    if (label === "Ra") return <InlineMath>{`R_a`}</InlineMath>;
    if (label === "Rb") return <InlineMath>{`R_b`}</InlineMath>;
    if (label === "Ca") return <InlineMath>{`C_a`}</InlineMath>;
    if (label === "Cb") return <InlineMath>{`C_b`}</InlineMath>;
    return label;
  };

  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-xs text-[#5B616B]">{getFormattedLabel(label)}</span>
      <div className="flex items-center gap-1">
        <input
          type="text"
          value={inputValue}
          onChange={handleChange}
          className={`w-20 px-2 py-0.5 text-right text-xs border rounded focus:outline-none focus:ring-1 ${
            error 
              ? 'border-[#981B1E] focus:ring-[#981B1E] text-[#981B1E]' 
              : 'border-[#E0E0E0] focus:ring-[#205493] text-[#112E51]'
          }`}
        />
        <span className="text-xs text-[#5B616B] min-w-[20px]">{unit}</span>
      </div>
    </div>
  );
};

export default function CircuitSimulator() {
  // Default initial values
  const defaultValues = {
    rBlank: 30,
    rs: 1000,
    ra: 1000,
    ca: 1e-6,
    rb: 1000,
    cb: 1e-6
  };

  // Add visualization type state
  const [visualizationType, setVisualizationType] = useState<VisualizationType>('nyquist');
  
  const [rBlank, setRBlank] = useState<number>(defaultValues.rBlank);
  const [rs, setRs] = useState<number>(defaultValues.rs);
  const [ra, setRa] = useState<number>(defaultValues.ra);
  const [ca, setCa] = useState<number>(defaultValues.ca);
  const [rb, setRb] = useState<number>(defaultValues.rb);
  const [cb, setCb] = useState<number>(defaultValues.cb);
  const [frequencyRange] = useState<[number, number]>([1, 1e4]);
  const [snapshots, setSnapshots] = useState<ModelSnapshot[]>([]);
  const [snapshotName, setSnapshotName] = useState<string>("");
  const [activeSnapshot, setActiveSnapshot] = useState<string | null>(null);
  const colors = ['#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'];
  const [editingName, setEditingName] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'circuit' | 'insights'>('circuit');
  const [isStatesExpanded, setIsStatesExpanded] = useState(true);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [activeColorPicker, setActiveColorPicker] = useState<string | null>(null);

  // Load data from localStorage on initial render
  useEffect(() => {
    const savedSnapshots = localStorage.getItem('rpe-snapshots');
    if (savedSnapshots) {
      const parsedSnapshots = JSON.parse(savedSnapshots);
      setSnapshots(parsedSnapshots);
      
      // If there was an active snapshot, restore its values
      const activeSnap = parsedSnapshots.find((s: ModelSnapshot) => s.id === localStorage.getItem('rpe-active-snapshot'));
      if (activeSnap) {
        setActiveSnapshot(activeSnap.id);
        setRBlank(activeSnap.parameters.rBlank);
        setRs(activeSnap.parameters.rs);
        setRa(activeSnap.parameters.ra);
        setCa(activeSnap.parameters.ca);
        setRb(activeSnap.parameters.rb);
        setCb(activeSnap.parameters.cb);
      }
    } else if (snapshots.length === 0) {
      // Create initial state if no saved data
      const initialState: ModelSnapshot = {
        id: 'initial',
        name: 'Initial State',
        timestamp: Date.now(),
        parameters: defaultValues,
        ter: defaultValues.ra + defaultValues.rb + defaultValues.rs,
        data: generateNyquistData(),
        color: colors[0],
        isVisible: true
      };
      setSnapshots([initialState]);
      setActiveSnapshot('initial');
    }
  }, []);

  // Save to localStorage whenever snapshots change
  useEffect(() => {
    if (snapshots.length > 0) {
      localStorage.setItem('rpe-snapshots', JSON.stringify(snapshots));
      localStorage.setItem('rpe-active-snapshot', activeSnapshot || '');
    }
  }, [snapshots, activeSnapshot]);

  // Update snapshot when parameters change
  const updateActiveSnapshot = () => {
    if (activeSnapshot) {
      setSnapshots(prev => prev.map(s => {
        if (s.id === activeSnapshot) {
          return {
            ...s,
            parameters: {
              rBlank,
              rs,
              ra,
              ca,
              rb,
              cb
            },
            ter: calculateTER(),
            data: generateNyquistData()
          };
        }
        return s;
      }));
    }
  };

  // Create new state with default values
  const createNewState = () => {
    // Reset to default values
    setRBlank(defaultValues.rBlank);
    setRs(defaultValues.rs);
    setRa(defaultValues.ra);
    setCa(defaultValues.ca);
    setRb(defaultValues.rb);
    setCb(defaultValues.cb);

    const newSnapshot: ModelSnapshot = {
      id: Date.now().toString(),
      name: snapshotName || `State ${snapshots.length}`,
      timestamp: Date.now(),
      parameters: defaultValues,
      ter: defaultValues.ra + defaultValues.rb + defaultValues.rs,
      data: generateNyquistDataWithParams(defaultValues),
      color: colors[snapshots.length % colors.length],
      isVisible: true
    };
    setSnapshots(prev => [...prev, newSnapshot]);
    setActiveSnapshot(newSnapshot.id);
    setSnapshotName("");
  };

  // Helper function to generate Nyquist data with specific parameters
  const generateNyquistDataWithParams = (params: ModelSnapshot['parameters']): ImpedancePoint[] => {
    const frequencies = generateFrequencies(frequencyRange[0], frequencyRange[1], 200);
    return frequencies.map(f => {
      const za = calculateZaWithParams(f, params);
      const zb = calculateZbWithParams(f, params);
      
      const sumReal = za.real + zb.real;
      const sumImaginary = za.imaginary + zb.imaginary;
      
      const numeratorReal = params.rs * sumReal;
      const numeratorImaginary = params.rs * sumImaginary;
      
      const denominatorReal = params.rs + sumReal;
      const denominatorImaginary = sumImaginary;
      
      const denomMagnitudeSq = denominatorReal ** 2 + denominatorImaginary ** 2;
      
      const real = params.rBlank + (numeratorReal * denominatorReal + numeratorImaginary * denominatorImaginary) / denomMagnitudeSq;
      const imaginary = (numeratorImaginary * denominatorReal - numeratorReal * denominatorImaginary) / denomMagnitudeSq;
      
      return {
        real,
        imaginary,
        frequency: f,
        magnitude: Math.sqrt(real * real + imaginary * imaginary),
        phase: Math.atan2(imaginary, real) * (180 / Math.PI)
      };
    });
  };

  const calculateZaWithParams = (f: number, params: ModelSnapshot['parameters']): { real: number; imaginary: number } => {
    const omega = 2 * Math.PI * f;
    const real = params.ra;
    const imaginary = -1 / (omega * params.ca);
    return { real, imaginary };
  };

  const calculateZbWithParams = (f: number, params: ModelSnapshot['parameters']): { real: number; imaginary: number } => {
    const omega = 2 * Math.PI * f;
    const real = params.rb;
    const imaginary = -1 / (omega * params.cb);
    return { real, imaginary };
  };

  // Wrap parameter setters to update active snapshot
  const updateParameter = (setter: (value: number) => void, value: number) => {
    setter(value);
    // Use setTimeout to ensure state is updated before calculating new values
    setTimeout(updateActiveSnapshot, 0);
  };

  // Edit a snapshot
  const editSnapshot = (snapshot: ModelSnapshot) => {
    setActiveSnapshot(snapshot.id);
    setRBlank(snapshot.parameters.rBlank);
    setRs(snapshot.parameters.rs);
    setRa(snapshot.parameters.ra);
    setCa(snapshot.parameters.ca);
    setRb(snapshot.parameters.rb);
    setCb(snapshot.parameters.cb);

    if (!snapshot.isVisible) {
      toggleSnapshotVisibility(snapshot.id);
    }
  };

  // Save changes to current state or create new state
  const saveChanges = (snapshot: ModelSnapshot) => {
    setSnapshots(prev => prev.map(s => {
      if (s.id === snapshot.id) {
        return {
          ...s,
          parameters: {
            rBlank,
            rs,
            ra,
            ca,
            rb,
            cb
          },
          ter: calculateTER(),
          data: generateNyquistData()
        };
      }
      return s;
    }));
    // Deselect the state after saving
    setActiveSnapshot(null);
  };

  // Toggle snapshot visibility
  const toggleSnapshotVisibility = (id: string) => {
    setSnapshots(prev => prev.map(s => 
      s.id === id ? { ...s, isVisible: !s.isVisible } : s
    ));
  };

  // Get parameter values based on active state or current values
  const getParameterValues = (stateId: string) => {
    if (activeSnapshot === stateId) {
      return {
        rBlank,
        rs,
        ra,
        ca,
        rb,
        cb
      };
    }
    const state = snapshots.find(s => s.id === stateId);
    return state ? state.parameters : {
      rBlank,
      rs,
      ra,
      ca,
      rb,
      cb
    };
  };

  // Get parameter difference for display
  const getParameterDiff = (snapshot: ModelSnapshot) => {
    const current = getParameterValues(snapshot.id);
    
    const changes: ParameterDiff[] = [];
    Object.entries(snapshot.parameters).forEach(([key, value]) => {
      const currentValue = current[key as keyof typeof current];
      
      // Format the parameter name with subscripts using KaTeX
      const getFormattedParamName = (name: string) => {
        if (name === "rBlank") return <InlineMath>{`R_{blank}`}</InlineMath>;
        if (name === "rs") return <InlineMath>{`R_s`}</InlineMath>;
        if (name === "ra") return <InlineMath>{`R_a`}</InlineMath>;
        if (name === "rb") return <InlineMath>{`R_b`}</InlineMath>;
        if (name === "ca") return <InlineMath>{`C_a`}</InlineMath>;
        if (name === "cb") return <InlineMath>{`C_b`}</InlineMath>;
        return name;
      };
      
      if (key.startsWith('c')) {
        // Format capacitance in µF
        const valueInMicroF = value * 1e6;
        const currentInMicroF = currentValue * 1e6;
        const percentChange = ((currentInMicroF - valueInMicroF) / valueInMicroF) * 100;
        
        changes.push({
          param: getFormattedParamName(key),
          value: `${currentInMicroF.toFixed(1)} µF`,
          initialValue: `${valueInMicroF.toFixed(1)} µF`,
          percentChange: percentChange
        });
      } else {
        // Format resistance in Ω
        const percentChange = ((currentValue - value) / value) * 100;
        changes.push({
          param: getFormattedParamName(key),
          value: `${currentValue.toFixed(0)} Ω`,
          initialValue: `${value.toFixed(0)} Ω`,
          percentChange: percentChange
        });
      }
    });
    return changes;
  };

  // Delete a snapshot
  const deleteSnapshot = (id: string) => {
    setSnapshots(prev => prev.filter(s => s.id !== id));
  };

  // Calculate Za(ω) with proper complex arithmetic
  const calculateZa = (f: number): { real: number; imaginary: number } => {
    const omega = 2 * Math.PI * f;
    const real = ra;
    const imaginary = -1 / (omega * ca);
    return { real, imaginary };
  };

  // Calculate Zb(ω) with proper complex arithmetic
  const calculateZb = (f: number): { real: number; imaginary: number } => {
    const omega = 2 * Math.PI * f;
    const real = rb;
    const imaginary = -1 / (omega * cb);
    return { real, imaginary };
  };

  // Calculate complex impedance with proper arithmetic
  const calculateImpedance = (f: number): ImpedancePoint => {
    const za = calculateZa(f);
    const zb = calculateZb(f);
    
    // Za + Zb
    const sumReal = za.real + zb.real;
    const sumImaginary = za.imaginary + zb.imaginary;
    
    // Rs * (Za + Zb)
    const numeratorReal = rs * sumReal;
    const numeratorImaginary = rs * sumImaginary;
    
    // Rs + Za + Zb
    const denominatorReal = rs + sumReal;
    const denominatorImaginary = sumImaginary;
    
    // Complex division: (Rs * (Za + Zb)) / (Rs + Za + Zb)
    const denomMagnitudeSq = denominatorReal ** 2 + denominatorImaginary ** 2;
    
    // Final impedance calculation including Rblank
    const real = rBlank + (numeratorReal * denominatorReal + numeratorImaginary * denominatorImaginary) / denomMagnitudeSq;
    const imaginary = (numeratorImaginary * denominatorReal - numeratorReal * denominatorImaginary) / denomMagnitudeSq;
    
    return {
      real,
      imaginary,
      frequency: f,
      magnitude: Math.sqrt(real * real + imaginary * imaginary),
      phase: Math.atan2(imaginary, real) * (180 / Math.PI)
    };
  };

  // Generate logarithmically spaced frequencies
  const generateFrequencies = (start: number, end: number, points: number): number[] => {
    const frequencies: number[] = [];
    const logStart = Math.log10(start);
    const logEnd = Math.log10(end);
    const step = (logEnd - logStart) / (points - 1);
    
    for (let i = 0; i < points; i++) {
      frequencies.push(Math.pow(10, logStart + i * step));
    }
    return frequencies;
  };

  // Generate Nyquist plot data points
  const generateNyquistData = (): ImpedancePoint[] => {
    const frequencies = generateFrequencies(frequencyRange[0], frequencyRange[1], 200);
    return frequencies.map(f => calculateImpedance(f));
  };

  // Calculate proper axis ranges
  const calculateDomains = (data: ImpedancePoint[]) => {
    const padding = 0.1; // 10% padding
    
    const realValues = data.map(d => d.real);
    const imagValues = data.map(d => Math.abs(d.imaginary)); // Use absolute values for scaling
    
    const realMin = Math.min(...realValues);
    const realMax = Math.max(...realValues);
    const realRange = realMax - realMin;
    
    const imagMax = Math.max(...imagValues);
    
    // Make plot more square-like for better visualization
    const maxRange = Math.max(realRange, imagMax * 2);
    
    return {
      realDomain: [
        Math.max(0, realMin - maxRange * padding),
        realMax + maxRange * padding
      ],
      imagDomain: [
        -imagMax * (1 + padding),
        imagMax * padding
      ]
    };
  };

  const data = generateNyquistData();
  const { realDomain, imagDomain } = calculateDomains(data);

  // Calculate TER for display
  const calculateTER = () => {
    return ra + rb + rs;
  };

  // Generate Nyquist data for a specific state
  const generateStateData = (stateId: string) => {
    const parameters = getParameterValues(stateId);
    const frequencies = generateFrequencies(frequencyRange[0], frequencyRange[1], 200);
    return frequencies.map(f => {
      const za = calculateZaForState(f, parameters);
      const zb = calculateZbForState(f, parameters);
      
      const sumReal = za.real + zb.real;
      const sumImaginary = za.imaginary + zb.imaginary;
      
      const numeratorReal = parameters.rs * sumReal;
      const numeratorImaginary = parameters.rs * sumImaginary;
      
      const denominatorReal = parameters.rs + sumReal;
      const denominatorImaginary = sumImaginary;
      
      const denomMagnitudeSq = denominatorReal ** 2 + denominatorImaginary ** 2;
      
      const real = parameters.rBlank + (numeratorReal * denominatorReal + numeratorImaginary * denominatorImaginary) / denomMagnitudeSq;
      const imaginary = (numeratorImaginary * denominatorReal - numeratorReal * denominatorImaginary) / denomMagnitudeSq;
      
      return {
        real,
        imaginary,
        frequency: f,
        magnitude: Math.sqrt(real * real + imaginary * imaginary),
        phase: Math.atan2(imaginary, real) * (180 / Math.PI)
      };
    });
  };

  // Calculate Za(ω) for a specific state
  const calculateZaForState = (f: number, parameters: ModelSnapshot['parameters']): { real: number; imaginary: number } => {
    const omega = 2 * Math.PI * f;
    const real = parameters.ra;
    const imaginary = -1 / (omega * parameters.ca);
    return { real, imaginary };
  };

  // Calculate Zb(ω) for a specific state
  const calculateZbForState = (f: number, parameters: ModelSnapshot['parameters']): { real: number; imaginary: number } => {
    const omega = 2 * Math.PI * f;
    const real = parameters.rb;
    const imaginary = -1 / (omega * parameters.cb);
    return { real, imaginary };
  };

  // Add state name editing
  const updateStateName = (id: string, newName: string) => {
    setSnapshots(prev => prev.map(s => 
      s.id === id ? { ...s, name: newName } : s
    ));
    setEditingName(null);
  };

  // Update tab toggle function
  const toggleTab = (tab: 'circuit' | 'insights') => {
    setActiveTab(tab);
  };

  // Add visualization options
  const visualizationOptions: { value: VisualizationType; label: React.ReactNode; description: string }[] = [
    {
      value: 'nyquist',
      label: <InlineMath>{`Z_{eq}(\\omega)`}</InlineMath>,
      description: 'Nyquist Plot'
    },
    {
      value: 'bode',
      label: <InlineMath>{`|Z_{eq}(\\omega)|, \\phi(\\omega)`}</InlineMath>,
      description: 'Bode Plot'
    },
    {
      value: 'magnitude',
      label: <InlineMath>{`|Z_{eq}(\\omega)|`}</InlineMath>,
      description: 'Magnitude Plot'
    },
    {
      value: 'phase',
      label: <InlineMath>{`\\phi(\\omega)`}</InlineMath>,
      description: 'Phase Plot'
    }
  ];

  // Add click outside handler
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const dropdown = document.getElementById('visualization-dropdown');
      const button = document.getElementById('visualization-button');
      if (dropdown && button && !dropdown.contains(event.target as Node) && !button.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Add click outside handler for color picker
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const colorPicker = document.getElementById('color-picker');
      if (colorPicker && !colorPicker.contains(event.target as Node)) {
        setActiveColorPicker(null);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Update ResizeHandle component
  const ResizeHandle = ({ className = '', orientation = 'vertical' }: { className?: string; orientation?: 'vertical' | 'horizontal' }) => (
    <PanelResizeHandle 
      className={`${orientation === 'vertical' ? styles.resizeHandleVertical : styles.resizeHandleHorizontal} ${className}`} 
    />
  );

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <header className="flex-none border-b border-[#E0E0E0] py-3 px-4 bg-white">
        <h1 className="text-2xl font-medium text-[#205493]">
          RPE Impedance Playground
        </h1>
        <p className="text-[#5B616B] text-sm mt-0.5">
          Interactive graphical visualization of RPE impedance characteristics using a circuit model
        </p>
      </header>

      <div className="flex-1 min-h-0">
        <PanelGroup direction="horizontal" className="h-full">
          <Panel defaultSize={20} minSize={15} maxSize={40} className={styles.panel}>
            <div className={`${styles.panelContent} border-r border-[#E0E0E0] bg-white flex flex-col`}>
              {/* States Panel */}
              <div className="flex-1 flex flex-col min-h-0">
                <button
                  onClick={() => setIsStatesExpanded(!isStatesExpanded)}
                  className="flex-none w-full flex items-center justify-between p-4 hover:bg-[#F8F9FA] transition-colors border-b border-[#E0E0E0] bg-white z-10"
                >
                  <span className="text-sm font-medium text-[#112E51]">States</span>
                  <svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    className={`w-4 h-4 text-[#5B616B] transition-transform ${isStatesExpanded ? 'transform rotate-180' : ''}`}
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="2" 
                    strokeLinecap="round" 
                    strokeLinejoin="round"
                  >
                    <polyline points="6 9 12 15 18 9"></polyline>
                  </svg>
                </button>

                {isStatesExpanded && (
                  <div className="flex-1 overflow-y-auto p-4 space-y-2 bg-[#FAFAFA] min-h-0">
                    {snapshots.map(snapshot => (
                      <div 
                        key={snapshot.id} 
                        className={`border rounded-lg transition-all shadow-sm ${
                          activeSnapshot === snapshot.id ? 'border-[#205493] bg-[#F1F9FF] ring-1 ring-[#205493]' : 'border-[#E0E0E0] bg-white hover:border-[#AEB0B5]'
                        }`}
                      >
                        {/* State Header */}
                        <div className="flex items-center gap-2 p-3">
                          <div className="relative">
                            {activeSnapshot === snapshot.id ? (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setActiveColorPicker(activeColorPicker === snapshot.id ? null : snapshot.id);
                                }}
                                className={`w-6 h-6 rounded-full transition-all flex items-center justify-center group ${
                                  snapshot.isVisible ? 'bg-white' : 'bg-[#F5F5F5]'
                                } hover:bg-[#F1F9FF] border border-[#E0E0E0]`}
                                title="Change visibility and color"
                              >
                                <div
                                  className={`w-3 h-3 rounded-full transition-colors ring-1 ring-inset ${
                                    snapshot.isVisible ? `ring-${snapshot.color}` : 'ring-[#E0E0E0]'
                                  } group-hover:scale-110`}
                                  style={{ backgroundColor: snapshot.isVisible ? snapshot.color : '#FFF' }}
                                />
                              </button>
                            ) : (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  toggleSnapshotVisibility(snapshot.id);
                                }}
                                className="w-6 h-6 rounded-full transition-all flex items-center justify-center hover:bg-[#F1F9FF] border border-[#E0E0E0]"
                                title={snapshot.isVisible ? "Hide from plot" : "Show in plot"}
                              >
                                <div
                                  className="w-3 h-3 rounded-full transition-transform hover:scale-110"
                                  style={{ backgroundColor: snapshot.color, opacity: snapshot.isVisible ? 1 : 0.3 }}
                                />
                              </button>
                            )}
                            {activeColorPicker === snapshot.id && (
                              <div 
                                id="color-picker"
                                className="absolute left-0 top-full mt-1 z-20"
                              >
                                <div className="bg-white border border-[#E0E0E0] rounded-lg shadow-lg p-2">
                                  <div className="mb-2 pb-2 border-b border-[#E0E0E0]">
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        toggleSnapshotVisibility(snapshot.id);
                                      }}
                                      className={`w-full px-2 py-1 text-xs rounded flex items-center gap-2 ${
                                        snapshot.isVisible 
                                          ? 'text-[#981B1E] hover:bg-[#FDE4E4]' 
                                          : 'text-[#205493] hover:bg-[#F1F9FF]'
                                      }`}
                                    >
                                      <svg 
                                        xmlns="http://www.w3.org/2000/svg" 
                                        className="w-3 h-3" 
                                        viewBox="0 0 24 24" 
                                        fill="none" 
                                        stroke="currentColor" 
                                        strokeWidth="2" 
                                        strokeLinecap="round" 
                                        strokeLinejoin="round"
                                      >
                                        {snapshot.isVisible ? (
                                          <>
                                            <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
                                            <line x1="1" y1="1" x2="23" y2="23" />
                                          </>
                                        ) : (
                                          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                                        )}
                                      </svg>
                                      {snapshot.isVisible ? 'Hide from plot' : 'Show in plot'}
                                    </button>
                                  </div>
                                  <div className="grid grid-cols-5 gap-1 w-[120px]">
                                    {availableColors.map((color) => (
                                      <button
                                        key={color}
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          setSnapshots(prev => prev.map(s => 
                                            s.id === snapshot.id ? { ...s, color } : s
                                          ));
                                        }}
                                        className={`w-4 h-4 rounded-full transition-transform hover:scale-110 ${
                                          snapshot.color === color ? 'ring-2 ring-[#205493] ring-offset-1' : ''
                                        }`}
                                        style={{ backgroundColor: color }}
                                        title="Change color"
                                      />
                                    ))}
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                          {editingName === snapshot.id ? (
                            <input
                              type="text"
                              value={snapshot.name}
                              onChange={(e) => updateStateName(snapshot.id, e.target.value)}
                              onBlur={() => setEditingName(null)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                  updateStateName(snapshot.id, e.currentTarget.value);
                                }
                              }}
                              className="flex-1 px-2 py-1 text-sm border border-[#E0E0E0] rounded-md focus:outline-none focus:ring-1 focus:ring-[#205493]"
                              autoFocus
                            />
                          ) : (
                            <span 
                              className="flex-1 text-sm text-[#112E51] cursor-pointer hover:text-[#205493]"
                              onClick={() => setEditingName(snapshot.id)}
                            >
                              {snapshot.name}
                            </span>
                          )}
                          
                          {/* Action Buttons */}
                          <div className="flex items-center gap-1">
                            {activeSnapshot === snapshot.id ? (
                              <button
                                onClick={() => saveChanges(snapshot)}
                                className="p-1.5 text-xs bg-[#E1F3F8] text-[#205493] rounded-md hover:bg-[#D6E8F7]"
                                title="Save changes"
                              >
                                <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
                                  <polyline points="17 21 17 13 7 13 7 21"/>
                                  <polyline points="7 3 7 8 15 8"/>
                                </svg>
                              </button>
                            ) : (
                              <button
                                onClick={() => editSnapshot(snapshot)}
                                className="p-1.5 text-xs bg-[#F1F1F1] text-[#5B616B] rounded-md hover:bg-[#E6E6E6]"
                                title="Edit this state"
                              >
                                <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                                </svg>
                              </button>
                            )}
                            <button
                              onClick={() => deleteSnapshot(snapshot.id)}
                              className="p-1.5 text-xs bg-[#FDE4E4] text-[#981B1E] rounded-md hover:bg-[#FBD1D1]"
                              title="Delete state"
                            >
                              <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <polyline points="3 6 5 6 21 6"/>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                                <line x1="10" y1="11" x2="10" y2="17"/>
                                <line x1="14" y1="11" x2="14" y2="17"/>
                              </svg>
                            </button>
                          </div>
                        </div>

                        {/* Expanded State Content */}
                        {activeSnapshot === snapshot.id ? (
                          <div className="p-3 bg-[#FAFAFA] space-y-3 border-t border-[#E0E0E0] rounded-b-lg">
                            {/* Compact Controls */}
                            <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                              <CompactControl
                                label="Rblank"
                                value={rBlank}
                                unit="Ω"
                                onChange={(value) => updateParameter(setRBlank, value)}
                              />
                              <CompactControl
                                label="Rs"
                                value={rs}
                                unit="Ω"
                                onChange={(value) => updateParameter(setRs, value)}
                              />
                              <CompactControl
                                label="Ra"
                                value={ra}
                                unit="Ω"
                                onChange={(value) => updateParameter(setRa, value)}
                              />
                              <CompactControl
                                label="Ca"
                                value={ca}
                                unit="µF"
                                multiplier={1e6}
                                onChange={(value) => updateParameter(setCa, value)}
                              />
                              <CompactControl
                                label="Rb"
                                value={rb}
                                unit="Ω"
                                onChange={(value) => updateParameter(setRb, value)}
                              />
                              <CompactControl
                                label="Cb"
                                value={cb}
                                unit="µF"
                                multiplier={1e6}
                                onChange={(value) => updateParameter(setCb, value)}
                              />
                            </div>

                            {/* Sliders */}
                            <div className="space-y-2">
                              <ControlSlider
                                label="Rblank"
                                value={rBlank}
                                min={10}
                                max={50}
                                step={1}
                                unit="Ω"
                                onChange={(value) => updateParameter(setRBlank, value)}
                              />
                              <ControlSlider
                                label="Rs"
                                value={rs}
                                min={100}
                                max={10000}
                                step={100}
                                unit="Ω"
                                onChange={(value) => updateParameter(setRs, value)}
                              />
                              <ControlSlider
                                label="Ra"
                                value={ra}
                                min={100}
                                max={10000}
                                step={100}
                                unit="Ω"
                                onChange={(value) => updateParameter(setRa, value)}
                              />
                              <ControlSlider
                                label="Ca"
                                value={ca}
                                min={0.1e-6}
                                max={10e-6}
                                step={0.1e-6}
                                unit="µF"
                                multiplier={1e6}
                                onChange={(value) => updateParameter(setCa, value)}
                              />
                              <ControlSlider
                                label="Rb"
                                value={rb}
                                min={100}
                                max={10000}
                                step={100}
                                unit="Ω"
                                onChange={(value) => updateParameter(setRb, value)}
                              />
                              <ControlSlider
                                label="Cb"
                                value={cb}
                                min={0.1e-6}
                                max={10e-6}
                                step={0.1e-6}
                                unit="µF"
                                multiplier={1e6}
                                onChange={(value) => updateParameter(setCb, value)}
                              />
                            </div>
                          </div>
                        ) : (
                          <div className="text-xs border-t border-[#E0E0E0] bg-[#FAFAFA] rounded-b-lg divide-y divide-[#E0E0E0]">
                            {getParameterDiff(snapshot).map((change, i) => (
                              <div 
                                key={i} 
                                className="flex items-center justify-between px-3 py-1.5"
                              >
                                <span className="text-[#5B616B]">{change.param}</span>
                                <span className="text-[#112E51] font-medium">{change.value}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                    
                    {/* Add State Button */}
                    <button
                      onClick={createNewState}
                      className="w-full p-3 flex items-center justify-center gap-2 bg-white border border-dashed border-[#AEB0B5] rounded-lg text-[#5B616B] hover:border-[#205493] hover:text-[#205493] hover:bg-[#F1F9FF] transition-all group"
                    >
                      <svg 
                        xmlns="http://www.w3.org/2000/svg" 
                        className="w-4 h-4" 
                        viewBox="0 0 24 24" 
                        fill="none" 
                        stroke="currentColor" 
                        strokeWidth="2" 
                        strokeLinecap="round" 
                        strokeLinejoin="round"
                      >
                        <line x1="12" y1="5" x2="12" y2="19"/>
                        <line x1="5" y1="12" x2="19" y2="12"/>
                      </svg>
                      <span className="text-sm font-medium">Add New State</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          </Panel>

          <ResizeHandle orientation="horizontal" />

          <Panel defaultSize={80} className={styles.panel}>
            <PanelGroup direction="vertical" className="h-full">
              <Panel defaultSize={60} minSize={30} className={styles.panel}>
                <div className={`${styles.panelContent} p-3`}>
                  <div className="h-full bg-white rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <div className="relative">
                        <button
                          id="visualization-button"
                          onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                          aria-haspopup="true"
                          aria-expanded={isDropdownOpen}
                          aria-controls="visualization-dropdown"
                          className="flex items-center gap-2 text-sm font-medium text-[#112E51] hover:text-[#205493] transition-colors group"
                        >
                          <span className="flex items-center gap-1">
                            {visualizationOptions.find(opt => opt.value === visualizationType)?.label}
                            <span className="mx-1">-</span>
                            <span>{visualizationOptions.find(opt => opt.value === visualizationType)?.description}</span>
                          </span>
                          <svg 
                            xmlns="http://www.w3.org/2000/svg" 
                            className={`w-4 h-4 text-[#5B616B] group-hover:text-[#205493] transition-transform ${isDropdownOpen ? 'transform rotate-180' : ''}`}
                            viewBox="0 0 24 24" 
                            fill="none" 
                            stroke="currentColor" 
                            strokeWidth="2" 
                            strokeLinecap="round" 
                            strokeLinejoin="round"
                            aria-hidden="true"
                          >
                            <polyline points="6 9 12 15 18 9"></polyline>
                          </svg>
                        </button>
                        <div 
                          id="visualization-dropdown"
                          role="listbox"
                          aria-label="Visualization options"
                          className={`absolute top-full left-0 mt-1 bg-white border border-[#E0E0E0] rounded-lg shadow-lg z-10 min-w-[200px] ${isDropdownOpen ? 'block' : 'hidden'}`}
                        >
                          {visualizationOptions.map((option) => (
                            <button
                              key={option.value}
                              role="option"
                              aria-selected={visualizationType === option.value}
                              onClick={() => {
                                setVisualizationType(option.value);
                                setIsDropdownOpen(false);
                              }}
                              className={`w-full px-4 py-2 text-left hover:bg-[#F8F9FA] transition-colors ${
                                visualizationType === option.value 
                                  ? 'bg-[#F1F9FF] text-[#205493]' 
                                  : 'text-[#112E51]'
                              }`}
                            >
                              <div className="flex items-center gap-1">
                                {option.label}
                                <span className="mx-1">-</span>
                                <span className="text-sm">{option.description}</span>
                              </div>
                            </button>
                          ))}
                        </div>
                      </div>
                      <span className="text-xs text-[#5B616B]">Impedance characteristics</span>
                    </div>
                    <div className="h-[calc(100%-24px)]">
                      <ResponsiveContainer width="100%" height="100%">
                        {visualizationType === 'nyquist' ? (
                          <ScatterChart 
                            margin={{ top: 20, right: 30, bottom: 60, left: 60 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="#E0E0E0" />
                            <XAxis
                              dataKey="real"
                              type="number"
                              domain={realDomain}
                              tickFormatter={value => value.toFixed(0)}
                              tick={{ fontSize: 11, fill: '#5B616B' }}
                              tickCount={6}
                              label={{ 
                                value: "Real Impedance (Ω)", 
                                position: 'bottom', 
                                offset: 40,
                                style: { fontSize: 12, fill: '#112E51' }
                              }}
                              stroke="#AEB0B5"
                            />
                            <YAxis
                              dataKey="imaginary"
                              type="number"
                              domain={imagDomain}
                              tickFormatter={value => value.toFixed(0)}
                              tick={{ fontSize: 11, fill: '#5B616B' }}
                              tickCount={6}
                              label={{ 
                                value: "Imaginary Impedance (Ω)", 
                                angle: -90, 
                                position: 'left',
                                offset: 40,
                                style: { fontSize: 12, fill: '#112E51' }
                              }}
                              stroke="#AEB0B5"
                            />
                            <Tooltip
                              content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                  const data = payload[0].payload as ImpedancePoint;
                                  return (
                                    <div className="bg-white p-3 border border-[#E0E0E0] shadow-lg rounded-lg text-xs">
                                      <p className="font-medium mb-1.5 text-[#112E51]">f = {data.frequency.toExponential(1)} Hz</p>
                                      <p className="text-[#5B616B]">
                                        <InlineMath>{`Z'`}</InlineMath> = {formatValue(data.real, 'Ω')}
                                      </p>
                                      <p className="text-[#5B616B]">
                                        <InlineMath>{`-Z''`}</InlineMath> = {formatValue(-data.imaginary, 'Ω')}
                                      </p>
                                      <p className="text-[#5B616B]">
                                        <InlineMath>{`|Z|`}</InlineMath> = {formatValue(data.magnitude, 'Ω')}
                                      </p>
                                      <p className="text-[#5B616B]">
                                        <InlineMath>{`\\phi`}</InlineMath> = {data.phase.toFixed(1)}°
                                      </p>
                                    </div>
                                  );
                                }
                                return null;
                              }}
                            />
                            <ReferenceLine x={0} stroke="#AEB0B5" strokeWidth={1} />
                            <ReferenceLine y={0} stroke="#AEB0B5" strokeWidth={1} />
                            
                            {snapshots
                              .filter(s => s.isVisible)
                              .map((snapshot) => (
                                <Scatter
                                  key={snapshot.id}
                                  name={snapshot.name}
                                  data={generateStateData(snapshot.id)}
                                  fill={snapshot.color}
                                  line={{ 
                                    stroke: snapshot.color,
                                    strokeWidth: 1.5,
                                    strokeDasharray: activeSnapshot === snapshot.id ? undefined : '5 5'
                                  }}
                                  shape={(props: unknown) => {
                                    const { cx, cy, fill } = props as { cx: number; cy: number; fill: string };
                                    return (
                                      <circle
                                        cx={cx}
                                        cy={cy}
                                        r={2}
                                        fill={fill}
                                        stroke="none"
                                      />
                                    );
                                  }}
                                />
                            ))}
                          </ScatterChart>
                        ) : (
                          <div className="h-full w-full flex items-center justify-center">
                            <div className="text-center">
                              <div className="text-[#5B616B] mb-2">
                                {visualizationType === 'bode' && (
                                  <InlineMath>{`|Z_{eq}(\\omega)|, \\phi(\\omega)`}</InlineMath>
                                )}
                                {visualizationType === 'magnitude' && (
                                  <InlineMath>{`|Z_{eq}(\\omega)|`}</InlineMath>
                                )}
                                {visualizationType === 'phase' && (
                                  <InlineMath>{`\\phi(\\omega)`}</InlineMath>
                                )}
                              </div>
                              <p className="text-sm text-[#5B616B]">Plot coming soon</p>
                            </div>
                          </div>
                        )}
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </Panel>

              <ResizeHandle orientation="vertical" />

              <Panel defaultSize={40} className={styles.panel}>
                <div className={`${styles.panelContent} flex flex-col`}>
                  {/* Tabs */}
                  <div className="flex border-b border-[#E0E0E0] bg-white px-4 flex-none">
                    <button
                      onClick={() => toggleTab('circuit')}
                      className={`px-4 py-2 text-sm font-medium transition-colors relative
                        ${activeTab === 'circuit'
                          ? 'text-[#205493]' 
                          : 'text-[#5B616B] hover:text-[#112E51]'}`}
                    >
                      Circuit Model
                      {activeTab === 'circuit' && (
                        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[#205493]" />
                      )}
                    </button>

                    <button
                      onClick={() => toggleTab('insights')}
                      className={`px-4 py-2 text-sm font-medium transition-colors relative
                        ${activeTab === 'insights'
                          ? 'text-[#205493]' 
                          : 'text-[#5B616B] hover:text-[#112E51]'}`}
                    >
                      Insights
                      {activeTab === 'insights' && (
                        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[#205493]" />
                      )}
                    </button>
                  </div>
                  
                  {/* Scrollable Content */}
                  <div className="flex-1 overflow-y-auto bg-[#FAFAFA] p-4">
                    {activeTab === 'circuit' && (
                      <div className="space-y-4">
                        <div className="bg-white rounded-lg border border-[#E0E0E0] p-4">
                          <h3 className="text-sm font-medium text-[#112E51] mb-3">Equivalent Circuit Model</h3>
                          <div className="max-w-full">
                            <div style={{ fontSize: '0.85em', maxWidth: '100%', overflowX: 'hidden' }}>
                              <BlockMath>
                                {`Z_{eq}(\\omega) = R_{blank} + \\frac{R_s(Z_a(\\omega) + Z_b(\\omega))}{R_s + Z_a(\\omega) + Z_b(\\omega)}`}
                              </BlockMath>
                            </div>
                            <div className="mt-4 space-y-3">
                              <div style={{ fontSize: '0.85em' }}>
                                <BlockMath>
                                  {`Z_a(\\omega) = R_a + \\frac{1}{j\\omega C_a}`}
                                </BlockMath>
                              </div>
                              <div style={{ fontSize: '0.85em' }}>
                                <BlockMath>
                                  {`Z_b(\\omega) = R_b + \\frac{1}{j\\omega C_b}`}
                                </BlockMath>
                              </div>
                            </div>
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-white rounded-lg border border-[#E0E0E0] p-4">
                            <h3 className="text-sm font-medium text-[#112E51] mb-3">Resistances</h3>
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <InlineMath>{`R_{blank}`}</InlineMath>
                                  <span className="text-sm text-[#5B616B]">Blank resistance</span>
                                </div>
                                <span className="text-xs text-[#5B616B]">10-50 Ω</span>
                              </div>
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <InlineMath>{`R_s`}</InlineMath>
                                  <span className="text-sm text-[#5B616B]">Shunt resistance</span>
                                </div>
                                <span className="text-xs text-[#5B616B]">10²-10⁴ Ω</span>
                              </div>
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <InlineMath>{`R_a, R_b`}</InlineMath>
                                  <span className="text-sm text-[#5B616B]">Apical & basal</span>
                                </div>
                                <span className="text-xs text-[#5B616B]">10²-10⁴ Ω</span>
                              </div>
                            </div>
                          </div>
                          <div className="bg-white rounded-lg border border-[#E0E0E0] p-4">
                            <h3 className="text-sm font-medium text-[#112E51] mb-3">Capacitances</h3>
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <InlineMath>{`C_a`}</InlineMath>
                                  <span className="text-sm text-[#5B616B]">Apical membrane</span>
                                </div>
                                <span className="text-xs text-[#5B616B]">0.1-10 µF</span>
                              </div>
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <InlineMath>{`C_b`}</InlineMath>
                                  <span className="text-sm text-[#5B616B]">Basal membrane</span>
                                </div>
                                <span className="text-xs text-[#5B616B]">0.1-10 µF</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <InlineMath>{`\\omega`}</InlineMath>
                                <span className="text-sm text-[#5B616B]">Angular frequency</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {activeTab === 'insights' && (
                      <div className="space-y-4">
                        {/* TER Section */}
                        <div className="bg-white p-3 rounded-lg border border-[#E0E0E0]">
                          <div className="flex items-center justify-between">
                            <div>
                              <h4 className="text-sm font-medium text-[#112E51]">
                                Total Epithelial Resistance
                              </h4>
                              <p className="text-xs text-[#5B616B]">Membrane & shunt sum</p>
                            </div>
                            <div className="text-right">
                              <p className="text-base font-medium text-[#205493]">{formatValue(calculateTER(), 'Ω')}</p>
                            </div>
                          </div>
                        </div>

                        {/* Frequency Range Section */}
                        <div className="bg-white p-3 rounded-lg border border-[#E0E0E0]">
                          <div className="flex items-center justify-between">
                            <div>
                              <h4 className="text-sm font-medium text-[#112E51]">Frequency Range</h4>
                              <p className="text-xs text-[#5B616B]">Bandwidth</p>
                            </div>
                            <div className="text-right">
                              <div className="flex items-center gap-1 text-sm text-[#5B616B]">
                                <span>{formatValue(frequencyRange[0], 'Hz')}</span>
                                <span className="text-[#AEB0B5]">→</span>
                                <span>{formatValue(frequencyRange[1], 'Hz')}</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </Panel>
            </PanelGroup>
          </Panel>
        </PanelGroup>
      </div>
    </div>
  );
}