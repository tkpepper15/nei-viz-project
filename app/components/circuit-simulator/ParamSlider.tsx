import React from 'react';

interface ParamSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (value: number) => void;
  onMinChange?: (min: number) => void;
  onMaxChange?: (max: number) => void;
  transformValue?: (value: number) => string;
  ticks?: { level: number; value: number }[];
  log?: boolean; // If true, use logarithmic slider
  tickLabels?: string[]; // Custom tick labels for below the slider
  readOnlyRange?: boolean; // If true, min/max inputs are read-only
}

export const ParamSlider: React.FC<ParamSliderProps> = ({ 
  label, 
  value, 
  min, 
  max, 
  step, 
  unit, 
  onChange,
  onMinChange,
  onMaxChange,
  transformValue,
  ticks,
  log = false,
  tickLabels,
  readOnlyRange = true
}) => {
  // For log sliders, map slider position (0-100) to value and vice versa
  const sliderMin = 0;
  const sliderMax = 100;
  const valueToSlider = (val: number) => {
    if (log) {
      const logMin = Math.log10(min);
      const logMax = Math.log10(max);
      return ((Math.log10(val) - logMin) / (logMax - logMin)) * 100;
    }
    return ((val - min) / (max - min)) * 100;
  };
  const sliderToValue = (slider: number) => {
    if (log) {
      const logMin = Math.log10(min);
      const logMax = Math.log10(max);
      return Math.pow(10, logMin + (slider / 100) * (logMax - logMin));
    }
    return min + (slider / 100) * (max - min);
  };

  const handleMinChange = (newMin: number) => {
    if (onMinChange && newMin < max && newMin > 0) {
      onMinChange(newMin);
    }
  };

  const handleMaxChange = (newMax: number) => {
    if (onMaxChange && newMax > min) {
      onMaxChange(newMax);
    }
  };



  return (
    <div className="param-slider">
      {/* Parameter Label and Current Value */}
      {label && (
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-medium text-neutral-200">{label}</span>
          <span className="bg-neutral-700 text-white px-2 py-1 rounded text-xs font-medium">
            {transformValue ? transformValue(value) : value.toFixed(step < 1 ? 2 : 0)} {unit}
          </span>
        </div>
      )}



      {/* Min/Max Range Inputs and Slider Container */}
      <div className="flex items-center gap-3">
        <div className="flex flex-col items-center">
          <span className="text-xs text-neutral-400 mb-1">Min</span>
          <input
            type="number"
            value={transformValue ? 
              (isNaN(parseFloat(transformValue(min))) ? '' : parseFloat(transformValue(min))) : 
              (isNaN(min) ? '' : min)
            }
            onChange={(e) => {
              const newMin = parseFloat(e.target.value);
              if (!isNaN(newMin)) handleMinChange(newMin);
            }}
            className="w-16 bg-neutral-700 text-white text-xs px-2 py-1 rounded border border-neutral-600 focus:border-blue-500 focus:outline-none text-center"
            readOnly={readOnlyRange}
            step={step}
          />
        </div>
        
        <div className="flex-1 relative flex items-center">
          {/* Tick Marks - positioned above slider and closer */}
          {ticks && tickLabels && (
            <div className="absolute -top-4 left-0 right-0">
              <div className="flex justify-between px-1">
                {ticks.map((tick, i) => (
                  <div 
                    key={i} 
                    className="relative flex flex-col items-center group"
                  >
                    {/* Hover Label */}
                    <div className="absolute -top-8 opacity-0 group-hover:opacity-100 transition-opacity duration-200 bg-neutral-800 text-white text-xs px-2 py-1 rounded shadow-lg whitespace-nowrap pointer-events-none z-10">
                      {tickLabels[i]}
                    </div>
                    {/* Tick Mark - closer to slider */}
                    <div className="w-px h-3 bg-neutral-500 cursor-pointer"></div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Slider Track - better centered */}
          <div className="relative w-full h-2 bg-neutral-700 rounded-full">
            <div 
              className="absolute h-2 bg-neutral-500 rounded-full" 
              style={{ width: `${valueToSlider(value)}%` }}
            ></div>
          </div>
          
          {/* Main Slider */}
          <input 
            type="range" 
            min={sliderMin} 
            max={sliderMax} 
            step={log ? 0.1 : step}
            value={valueToSlider(value)} 
            onChange={(e) => onChange(sliderToValue(parseFloat(e.target.value)))} 
            className="absolute inset-0 w-full h-2 opacity-0 cursor-pointer"
          />
        </div>

        <div className="flex flex-col items-center">
          <span className="text-xs text-neutral-400 mb-1">Max</span>
          <input
            type="number"
            value={transformValue ? 
              (isNaN(parseFloat(transformValue(max))) ? '' : parseFloat(transformValue(max))) : 
              (isNaN(max) ? '' : max)
            }
            onChange={(e) => {
              const newMax = parseFloat(e.target.value);
              if (!isNaN(newMax)) handleMaxChange(newMax);
            }}
            className="w-16 bg-neutral-700 text-white text-xs px-2 py-1 rounded border border-neutral-600 focus:border-blue-500 focus:outline-none text-center"
            readOnly={readOnlyRange}
            step={step}
          />
        </div>
      </div>
    </div>
  );
}; 