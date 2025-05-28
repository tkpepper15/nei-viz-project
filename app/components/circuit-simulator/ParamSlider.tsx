import React from 'react';

interface ParamSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (value: number) => void;
  transformValue?: (value: number) => string;
}

export const ParamSlider: React.FC<ParamSliderProps> = ({ 
  label, 
  value, 
  min, 
  max, 
  step, 
  unit, 
  onChange,
  transformValue 
}) => {
  return (
    <div className="param-slider group">
      <div className="slider-label">
        <label className="slider-label-text group-hover:text-neutral-500 transition-colors">{label}</label>
        <span className="slider-value group-hover:bg-neutral-300 transition-colors">
          {transformValue ? transformValue(value) : value.toFixed(step < 1 ? 2 : 0)} {unit}
        </span>
      </div>
      <div className="relative pt-1">
        <input 
          type="range" 
          min={min} 
          max={max} 
          step={step} 
          value={value} 
          onChange={(e) => onChange(parseFloat(e.target.value))} 
          className="slider-input" 
        />
        
        {/* Range markers */}
        <div className="flex justify-between px-1 mt-1">
          <span className="text-[9px] text-neutral-500">{transformValue ? transformValue(min) : min}{unit}</span>
          <span className="text-[9px] text-neutral-500">{transformValue ? transformValue(max) : max}{unit}</span>
        </div>
      </div>
    </div>
  );
}; 