"use client";

import React, { useRef, useEffect, useCallback } from 'react';

interface DualRangeSliderProps {
  label: string;
  minValue: number;
  maxValue: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (min: number, max: number) => void;
  className?: string;
}

export const DualRangeSlider: React.FC<DualRangeSliderProps> = ({
  label,
  minValue,
  maxValue,
  min,
  max,
  step,
  unit = "",
  onChange,
  className = ""
}) => {
  const minRef = useRef<HTMLInputElement>(null);
  const maxRef = useRef<HTMLInputElement>(null);
  const rangeRef = useRef<HTMLDivElement>(null);

  // Convert value to percentage
  const getPercent = useCallback((value: number) => 
    ((value - min) / (max - min)) * 100, 
    [min, max]
  );

  // Handle min value change with constraints
  const handleMinChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = Math.min(Number(e.target.value), maxValue - step);
    onChange(value, maxValue);
  }, [maxValue, step, onChange]);

  // Handle max value change with constraints
  const handleMaxChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = Math.max(Number(e.target.value), minValue + step);
    onChange(minValue, value);
  }, [minValue, step, onChange]);

  // Handle number input changes
  const handleMinInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newMin = parseFloat(e.target.value) || minValue;
    onChange(newMin, maxValue);
  }, [minValue, maxValue, onChange]);

  const handleMaxInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newMax = parseFloat(e.target.value) || maxValue;
    onChange(minValue, newMax);
  }, [minValue, maxValue, onChange]);

  // Update range highlight when values change
  useEffect(() => {
    if (rangeRef.current) {
      const minPercent = getPercent(minValue);
      const maxPercent = getPercent(maxValue);

      rangeRef.current.style.left = `${minPercent}%`;
      rangeRef.current.style.width = `${maxPercent - minPercent}%`;
    }
  }, [minValue, maxValue, getPercent]);

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Label and Values */}
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-neutral-200">
          {label}{unit ? ` (${unit})` : ''}
        </label>
        <div className="text-sm text-neutral-400">
          {minValue} - {maxValue}{unit ? ` ${unit}` : ''}
        </div>
      </div>

      {/* Dual Range Container */}
      <div className="dual-range-container">
        {/* Min range input */}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={minValue}
          ref={minRef}
          onChange={handleMinChange}
          className="dual-range-thumb dual-range-thumb-min"
        />
        
        {/* Max range input */}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={maxValue}
          ref={maxRef}
          onChange={handleMaxChange}
          className="dual-range-thumb dual-range-thumb-max"
        />

        {/* Slider track */}
        <div className="dual-range-slider">
          <div className="dual-range-track" />
          <div ref={rangeRef} className="dual-range-highlight" />
        </div>
      </div>

      {/* Min/Max boundary labels */}
      <div className="flex justify-between text-xs text-neutral-500 px-2">
        <span>{min}{unit ? ` ${unit}` : ''}</span>
        <span>{max}{unit ? ` ${unit}` : ''}</span>
      </div>

      {/* Individual value inputs */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-neutral-400 mb-1">Min Value</label>
          <input
            type="number"
            value={minValue}
            onChange={handleMinInputChange}
            step={step}
            className="w-full px-2 py-1 text-sm bg-neutral-800 border border-neutral-600 rounded text-white focus:border-blue-400 focus:outline-none"
          />
        </div>
        <div>
          <label className="block text-xs text-neutral-400 mb-1">Max Value</label>
          <input
            type="number"
            value={maxValue}
            onChange={handleMaxInputChange}
            step={step}
            className="w-full px-2 py-1 text-sm bg-neutral-800 border border-neutral-600 rounded text-white focus:border-blue-400 focus:outline-none"
          />
        </div>
      </div>
    </div>
  );
};