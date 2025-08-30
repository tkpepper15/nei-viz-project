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
  formatValue?: (value: number) => string;
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
  className = "",
  formatValue
}) => {
  const minRef = useRef<HTMLInputElement>(null);
  const maxRef = useRef<HTMLInputElement>(null);
  const rangeRef = useRef<HTMLDivElement>(null);

  // Convert value to percentage
  const getPercent = useCallback((value: number) => 
    ((value - min) / (max - min)) * 100, 
    [min, max]
  );

  // Apple-style dual slider handlers with proper thumb behavior
  const handleMinThumbChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = Number(e.target.value);
    // Min thumb can't go above max value
    const constrainedValue = Math.min(newValue, maxValue);
    onChange(constrainedValue, maxValue);
  }, [maxValue, onChange]);

  const handleMaxThumbChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = Number(e.target.value);
    // Max thumb can't go below min value
    const constrainedValue = Math.max(newValue, minValue);
    onChange(minValue, constrainedValue);
  }, [minValue, onChange]);


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
      {(label || (minValue !== 0 || maxValue !== 0)) && (
        <div className="flex items-center justify-between">
          {label && (
            <label className="text-sm font-medium text-neutral-200">
              {label}
            </label>
          )}
          {(minValue !== 0 || maxValue !== 0) && (
            <div className="text-sm text-neutral-400">
              {formatValue ? formatValue(minValue) : minValue} - {formatValue ? formatValue(maxValue) : maxValue}{unit ? ` ${unit}` : ''}
            </div>
          )}
        </div>
      )}

      {/* Dual Range Container */}
      <div className="dual-range-container">
        {/* Min thumb slider */}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={isNaN(minValue) ? min : minValue}
          ref={minRef}
          onChange={handleMinThumbChange}
          className="dual-range-thumb dual-range-thumb-min"
        />
        
        {/* Max thumb slider */}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={isNaN(maxValue) ? max : maxValue}
          ref={maxRef}
          onChange={handleMaxThumbChange}
          className="dual-range-thumb dual-range-thumb-max"
        />

        {/* Slider track */}
        <div className="dual-range-slider">
          <div className="dual-range-track" />
          <div ref={rangeRef} className="dual-range-highlight" />
        </div>
      </div>

      {/* Slider bounds labels - always show the full range */}
      <div className="flex justify-between text-xs text-neutral-500 px-2 mt-1">
        <span>{formatValue ? formatValue(min) : min}{unit ? ` ${unit}` : ''}</span>
        <span>{formatValue ? formatValue(max) : max}{unit ? ` ${unit}` : ''}</span>
      </div>

    </div>
  );
};