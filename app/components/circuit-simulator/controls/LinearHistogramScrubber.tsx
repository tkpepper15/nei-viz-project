"use client";

import React, { useRef, useCallback, useState } from 'react';

interface LinearHistogramScrubberProps {
  min: number;
  max: number;
  value: number | null;
  onChange: (value: number) => void;
  width?: number;
  className?: string;
  disabled?: boolean;
}

const LinearHistogramScrubber: React.FC<LinearHistogramScrubberProps> = ({
  min,
  max,
  value,
  onChange,
  width = 360,
  className = "",
  disabled = false
}) => {
  const sliderRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showValueTooltip, setShowValueTooltip] = useState(false);

  // Calculate the slider value (0-1000 for precision) from resnorm value
  const sliderValue = value !== null && value >= min && value <= max && max > min
    ? ((value - min) / (max - min)) * 1000
    : 0;

  // Convert slider position back to resnorm value
  const sliderToResnorm = useCallback((sliderVal: number) => {
    if (max <= min) return min; // Prevent division by zero
    return min + (sliderVal / 1000) * (max - min);
  }, [min, max]);

  const handleSliderChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const sliderVal = parseFloat(e.target.value);
    const resnormValue = sliderToResnorm(sliderVal);
    onChange(resnormValue);
  }, [sliderToResnorm, onChange]);

  const handleMouseDown = useCallback(() => {
    setIsDragging(true);
    setShowValueTooltip(true);
  }, []);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    // Keep tooltip visible for a moment after drag ends
    setTimeout(() => setShowValueTooltip(false), 1000);
  }, []);

  const handleMouseEnter = useCallback(() => {
    setShowValueTooltip(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    if (!isDragging) {
      setShowValueTooltip(false);
    }
  }, [isDragging]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (disabled) return;
    
    let stepSize = 1; // Default step
    if (e.shiftKey) stepSize = 10; // Larger steps with Shift
    if (e.ctrlKey || e.metaKey) stepSize = 0.1; // Smaller steps with Ctrl/Cmd
    
    let newSliderValue = sliderValue;
    
    switch (e.key) {
      case 'ArrowLeft':
      case 'ArrowDown':
        newSliderValue = Math.max(0, sliderValue - stepSize);
        break;
      case 'ArrowRight':
      case 'ArrowUp':
        newSliderValue = Math.min(1000, sliderValue + stepSize);
        break;
      case 'Home':
        newSliderValue = 0;
        break;
      case 'End':
        newSliderValue = 1000;
        break;
      default:
        return;
    }
    
    e.preventDefault();
    const resnormValue = sliderToResnorm(newSliderValue);
    onChange(resnormValue);
  }, [disabled, sliderValue, sliderToResnorm, onChange]);

  const formatResnormValue = useCallback((val: number) => {
    return val.toExponential(2);
  }, []);

  return (
    <div 
      ref={containerRef}
      className={`relative ${className}`}
      style={{ width: width }}
    >
      {/* Scrubber Label */}
      <div className="flex items-center justify-between mb-2">
        <label className="text-xs font-medium text-neutral-300">
          Linear Scrubber
        </label>
        {value !== null && (
          <div className="text-xs text-neutral-400 font-mono">
            {formatResnormValue(value)}
          </div>
        )}
      </div>

      {/* Scrubber Container */}
      <div className="relative">
        {/* Track Background */}
        <div className="absolute top-1/2 transform -translate-y-1/2 w-full h-2 bg-neutral-700 rounded-full" />
        
        {/* Active Track (from min to current value) */}
        {value !== null && (
          <div 
            className="absolute top-1/2 transform -translate-y-1/2 h-2 bg-yellow-500 rounded-full transition-all duration-150"
            style={{ 
              width: `${(sliderValue / 1000) * 100}%`
            }}
          />
        )}

        {/* Main Slider Input */}
        <input
          ref={sliderRef}
          type="range"
          min={0}
          max={1000}
          step={0.1}
          value={sliderValue}
          onChange={handleSliderChange}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          className={`
            slider-enhanced w-full relative z-10 bg-transparent
            ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          `}
          style={{ 
            background: 'transparent',
            height: '20px'
          }}
        />

        {/* Value Tooltip */}
        {showValueTooltip && value !== null && (
          <div 
            className="absolute -top-8 bg-neutral-800 text-white text-xs px-2 py-1 rounded shadow-lg whitespace-nowrap pointer-events-none z-20 transform -translate-x-1/2 transition-opacity duration-200"
            style={{ 
              left: `${(sliderValue / 1000) * 100}%`,
              opacity: showValueTooltip ? 1 : 0
            }}
          >
            {formatResnormValue(value)}
            <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-2 border-r-2 border-t-2 border-transparent border-t-neutral-800" />
          </div>
        )}
      </div>

      {/* Range Labels */}
      <div className="flex justify-between mt-1 text-xs text-neutral-500">
        <span>{formatResnormValue(min)}</span>
        <span>{formatResnormValue(max)}</span>
      </div>
    </div>
  );
};

export default LinearHistogramScrubber;