"use client";

import React, { useState, useCallback, useEffect } from 'react';

interface EnhancedInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  unit?: string;
  min: number;
  max: number;
  step: number;
  className?: string;
  showSlider?: boolean;
  disabled?: boolean;
  placeholder?: string;
  id?: string;
}

export const EnhancedInput: React.FC<EnhancedInputProps> = ({
  label,
  value,
  onChange,
  unit = "",
  min,
  max,
  step,
  className = "",
  showSlider = true,
  disabled = false,
  placeholder,
  id
}) => {
  const [inputValue, setInputValue] = useState(
    (value != null && !isNaN(value)) ? value.toString() : ''
  );
  const [isFocused, setIsFocused] = useState(false);

  useEffect(() => {
    if (!isFocused) {
      setInputValue(
        (value != null && !isNaN(value)) ? value.toString() : ''
      );
    }
  }, [value, isFocused]);

  const isOutOfRange = value < min || value > max;
  const isValid = !isOutOfRange && !isNaN(value) && isFinite(value);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setInputValue(newValue);
    const numValue = parseFloat(newValue);
    if (!isNaN(numValue) && isFinite(numValue)) {
      onChange(numValue);
    }
  }, [onChange]);

  const handleFocus = useCallback(() => setIsFocused(true), []);

  const handleBlur = useCallback(() => {
    setIsFocused(false);
    const numValue = parseFloat(inputValue);
    if (isNaN(numValue) || !isFinite(numValue)) {
      setInputValue((value != null && !isNaN(value)) ? value.toString() : '');
    } else {
      setInputValue(numValue.toString());
    }
  }, [inputValue, value]);

  const handleIncrement = useCallback(() => {
    if (disabled) return;
    onChange(value + step);
  }, [value, step, onChange, disabled]);

  const handleDecrement = useCallback(() => {
    if (disabled) return;
    onChange(value - step);
  }, [value, step, onChange, disabled]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowUp') { e.preventDefault(); handleIncrement(); }
    else if (e.key === 'ArrowDown') { e.preventDefault(); handleDecrement(); }
  }, [handleIncrement, handleDecrement]);

  const getInputStyling = () => {
    if (disabled) return 'bg-neutral-800 border-neutral-800 text-neutral-500 cursor-not-allowed';
    if (isOutOfRange) return 'bg-neutral-800 border-neutral-500 text-white focus:border-neutral-400 focus:ring-neutral-500/30';
    if (!isValid && inputValue !== '') return 'bg-neutral-800 border-danger text-white focus:border-danger focus:ring-danger/30';
    return 'bg-neutral-800 border-neutral-800 text-white focus:border-primary focus:ring-primary/20';
  };

  const getValidationMessage = () => {
    if (!isValid && inputValue !== '' && !isNaN(parseFloat(inputValue))) {
      if (isOutOfRange) return `Range: ${min}–${max}${unit ? ` ${unit}` : ''}`;
    }
    if (!isFinite(parseFloat(inputValue)) && inputValue !== '') return 'Enter a valid number';
    return null;
  };

  const validationMessage = getValidationMessage();

  return (
    <div className={`space-y-1.5 ${className}`}>
      {label && (
        <label htmlFor={id} className="block text-xs font-medium text-neutral-400">
          {label}{unit ? ` (${unit})` : ''}
        </label>
      )}

      <div className="relative">
        <input
          id={id}
          type="number"
          value={inputValue}
          onChange={handleInputChange}
          onFocus={handleFocus}
          onBlur={handleBlur}
          onKeyDown={handleKeyDown}
          step={step}
          disabled={disabled}
          placeholder={placeholder}
          className={`w-full px-2.5 py-1.5 pr-10 rounded text-xs transition-colors focus:outline-none focus:ring-1 border ${getInputStyling()}`}
        />

        <div className="absolute right-0.5 top-1/2 -translate-y-1/2 flex flex-col">
          <button
            type="button"
            onClick={handleIncrement}
            disabled={disabled}
            className="w-7 h-4 flex items-center justify-center text-neutral-500 hover:text-neutral-300 hover:bg-neutral-700 rounded-t transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            tabIndex={-1}
          >
            <svg className="w-2.5 h-2.5" fill="currentColor" viewBox="0 0 16 16">
              <path d="M7.247 4.86l-4.796 5.481c-.566.647-.106 1.659.753 1.659h9.592a1 1 0 0 0 .753-1.659l-4.796-5.48a1 1 0 0 0-1.506 0z"/>
            </svg>
          </button>
          <button
            type="button"
            onClick={handleDecrement}
            disabled={disabled}
            className="w-7 h-4 flex items-center justify-center text-neutral-500 hover:text-neutral-300 hover:bg-neutral-700 rounded-b transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            tabIndex={-1}
          >
            <svg className="w-2.5 h-2.5" fill="currentColor" viewBox="0 0 16 16">
              <path d="M7.247 11.14 2.451 5.658C1.885 5.013 2.345 4 3.204 4h9.592a1 1 0 0 1 .753 1.659l-4.796 5.48a1 1 0 0 1-1.506 0z"/>
            </svg>
          </button>
        </div>
      </div>

      {validationMessage && (
        <div className="text-xs text-neutral-500">{validationMessage}</div>
      )}

      {showSlider && (
        <div>
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={Math.min(Math.max(value, min), max)}
            onChange={(e) => onChange(parseFloat(e.target.value))}
            disabled={disabled}
            className="w-full h-1.5 bg-neutral-700 rounded appearance-none cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary"
          />
          <div className="flex justify-between text-xs text-neutral-600 mt-0.5">
            <span>{min}</span>
            <span>{max}</span>
          </div>
        </div>
      )}
    </div>
  );
};
