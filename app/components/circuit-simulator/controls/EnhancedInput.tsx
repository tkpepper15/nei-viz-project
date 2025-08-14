"use client";

import React, { useState, useCallback, useEffect } from 'react';

// Enhanced input component with validation, numerical arrows, and warnings
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
  // Local state for input value to allow typing without immediate validation
  const [inputValue, setInputValue] = useState(value.toString());
  const [isFocused, setIsFocused] = useState(false);
  
  // Update input value when prop value changes (external updates)
  useEffect(() => {
    if (!isFocused) {
      setInputValue(value.toString());
    }
  }, [value, isFocused]);
  
  // Validation state
  const isOutOfRange = value < min || value > max;
  const isValid = !isOutOfRange && !isNaN(value) && isFinite(value);
  
  // Input change handler - allows any input while typing
  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setInputValue(newValue);
    
    // Parse and validate the number
    const numValue = parseFloat(newValue);
    if (!isNaN(numValue) && isFinite(numValue)) {
      onChange(numValue);
    }
  }, [onChange]);
  
  // Handle focus events
  const handleFocus = useCallback(() => {
    setIsFocused(true);
  }, []);
  
  const handleBlur = useCallback(() => {
    setIsFocused(false);
    // Reset to current value if input is invalid
    const numValue = parseFloat(inputValue);
    if (isNaN(numValue) || !isFinite(numValue)) {
      setInputValue(value.toString());
    } else {
      setInputValue(numValue.toString());
    }
  }, [inputValue, value]);
  
  // Increment/decrement handlers
  const handleIncrement = useCallback(() => {
    if (disabled) return;
    const newValue = value + step;
    onChange(newValue);
  }, [value, step, onChange, disabled]);
  
  const handleDecrement = useCallback(() => {
    if (disabled) return;
    const newValue = value - step;
    onChange(newValue);
  }, [value, step, onChange, disabled]);
  
  // Keyboard handling for arrow keys
  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      handleIncrement();
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      handleDecrement();
    }
  }, [handleIncrement, handleDecrement]);
  
  // Get validation styling
  const getInputStyling = () => {
    if (disabled) {
      return 'bg-neutral-800 border-neutral-700 text-neutral-500 cursor-not-allowed';
    }
    if (isOutOfRange) {
      return 'bg-neutral-800 border-yellow-500 text-white focus:border-yellow-400 focus:ring-yellow-500/50';
    }
    if (!isValid && inputValue !== '') {
      return 'bg-neutral-800 border-red-500 text-white focus:border-red-400 focus:ring-red-500/50';
    }
    return 'bg-neutral-800 border-neutral-600 text-white focus:border-blue-400 focus:ring-blue-500/50';
  };
  
  const getValidationMessage = () => {
    if (!isValid && inputValue !== '' && !isNaN(parseFloat(inputValue))) {
      if (isOutOfRange) {
        return `⚠️ Value should be between ${min} and ${max}${unit ? ` ${unit}` : ''}`;
      }
    }
    if (!isFinite(parseFloat(inputValue)) && inputValue !== '') {
      return `❌ Please enter a valid number`;
    }
    return null;
  };
  
  const validationMessage = getValidationMessage();
  
  return (
    <div className={`space-y-2 ${className}`}>
      {/* Label */}
      <label htmlFor={id} className="block text-sm font-medium text-neutral-200">
        {label}{unit ? ` (${unit})` : ''}
      </label>
      
      {/* Input with numerical arrows */}
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
          className={`w-full px-3 py-2 pr-12 rounded-lg text-sm transition-colors focus:outline-none focus:ring-2 ${getInputStyling()}`}
        />
        
        {/* Numerical up/down arrows */}
        <div className="absolute right-1 top-1/2 -translate-y-1/2 flex flex-col">
          <button
            type="button"
            onClick={handleIncrement}
            disabled={disabled}
            className="w-8 h-4 flex items-center justify-center text-xs text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700 rounded-t transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            tabIndex={-1}
          >
            ▲
          </button>
          <button
            type="button"
            onClick={handleDecrement}
            disabled={disabled}
            className="w-8 h-4 flex items-center justify-center text-xs text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700 rounded-b transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            tabIndex={-1}
          >
            ▼
          </button>
        </div>
        
        {/* Validation indicator */}
        {(isOutOfRange || !isValid) && inputValue !== '' && (
          <div className={`absolute -top-1 -right-1 w-3 h-3 rounded-full animate-pulse ${
            !isValid ? 'bg-red-500' : 'bg-yellow-500'
          }`} />
        )}
      </div>
      
      {/* Validation message */}
      {validationMessage && (
        <div className={`text-xs ${!isValid ? 'text-red-400' : 'text-yellow-400'}`}>
          {validationMessage}
        </div>
      )}
      
      {/* Optional slider */}
      {showSlider && (
        <div className="px-2">
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={Math.min(Math.max(value, min), max)}
            onChange={(e) => onChange(parseFloat(e.target.value))}
            disabled={disabled}
            className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed slider-enhanced"
          />
          <div className="flex justify-between text-xs text-neutral-500 mt-1">
            <span>{min}</span>
            <span>{max}</span>
          </div>
        </div>
      )}
    </div>
  );
};