"use client";

import React from 'react';
import { ModelSnapshot, ImpedancePoint } from '../types';
import { calculateResnorm, groundTruthImpedance } from '../utils/resnorm';
import { formatValue } from '../utils/index';

interface ResnormDisplayProps {
  activeSnapshot: ModelSnapshot | null;
}

const ResnormDisplay: React.FC<ResnormDisplayProps> = ({ activeSnapshot }) => {
  if (!activeSnapshot) return null;

  const { resnorm, ter } = activeSnapshot;
  if (!resnorm) return null;

  // Get the active impedance point at 10kHz (ground truth frequency)
  const getActiveImpedance = (): ImpedancePoint | null => {
    if (!activeSnapshot || !activeSnapshot.data || activeSnapshot.data.length === 0) {
      return null;
    }
    
    // Find the data point closest to 10kHz
    const targetFreq = 10000;
    return activeSnapshot.data.find(point => point.frequency === targetFreq) || 
           activeSnapshot.data[0]; // Fallback to first point if exact match not found
  };
  
  // Calculate the resnorm if we have data
  const calculateResnormValue = (): string => {
    const impedancePoint = getActiveImpedance();
    if (!impedancePoint) {
      return 'N/A';
    }
    
    const resnorm = calculateResnorm(
      impedancePoint,
      groundTruthImpedance.real,
      groundTruthImpedance.imaginary,
      groundTruthImpedance.frequency
    );
    return resnorm.toExponential(3);
  };

  // Get formatted impedance components
  const formatImpedance = (): { real: string, imag: string } => {
    const impedancePoint = getActiveImpedance();
    if (!impedancePoint) {
      return { real: 'N/A', imag: 'N/A' };
    }
    
    return {
      real: impedancePoint.real.toFixed(3),
      imag: (-impedancePoint.imaginary).toFixed(3) // Display -Z''
    };
  };

  const impedance = formatImpedance();
  const resnormValue = calculateResnormValue();

  return (
    <div className="bg-white p-4 rounded-lg border border-[#E0E0E0]">
      <div className="mb-3">
        <h4 className="text-sm font-medium text-[#112E51] mb-1 flex items-center justify-between">
          <span>Impedance Values</span>
          <span className="text-xs font-normal text-primary">
            at {groundTruthImpedance.frequency < 1000 ? 
              groundTruthImpedance.frequency.toFixed(1) : 
              (groundTruthImpedance.frequency/1000).toFixed(1) + 'k'} Hz
          </span>
        </h4>
        <div className="flex justify-between items-center">
          <div>
            <p className="text-xs text-[#5B616B]">Current Model</p>
            <p className="text-sm">
              Z&Apos;: <span className="font-medium">{impedance.real} Ω</span>
            </p>
            <p className="text-sm">
              -Z&Apos;&Apos;: <span className="font-medium">{impedance.imag} Ω</span>
            </p>
          </div>
          <div>
            <p className="text-xs text-[#5B616B]">Ground Truth</p>
            <p className="text-sm">
            Z&Apos;: <span className="font-medium">{groundTruthImpedance.real.toFixed(3)} Ω</span>
            </p>
            <p className="text-sm">
            -Z&Apos;&Apos;: <span className="font-medium">{(-groundTruthImpedance.imaginary).toFixed(3)} Ω</span>
            </p>
          </div>
        </div>
      </div>
      
      <div className="pt-3 border-t">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-sm font-medium text-[#112E51]">
              Residual Norm
            </h4>
            <p className="text-xs text-[#5B616B]">Model fit quality (0 = perfect fit)</p>
          </div>
          <div className="text-right">
            {activeSnapshot ? (
              <p className={`text-lg font-medium ${
                parseFloat(resnormValue) < 0.1 ? 'text-green-600' :
                parseFloat(resnormValue) < 0.3 ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {resnormValue}
              </p>
            ) : (
              <p className="text-sm text-[#5B616B]">Select a state</p>
            )}
          </div>
        </div>
      </div>

      <div className="text-sm text-gray-600">
        <p>
          The model&apos;s resnorm value is {formatValue(resnorm, "")} which indicates the goodness of fit.
        </p>
        <p>
          A lower resnorm value doesn&apos;t necessarily mean a better model - it&apos;s important to consider the physical meaning.
        </p>
      </div>
      <div className="text-sm text-gray-600 mt-4">
        <p>
          The model&apos;s TER value is {formatValue(ter || 0, "Ω")} which represents the total electrode resistance.
        </p>
        <p>
          A higher TER doesn&apos;t necessarily mean a worse model - it&apos;s important to consider the experimental conditions.
        </p>
      </div>
    </div>
  );
};

export default ResnormDisplay;
