import React from 'react';
import { ModelSnapshot } from './utils/types';

interface BaseSpiderPlotProps {
  meshItems: ModelSnapshot[];
  referenceId?: string;
  opacityFactor?: number;
  logScalar?: number;
}

export const BaseSpiderPlot: React.FC<BaseSpiderPlotProps> = ({ 
  meshItems, 
  referenceId, 
  opacityFactor = 0.7,
  logScalar = 1.0
}) => {
  // Separate reference model from other items
  const referenceModel = meshItems.find(item => item.id === referenceId);
  const regularItems = meshItems.filter(item => item.id !== referenceId);
  
  // Define CSS animations for the component
  const pulseAnimation = `
    @keyframes subtlePulse {
      0% { opacity: 0.95; filter: drop-shadow(0 0 1px rgba(255, 255, 255, 0.4)); }
      50% { opacity: 1; filter: drop-shadow(0 0 2px rgba(255, 255, 255, 0.5)); }
      100% { opacity: 0.95; filter: drop-shadow(0 0 1px rgba(255, 255, 255, 0.4)); }
    }
    .reference-polygon {
      animation: subtlePulse 4s ease-in-out infinite;
    }
  `;
  
  // Scientific figure formatting for labels - shared across components
  const labelStyle = {
    fontFamily: '"Arial", sans-serif',
    fontSize: '10px',
    fontWeight: '500',
    opacity: 0.8
  };
  
  // Sort regular items by resnorm (worst first, best last)
  const sortedItems = [...regularItems].sort((a, b) => {
    if (!a.resnorm && !b.resnorm) return 0;
    if (!a.resnorm) return -1;
    if (!b.resnorm) return 1;
    return b.resnorm - a.resnorm;
  });
  
  // Helper functions for coordinate calculations
  const calcPosition = (param: number, max: number = 1, min: number = 0) => {
    // Normalize value between 0 and 1
    let normalized = (param - min) / (max - min);
    
    // Apply logScalar to amplify differences
    if (logScalar && logScalar > 1) {
      normalized = Math.pow(normalized, 1/logScalar);
    }
    
    // Return value between center (0) and max radius (200)
    return Math.max(0, Math.min(1, normalized)) * 200;
  };
  
  const toCartesian = (radius: number, angleDegrees: number) => {
    const angleRadians = (angleDegrees * Math.PI) / 180;
    return {
      x: 250 + radius * Math.cos(angleRadians),
      y: 250 + radius * Math.sin(angleRadians)
    };
  };
  
  // Function to generate pentagon points for an item
  const generatePointsForItem = (item: ModelSnapshot) => {
    if (!item || !item.parameters) return null;
    
    try {
      // Define proper ranges for each parameter to ensure consistent scaling
      const rsRange = { min: 10, max: 30 };
      const raRange = { min: 100, max: 1000 };
      const rbRange = { min: 100, max: 1000 };
      const caRange = { min: 1e-6, max: 5e-6 };
      const cbRange = { min: 1e-6, max: 5e-6 };
      
      // Get parameters with fallbacks to reasonable defaults
      const rs = item.parameters.Rs !== undefined ? item.parameters.Rs : 24;
      const ra = item.parameters.Ra !== undefined ? item.parameters.Ra : 500;
      const rb = item.parameters.Rb !== undefined ? item.parameters.Rb : 500;
      const ca = item.parameters.Ca !== undefined ? item.parameters.Ca : 3e-6;
      const cb = item.parameters.Cb !== undefined ? item.parameters.Cb : 3e-6;
      
      // Calculate scaled positions for each parameter (0-1 range)
      // Use non-linear scaling for capacitance to enhance visual differences
      const rsScaled = (rs - rsRange.min) / (rsRange.max - rsRange.min);
      const raScaled = (ra - raRange.min) / (raRange.max - raRange.min);
      const rbScaled = (rb - rbRange.min) / (rbRange.max - rbRange.min);
      
      // Enhanced capacitance scaling - using square root scaling to enhance visual differences
      // This makes differences in capacitance values more apparent on the plot
      const caScaledLinear = (ca - caRange.min) / (caRange.max - caRange.min);
      const cbScaledLinear = (cb - cbRange.min) / (cbRange.max - cbRange.min);
      
      // Apply square root scaling for capacitance to make small differences more visible
      const caScaled = Math.sqrt(caScaledLinear);
      const cbScaled = Math.sqrt(cbScaledLinear);
      
      // Convert to polar coordinates for pentagon points
      const rsPoint = toCartesian(calcPosition(rsScaled), -90);
      const raPoint = toCartesian(calcPosition(raScaled), -90 + 72);
      const rbPoint = toCartesian(calcPosition(rbScaled), -90 + 144);
      const caPoint = toCartesian(calcPosition(caScaled), -90 + 216);
      const cbPoint = toCartesian(calcPosition(cbScaled), -90 + 288);
      
      // Debug parameters
      if (item.id === 'dynamic-reference') {
        console.log('Reference model parameters:', {
          rs, ra, rb, ca, cb,
          rsScaled, raScaled, rbScaled, caScaled, cbScaled
        });
      }
      
      return {
        points: [
          `${rsPoint.x},${rsPoint.y}`,
          `${raPoint.x},${raPoint.y}`,
          `${rbPoint.x},${rbPoint.y}`,
          `${caPoint.x},${caPoint.y}`,
          `${cbPoint.x},${cbPoint.y}`
        ].join(' '),
        corners: [rsPoint, raPoint, rbPoint, caPoint, cbPoint]
      };
    } catch (err) {
      console.error('Error generating points for item:', err);
      return null;
    }
  };
  
  // Always show the spider plot, even with no data
  return (
    <div className="spider-plot h-full w-full">
      <style>{pulseAnimation}</style>
      <div className="h-full w-full flex items-center justify-center">
        <div className="spider-visualization h-full w-full">
          <svg 
            viewBox="0 0 500 500" 
            className="spider-svg h-full w-full"
            style={{ 
              stroke: '#64748b', 
              fill: '#64748b',
              backgroundColor: '#0f172a'
            }}
          >
            {/* Parameter axes */}
            {(() => {
              // Calculate angles for perfect pentagon (72 degrees apart)
              const angleRs = -90; // Top (270 degrees in standard position)
              const angleRa = -90 + 72; // Top-right
              const angleRb = -90 + 144; // Bottom-right
              const angleCa = -90 + 216; // Bottom-left
              const angleCb = -90 + 288; // Top-left
              
              // Distance from center to edge of the plot
              const axisLength = 220;
              
              const rsEnd = toCartesian(axisLength, angleRs);
              const raEnd = toCartesian(axisLength, angleRa);
              const rbEnd = toCartesian(axisLength, angleRb);
              const caEnd = toCartesian(axisLength, angleCa);
              const cbEnd = toCartesian(axisLength, angleCb);
              
              // Create tick marks along Rs axis to better indicate scale
              const gridLevels = [50, 100, 150, 200]; // Match the grid level sizes
              const tickMarks = gridLevels.map(size => {
                const point = toCartesian(size, angleRs);
                const perpAngle = angleRs + 90; // Perpendicular to the axis
                const tickLength = 4; // Length of tick mark
                
                const tick1X = point.x + Math.cos(perpAngle * Math.PI / 180) * tickLength;
                const tick1Y = point.y + Math.sin(perpAngle * Math.PI / 180) * tickLength;
                const tick2X = point.x - Math.cos(perpAngle * Math.PI / 180) * tickLength;
                const tick2Y = point.y - Math.sin(perpAngle * Math.PI / 180) * tickLength;
                
                return { point, tick1X, tick1Y, tick2X, tick2Y };
              });
              
              // Distance from center to label position (slightly beyond axis end)
              const labelDistance = 230;
              const raPos = toCartesian(labelDistance, angleRa);
              const rbPos = toCartesian(labelDistance, angleRb);
              const caPos = toCartesian(labelDistance, angleCa);
              const cbPos = toCartesian(labelDistance, angleCb);
              
              // Determine text anchor based on position
              const getTextAnchor = (angle: number): string => {
                if (angle > -135 && angle < -45) return "middle"; // top
                if (angle >= -45 && angle < 45) return "start";   // right
                if (angle >= 45 && angle < 135) return "middle";  // bottom
                if (angle >= 135 && angle < 225) return "end";    // left
                return "end";                                     // top-left
              };
              
              return (
                <>
                  {/* Draw axes */}
                  <line x1="250" y1="250" x2={rsEnd.x} y2={rsEnd.y} stroke="#64748b" strokeWidth="1.5" strokeOpacity="0.6" />
                  <line x1="250" y1="250" x2={raEnd.x} y2={raEnd.y} stroke="#64748b" strokeWidth="1.5" strokeOpacity="0.6" />
                  <line x1="250" y1="250" x2={rbEnd.x} y2={rbEnd.y} stroke="#64748b" strokeWidth="1.5" strokeOpacity="0.6" />
                  <line x1="250" y1="250" x2={caEnd.x} y2={caEnd.y} stroke="#64748b" strokeWidth="1.5" strokeOpacity="0.6" />
                  <line x1="250" y1="250" x2={cbEnd.x} y2={cbEnd.y} stroke="#64748b" strokeWidth="1.5" strokeOpacity="0.6" />
                  
                  {/* Draw tick marks */}
                  {tickMarks.map((tick, idx) => (
                    <g key={`tick-${idx}`}>
                      <line 
                        x1={tick.tick1X} 
                        y1={tick.tick1Y} 
                        x2={tick.tick2X} 
                        y2={tick.tick2Y} 
                        stroke="#64748b" 
                        strokeWidth="1" 
                        strokeOpacity="0.7" 
                      />
                    </g>
                  ))}
                  
                  {/* Draw labels */}
                  <text 
                    x="250" 
                    y="20" 
                    textAnchor="middle"
                    fill="white" 
                    fontFamily={labelStyle.fontFamily}
                    fontSize="12px"
                    fontWeight="600" 
                    style={{ 
                      filter: 'drop-shadow(0 0 2px rgba(0, 0, 0, 0.7))'
                    }}
                  >
                    R<tspan baselineShift="sub" fontSize="9px">s</tspan>
                  </text>
                  <text 
                    x={raPos.x + 3} 
                    y={raPos.y} 
                    textAnchor={getTextAnchor(angleRa)}
                    fill="white" 
                    fontFamily={labelStyle.fontFamily}
                    fontSize="12px"
                    fontWeight="600" 
                    style={{ 
                      filter: 'drop-shadow(0 0 2px rgba(0, 0, 0, 0.7))'
                    }}
                  >
                    R<tspan baselineShift="sub" fontSize="9px">a</tspan>
                  </text>
                  <text 
                    x={rbPos.x + 3} 
                    y={rbPos.y} 
                    textAnchor={getTextAnchor(angleRb)}
                    fill="white" 
                    fontFamily={labelStyle.fontFamily}
                    fontSize="12px"
                    fontWeight="600" 
                    style={{ 
                      filter: 'drop-shadow(0 0 2px rgba(0, 0, 0, 0.7))'
                    }}
                  >
                    R<tspan baselineShift="sub" fontSize="9px">b</tspan>
                  </text>
                  <text 
                    x={caPos.x} 
                    y={caPos.y} 
                    textAnchor={getTextAnchor(angleCa)}
                    fill="white" 
                    fontFamily={labelStyle.fontFamily}
                    fontSize="12px"
                    fontWeight="600" 
                    style={{ 
                      filter: 'drop-shadow(0 0 2px rgba(0, 0, 0, 0.7))'
                    }}
                  >
                    C<tspan baselineShift="sub" fontSize="9px">a</tspan>
                  </text>
                  <text 
                    x={cbPos.x} 
                    y={cbPos.y} 
                    textAnchor={getTextAnchor(angleCb)}
                    fill="white" 
                    fontFamily={labelStyle.fontFamily}
                    fontSize="12px"
                    fontWeight="600" 
                    style={{ 
                      filter: 'drop-shadow(0 0 2px rgba(0, 0, 0, 0.7))'
                    }}
                  >
                    C<tspan baselineShift="sub" fontSize="9px">b</tspan>
                  </text>
                </>
              );
            })()}
            
            {/* Grid pentagon levels with subtle tick marks */}
            {(() => {
              // Calculate positions for pentagon at different sizes
              const createPentagonPoints = (size: number) => {
                const points = [];
                for (let i = 0; i < 5; i++) {
                  const angle = (-90 + i * 72) * (Math.PI / 180);
                  points.push({
                    x: 250 + size * Math.cos(angle),
                    y: 250 + size * Math.sin(angle)
                  });
                }
                return points;
              };
              
              // Create multiple grid levels
              const gridLevels = [50, 100, 150, 200];
              
              return (
                <>
                  {/* Small center circle to mark origin (0) */}
                  <circle 
                    cx="250" 
                    cy="250" 
                    r="2" 
                    fill="#64748b" 
                    fillOpacity="0.6" 
                  />
                  <text
                    x="250"
                    y="266"
                    fill="#94a3b8"
                    fontSize={labelStyle.fontSize}
                    fontFamily={labelStyle.fontFamily}
                    fontWeight={labelStyle.fontWeight}
                    textAnchor="middle"
                    opacity={labelStyle.opacity}
                  >
                    0
                  </text>
                
                  {gridLevels.map((size, idx) => {
                    const points = createPentagonPoints(size);
                    const pointsStr = points.map(p => `${p.x},${p.y}`).join(' ');
                    
                    // Calculate midpoint between Rb and Ca axes (bottom side of pentagon)
                    const angle1 = (-90 + 144) * (Math.PI / 180); // Rb angle
                    const angle2 = (-90 + 216) * (Math.PI / 180); // Ca angle
                    const midAngle = (angle1 + angle2) / 2;
                    
                    // Position labels with consistent offsets and adjust for visibility
                    const offsetFactor = idx === 0 ? 13 : 15; // Smaller offset for innermost ring
                    const labelX = 250 + size * Math.cos(midAngle);
                    const labelY = 250 + size * Math.sin(midAngle) + offsetFactor;
                    
                    return (
                      <g key={`grid-${idx}`}>
                        <polygon
                          points={pointsStr}
                          fill="none"
                          stroke="#64748b"
                          strokeOpacity={idx === gridLevels.length - 1 ? 0.5 : 0.3}
                          strokeWidth={idx === gridLevels.length - 1 ? 1.5 : 1}
                          className="grid-pentagon"
                        />
                        
                        {/* Add proportion label with adjusted position */}
                        <text
                          x={labelX + 20}
                          y={labelY - 260}
                          fill="#94a3b8"
                          fontSize={labelStyle.fontSize}
                          fontFamily={labelStyle.fontFamily}
                          fontWeight={labelStyle.fontWeight}
                          textAnchor="middle"
                          opacity={labelStyle.opacity}
                          style={{
                            filter: idx === 0 ? 'drop-shadow(0 0 1px rgba(0, 0, 0, 0.8))' : 'none'
                          }}
                        >
                          {idx === 3 ? "1.00" : idx === 2 ? "0.75" : idx === 1 ? "0.50" : "0.25"}
                        </text>
                      </g>
                    );
                  })}
                </>
              );
            })()}
            
            {/* STEP 1: Render regular points first */}
            {sortedItems.map((item, index) => {
              const pointsData = generatePointsForItem(item);
              if (!pointsData) return null;
              
              const opacity = opacityFactor * (item.opacity || 0.7);
              const strokeColor = item.color || "#3B82F6"; // Use assigned color
              let strokeWidth = 2;
              let strokeOpacity = opacity;
              let glowEffect = "";
              let strokeDasharray = "";
              
              // Add visual effects based on item color
              if (strokeColor === "#10B981") { // Very Good - Green
                strokeWidth = 2;
                strokeOpacity = Math.min(1, opacity * 1.2);
                glowEffect = "drop-shadow(0 0 4px rgba(16, 185, 129, 0.8))";
              } 
              else if (strokeColor === "#3B82F6") { // Good - Blue
                strokeWidth = 1.5;
                strokeOpacity = opacity;
                glowEffect = "drop-shadow(0 0 3px rgba(59, 130, 246, 0.6))";
              } 
              else if (strokeColor === "#F59E0B") { // Moderate - Amber
                strokeWidth = 1.5;
                strokeOpacity = opacity * 0.8;
                strokeDasharray = "3,2";
              } 
              else if (strokeColor === "#EF4444") { // Poor - Red
                strokeWidth = 1;
                strokeOpacity = opacity * 0.6;
                strokeDasharray = "2,2";
              }
              
              return (
                <g key={`model-${index}`}>
                  {/* Draw polygon outline */}
                  <polygon
                    points={pointsData.points}
                    fill="none"
                    stroke={strokeColor}
                    strokeWidth={strokeWidth}
                    strokeOpacity={strokeOpacity}
                    strokeLinejoin="round"
                    strokeDasharray={strokeDasharray}
                    style={{
                      filter: glowEffect,
                      transition: 'all 0.3s ease',
                    }}
                  />
                  
                  {/* Draw corner points for very good fits */}
                  {strokeColor === "#10B981" && pointsData.corners.map((point, cornerIdx) => (
                    <circle
                      key={`corner-${index}-${cornerIdx}`}
                      cx={point.x}
                      cy={point.y}
                      r={3}
                      fill={strokeColor}
                      stroke="none"
                      style={{ 
                        filter: 'drop-shadow(0 0 3px rgba(0, 0, 0, 0.8))',
                        opacity: 0.8
                      }}
                    />
                  ))}
                </g>
              );
            })}
            
            {/* STEP 2: Render reference model on top */}
            {referenceModel && (
              <>
                {/* Reference model's polygon */}
                {(() => {
                  const pointsData = generatePointsForItem(referenceModel);
                  if (!pointsData) return null;
                  
                  return (
                    <>
                      {/* Highlight area */}
                      <polygon
                        points={pointsData.points}
                        fill="none"
                        stroke="white"
                        strokeWidth="3.5"
                        strokeOpacity={1}
                        className="reference-polygon"
                        style={{
                          filter: 'drop-shadow(0 0 3px rgba(255, 255, 255, 0.5))'
                        }}
                      />
                      
                      {/* Corner points for emphasis */}
                      {pointsData.corners.map((corner, idx) => (
                        <g key={`ref-point-${idx}`}>
                          <circle
                            cx={corner.x}
                            cy={corner.y}
                            r="5"
                            fill="white"
                            stroke="none"
                            className="reference-point"
                            style={{
                              filter: 'drop-shadow(0 0 2px rgba(255, 255, 255, 0.7))',
                            }}
                          />
                          {/* Parameter value labels */}
                          {idx === 0 && (
                            <text
                              x={corner.x}
                              y={corner.y - 12}
                              textAnchor="middle"
                              fill="white"
                              fontSize="12"
                              fontWeight="bold"
                              style={{ filter: 'drop-shadow(0 0 2px rgba(0, 0, 0, 0.8))' }}
                            >
                              {referenceModel.parameters.Rs.toFixed(0)}Ω
                            </text>
                          )}
                          {idx === 1 && (
                            <text
                              x={corner.x + 12}
                              y={corner.y}
                              textAnchor="start"
                              fill="white"
                              fontSize="12"
                              fontWeight="bold"
                              style={{ filter: 'drop-shadow(0 0 2px rgba(0, 0, 0, 0.8))' }}
                            >
                              {referenceModel.parameters.Ra.toFixed(0)}Ω
                            </text>
                          )}
                          {idx === 2 && (
                            <text
                              x={corner.x + 12}
                              y={corner.y}
                              textAnchor="start"
                              fill="white"
                              fontSize="12"
                              fontWeight="bold"
                              style={{ filter: 'drop-shadow(0 0 2px rgba(0, 0, 0, 0.8))' }}
                            >
                              {referenceModel.parameters.Rb.toFixed(0)}Ω
                            </text>
                          )}
                          {idx === 3 && (
                            <text
                              x={corner.x - 12}
                              y={corner.y}
                              textAnchor="end"
                              fill="white"
                              fontSize="12"
                              fontWeight="bold"
                              style={{ filter: 'drop-shadow(0 0 2px rgba(0, 0, 0, 0.8))' }}
                            >
                              {(referenceModel.parameters.Ca * 1e6).toFixed(1)}μF
                            </text>
                          )}
                          {idx === 4 && (
                            <text
                              x={corner.x - 12}
                              y={corner.y}
                              textAnchor="end"
                              fill="white"
                              fontSize="12"
                              fontWeight="bold"
                              style={{ filter: 'drop-shadow(0 0 2px rgba(0, 0, 0, 0.8))' }}
                            >
                              {(referenceModel.parameters.Cb * 1e6).toFixed(1)}μF
                            </text>
                          )}
                        </g>
                      ))}
                    </>
                  );
                })()}
              </>
            )}
          </svg>
        </div>
      </div>
    </div>
  );
}; 