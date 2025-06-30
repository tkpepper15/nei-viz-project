"use client";

import React, { useState, useRef, useCallback, useEffect } from 'react';

interface ResizableSplitPaneProps {
  children: React.ReactNode[];
  defaultSplitHeight?: number; // Height of bottom panel as percentage
  minTopHeight?: number;
  minBottomHeight?: number;
  className?: string;
  resizerClassName?: string;
}

export const ResizableSplitPane: React.FC<ResizableSplitPaneProps> = ({
  children,
  defaultSplitHeight = 35,
  minTopHeight = 200,
  minBottomHeight = 150,
  className = '',
  resizerClassName = ''
}) => {
  const [bottomHeight, setBottomHeight] = useState(defaultSplitHeight);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const resizerRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging || !containerRef.current) return;

    requestAnimationFrame(() => {
      if (!containerRef.current) return;

      const containerRect = containerRef.current.getBoundingClientRect();
      const containerHeight = containerRect.height;
      const mouseY = e.clientY - containerRect.top;
      
      // Calculate new bottom height as percentage
      const newBottomHeightPx = containerHeight - mouseY;
      const newBottomHeightPercent = (newBottomHeightPx / containerHeight) * 100;
      
      // Apply constraints
      const minBottomPercent = (minBottomHeight / containerHeight) * 100;
      const maxBottomPercent = 100 - (minTopHeight / containerHeight) * 100;
      
      const constrainedHeight = Math.max(
        minBottomPercent,
        Math.min(maxBottomPercent, newBottomHeightPercent)
      );
      
      setBottomHeight(constrainedHeight);
    });
  }, [isDragging, minTopHeight, minBottomHeight]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'ns-resize';
      document.body.style.userSelect = 'none';
      
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  const topHeight = 100 - bottomHeight;

  return (
    <div 
      ref={containerRef}
      className={`resizable-split-pane ${className}`}
    >
      {/* Top Section */}
      <div 
        className={`resizable-split-pane-top ${isDragging ? 'resizing' : ''}`}
        style={{ height: `${topHeight}%` }}
      >
        <div className="h-full">
          {children[0]}
        </div>
      </div>

      {/* Resizer */}
      <div
        ref={resizerRef}
        className={`resizable-split-pane-resizer ${resizerClassName} ${isDragging ? 'dragging' : ''}`}
        onMouseDown={handleMouseDown}
      />

      {/* Bottom Section */}
      <div 
        className={`resizable-split-pane-bottom ${isDragging ? 'resizing' : ''}`}
        style={{ height: `${bottomHeight}%` }}
      >
        <div className="h-full">
          {children[1]}
        </div>
      </div>
    </div>
  );
}; 