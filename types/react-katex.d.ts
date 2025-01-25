declare module 'react-katex' {
  import { ReactNode } from 'react';

  interface KaTeXProps {
    children: string | ReactNode;
    math?: string;
    block?: boolean;
    errorColor?: string;
    renderError?: (error: Error | TypeError) => ReactNode;
    settings?: Record<string, unknown>;
    as?: string | React.ComponentType;
  }

  export const InlineMath: React.FC<KaTeXProps>;
  export const BlockMath: React.FC<KaTeXProps>;
} 