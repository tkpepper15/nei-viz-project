import type { NextConfig } from "next";
import path from 'path';

const nextConfig: NextConfig = {
  /* config options here */
  transpilePackages: ['@mui/material', '@mui/icons-material', '@mui/system'],
  webpack: (config) => {
    // Add a fallback for @emotion packages using our shim
    config.resolve.alias = {
      ...config.resolve.alias,
      '@emotion/styled': path.resolve('./src/app/emotion-shim.js'),
      '@emotion/serialize': path.resolve('./src/app/emotion-shim.js'),
    };
    return config;
  },
  distDir: '.next'
};

export default nextConfig;
