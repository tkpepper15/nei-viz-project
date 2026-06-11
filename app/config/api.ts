export const ML_API_BASE =
  process.env.NEXT_PUBLIC_ML_API_URL?.replace(/\/$/, '') ?? 'http://localhost:5003';
