export default function MathDetailsLoading() {
  return (
    <div className="h-screen flex items-center justify-center bg-neutral-950">
      <div className="text-center">
        <div className="w-16 h-16 border-4 border-orange-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-neutral-400">Loading math details...</p>
      </div>
    </div>
  );
}
