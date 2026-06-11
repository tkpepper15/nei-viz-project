export default function SimulatorLoading() {
  return (
    <div className="h-screen flex items-center justify-center bg-neutral-950">
      <div className="h-0.5 w-32 bg-neutral-800 rounded-full overflow-hidden">
        <div className="h-full w-1/2 bg-neutral-600 rounded-full animate-pulse" />
      </div>
    </div>
  );
}
