import CircuitSimulator from './components/CircuitSimulator';

export default function Home() {
  return (
    <main className="h-[100vh] w-screen overflow-hidden">
      <div className="h-full w-full max-w-[1920px] mx-auto">
        <CircuitSimulator />
      </div>
    </main>
  );
}
