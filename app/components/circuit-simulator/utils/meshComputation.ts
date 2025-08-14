import { ModelSnapshot, StateFolder } from '../types';
import { CircuitParameters } from '../types/parameters';

interface MeshResponsePoint {
  parameters: CircuitParameters;
  resnorm: number;
  alpha: number;
}

/**
 * Appends a log message with timestamp
 */
export const createAppendLogFunction = (
  setComputationLogs: React.Dispatch<React.SetStateAction<string[]>>
) => {
  return (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    const logMessage = `[${timestamp}] ${message}`;
    setComputationLogs(prev => [...prev, logMessage]);
    // Auto-scroll the log container to the bottom
    const logContainer = document.getElementById('computation-log-container');
    if (logContainer) {
      setTimeout(() => {
        logContainer.scrollTop = logContainer.scrollHeight;
      }, 100);
    }
  };
};

/**
 * Compute regression mesh for parameter space exploration
 */
export const computeRegressionMesh = async (
  activeSnapshot: ModelSnapshot | null,
  setComputationLogs: React.Dispatch<React.SetStateAction<string[]>>,
  setShowNotification: React.Dispatch<React.SetStateAction<boolean>>,
  setIsLoadingMesh: React.Dispatch<React.SetStateAction<boolean>>,
  setMeshError: React.Dispatch<React.SetStateAction<string | null>>,
  setFolders: React.Dispatch<React.SetStateAction<StateFolder[]>>,
  frequencyRange: [number, number],
  meshResolution: number,
  noiseLevel: number,
  topPercentage: number,
  generateFrequencies: (start: number, end: number, points: number) => number[]
) => {
  if (!activeSnapshot) {
    setMeshError("Please select a state before computing mesh fits");
    return;
  }
  
  // Create append log function
  const appendLog = createAppendLogFunction(setComputationLogs);
  
  // Reset previous logs and show notification
  setComputationLogs([]);
  setShowNotification(true);
  setIsLoadingMesh(true);
  setMeshError(null);
  
  appendLog("Starting mesh computation...");
  appendLog(`Using state "${activeSnapshot.name}" as reference`);
  
  try {
    // First check if the server is running
    try {
      appendLog("Checking backend server connection...");
      const healthCheck = await fetch('http://localhost:8000/api/health');
      if (!healthCheck.ok) {
        throw new Error('Backend server is not responding');
      }
      appendLog("Connected to backend server successfully");
    } catch {
      appendLog("ERROR: Cannot connect to backend server");
      throw new Error('Cannot connect to backend server. Please ensure the server is running (python app/components/compute.py)');
    }

    const frequencies = generateFrequencies(frequencyRange[0], frequencyRange[1], 200);
    appendLog(`Generated ${frequencies.length} frequency points for analysis`);
    
    // Start computation request
    appendLog("Sending computation request to server...");
    appendLog(`Using mesh resolution: ${meshResolution}, noise level: ${noiseLevel}%`);
    
    const response = await fetch('http://localhost:8000/api/regression_mesh', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        reference_cell: {
          Rsh: activeSnapshot.parameters.Rsh,
          Ra: activeSnapshot.parameters.Ra,
          Ca: activeSnapshot.parameters.Ca,
          Rb: activeSnapshot.parameters.Rb,
          Cb: activeSnapshot.parameters.Cb,
          frequency_range: frequencies
        },
        mesh_resolution: meshResolution,
        noise_level: noiseLevel,
        top_percentage: topPercentage,
        use_ground_truth: true
      }),
    });

    // Start polling for progress while computation is running
    let pollingInterval: NodeJS.Timeout | undefined;
    try {
      pollingInterval = setInterval(async () => {
        try {
          const progressResponse = await fetch('http://localhost:8000/api/mesh_progress');
          if (progressResponse.ok) {
            const progressData = await progressResponse.json();
            if (progressData.logs && progressData.logs.length > 0) {
              // Add any new logs not already in our state
              setComputationLogs(prev => {
                const existingLogs = prev.map(log => log.substring(log.indexOf(']') + 2)); // Remove timestamp
                const newLogs: string[] = [];
                
                for (const log of progressData.logs) {
                  if (!existingLogs.includes(log)) {
                    const timestamp = new Date().toLocaleTimeString();
                    newLogs.push(`[${timestamp}] ${log}`);
                  }
                }
                
                return [...prev, ...newLogs];
              });
            }
          }
        } catch (error) {
          // Silently fail polling - the main computation is still running
          console.error("Error polling progress:", error);
        }
      }, 1000);
    } catch (error) {
      console.error("Error setting up polling:", error);
    }

    appendLog("Processing server response...");
    const responseData = await response.json();

    // Clear polling once response is received
    if (pollingInterval) clearInterval(pollingInterval);

    if (!response.ok) {
      appendLog(`ERROR: ${responseData.detail || 'Failed to compute regression mesh'}`);
      throw new Error(responseData.detail || 'Failed to compute regression mesh');
    }

    const resultCount = responseData.length;
    appendLog(`Received ${resultCount} results from server`);
    appendLog("Generating snapshots from results...");

    // Convert mesh points to snapshots
    const meshSnapshots: ModelSnapshot[] = (responseData as MeshResponsePoint[])
      .sort((a, b) => (a.resnorm ?? Infinity) - (b.resnorm ?? Infinity))
      .slice(0, 5)  // Take top 5 fits
      .map((point, index) => {
        appendLog(`Processing mesh fit ${index + 1}, resnorm: ${point.resnorm.toFixed(4)}`);
        
        // Generate impedance data for each frequency
        const data = frequencies.map(f => {
          const za_real = point.parameters.Ra / (1 + Math.pow(2 * Math.PI * f * point.parameters.Ra * point.parameters.Ca, 2));
          const za_imag = -2 * Math.PI * f * Math.pow(point.parameters.Ra, 2) * point.parameters.Ca / (1 + Math.pow(2 * Math.PI * f * point.parameters.Ra * point.parameters.Ca, 2));
          
          const zb_real = point.parameters.Rb / (1 + Math.pow(2 * Math.PI * f * point.parameters.Rb * point.parameters.Cb, 2));
          const zb_imag = -2 * Math.PI * f * Math.pow(point.parameters.Rb, 2) * point.parameters.Cb / (1 + Math.pow(2 * Math.PI * f * point.parameters.Rb * point.parameters.Cb, 2));
          
          // Add Za and Zb (in series) and then add Rsh
          const real = point.parameters.Rsh + za_real + zb_real;
          const imaginary = za_imag + zb_imag;
          
          return {
            real,
            imaginary,
            frequency: f,
            magnitude: Math.sqrt(real * real + imaginary * imaginary),
            phase: Math.atan2(imaginary, real) * (180 / Math.PI)
          };
        });

        // Calculate resnorm against the ground truth dataset
        const resnormValue = point.resnorm;

        return {
          id: `mesh-${index + 1}`,
          name: `Mesh Fit ${index + 1}`,
          timestamp: Date.now(),
          parameters: {
            Rsh: point.parameters.Rsh,
            Ra: point.parameters.Ra,
            Ca: point.parameters.Ca,
            Rb: point.parameters.Rb,
            Cb: point.parameters.Cb,
            R_blank: 10, // Default value
            frequency_range: frequencyRange
          },
          ter: point.parameters.Ra + point.parameters.Rb, // TER doesn't include Rsh
          data,
          color: `rgba(${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, ${point.alpha.toFixed(2)})`,
          isVisible: true,
          isMeshPoint: true,
          resnorm: resnormValue,
          parentId: activeSnapshot.id,
          opacity: point.alpha
        };
      });
    
    appendLog(`Created ${meshSnapshots.length} mesh fit snapshots`);

    // Create a new folder for mesh fits
    const meshFolder: StateFolder = {
      id: `mesh-folder-${Date.now()}`,
      name: `Mesh Fits for ${activeSnapshot.name}`,
      isExpanded: true,
      items: meshSnapshots,
      parentId: activeSnapshot.id,  // Link to parent state
      isVisible: true
    };

    // Remove any existing mesh folders for this state
    setFolders(prev => {
      const newFolders = prev.filter(f => f.parentId !== activeSnapshot.id);
      return [...newFolders, meshFolder];
    });
    
    appendLog(`Added mesh folder: "${meshFolder.name}"`);
    appendLog("Mesh computation completed successfully!");

    console.log('Added mesh folder with points:', meshSnapshots.length);
    console.log('First mesh point data:', meshSnapshots[0]?.data?.length || 0, 'points');

  } catch (error) {
    console.error('Error computing regression mesh:', error);
    appendLog(`Error: ${error instanceof Error ? error.message : 'An unexpected error occurred'}`);
    setMeshError(error instanceof Error ? error.message : 'An unexpected error occurred');
  } finally {
    setIsLoadingMesh(false);
    // Keep notification visible for a moment after completion
    setTimeout(() => {
      appendLog("Computation process finished");
    }, 1000);
  }
}; 