import { NextRequest, NextResponse } from 'next/server';
import { CircuitConfigService } from '../../../lib/circuitConfigService';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { configId, uiSettings } = body;

    if (!configId || !uiSettings) {
      return NextResponse.json({ error: 'Missing configId or uiSettings' }, { status: 400 });
    }

    const success = await CircuitConfigService.updateUISettings(configId, uiSettings);
    
    if (success) {
      return NextResponse.json({ success: true });
    } else {
      return NextResponse.json({ error: 'Failed to save UI settings' }, { status: 500 });
    }

  } catch (error) {
    console.error('API error saving UI settings:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}